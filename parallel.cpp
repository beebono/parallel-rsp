#ifdef DEBUG_JIT
#include "debug_rsp.hpp"
#else
#include "rsp_jit.hpp"
#endif
#include <stdint.h>

#include "m64p_plugin.h"
#include "rsp_1.1.h"
#include "prsp_rdp_defer.h"
#ifdef PRSP_SHADOW
#include "shadow.hpp"
#endif
#ifdef PRSP_NATIVE
#include "prsp_native.h"
#endif

#define RSP_PARALLEL_VERSION 0x0101
#define RSP_PLUGIN_API_VERSION 0x020000

namespace RSP
{
RSP_INFO rsp;
#ifdef DEBUG_JIT
RSP::CPU cpu;
#else
RSP::JIT::CPU cpu;
#endif
short MFC0_count[32];
int SP_STATUS_TIMEOUT;
} // namespace RSP

extern "C"
{
	// Hack entry point to use when loading savestates when we're tracing.
	void rsp_clear_registers()
	{
		memset(RSP::cpu.get_state().sr, 0, sizeof(uint32_t) * 32);
		memset(&RSP::cpu.get_state().cp2, 0, sizeof(RSP::cpu.get_state().cp2));
	}

#ifdef INTENSE_DEBUG
	// Need super-fast hash here.
	static uint64_t hash_imem(const uint8_t *data, size_t size)
	{
		uint64_t h = 0xcbf29ce484222325ull;
		size_t i;
		for (i = 0; i < size; i++)
			h = (h * 0x100000001b3ull) ^ data[i];
		return h;
	}

	void log_rsp_mem_parallel(void)
	{
		fprintf(stderr, "IMEM HASH: 0x%016llx\n", hash_imem(RSP::rsp.IMEM, 0x1000));
		fprintf(stderr, "DMEM HASH: 0x%016llx\n", hash_imem(RSP::rsp.DMEM, 0x1000));
	}
#endif

	extern "C++"
	{
	template <typename CPUType>
	static unsigned int run_rsp_task(CPUType &cpu, unsigned int cycles)
	{
		if (*RSP::rsp.SP_STATUS_REG & (SP_STATUS_HALT | SP_STATUS_BROKE))
			return 0;

		// We don't know if Mupen from the outside invalidated our IMEM.
		cpu.invalidate_imem();

		// Run CPU until we either break or we need to fire an IRQ.
		cpu.get_state().pc = *RSP::rsp.SP_PC_REG & 0xfff;

#ifdef INTENSE_DEBUG
		fprintf(stderr, "RUN TASK: %u\n", cpu.get_state().pc);
		log_rsp_mem_parallel();
#endif

		for (auto &count : RSP::MFC0_count)
			count = 0;

		while (!(*RSP::rsp.SP_STATUS_REG & SP_STATUS_HALT))
		{
			auto mode = cpu.run();
			if (mode == RSP::MODE_CHECK_FLAGS && (*cpu.get_state().cp0.irq & 1))
				break;
		}

		PRSP_RDP_DEFER_FLUSH(PRSP_RDP_FLUSH_RUN_END);
		*RSP::rsp.SP_PC_REG = 0x04001000 | (cpu.get_state().pc & 0xffc);

		// RSP_MFC0 forces HALT when the ucode polls SP_STATUS for too long (it is waiting for the CPU). That is not
		// the same as the ucode halting itself without BREAK (below): after a polling time-out the HALT bit has to be
		// cleared again so that the core re-enters the RSP once the CPU has run (the CXD4 WAIT_FOR_CPU_HOST path),
		// otherwise a task that waits on the CPU (Gauntlet Legends: graphics ucode idles at 0xfc8) is never resumed.
		bool polling_timed_out = false;
		for (auto count : RSP::MFC0_count)
			if (count >= RSP::SP_STATUS_TIMEOUT)
				polling_timed_out = true;

		// From CXD4.
		if (*RSP::rsp.SP_STATUS_REG & SP_STATUS_BROKE)
			return cycles;
		else if (*cpu.get_state().cp0.irq & 1)
			RSP::rsp.CheckInterrupts();
		else if ((*RSP::rsp.SP_STATUS_REG & SP_STATUS_HALT) && !polling_timed_out)
			return cycles;
		else if (*RSP::rsp.SP_SEMAPHORE_REG != 0) // Semaphore lock fixes.
		{
		}
		else
			RSP::SP_STATUS_TIMEOUT = 16; // From now on, wait 16 times, not 0x7fff

		// CPU restarts with the correct SIGs.
		*RSP::rsp.SP_STATUS_REG &= ~SP_STATUS_HALT;

		return cycles;
	}
	} // extern "C++"

#ifdef PRSP_SHADOW
	unsigned do_rsp_cycles_shadow(void *c, unsigned cycles)
	{
		return run_rsp_task(*static_cast<RSP::JIT::CPU *>(c), cycles);
	}
#endif

	EXPORT unsigned int CALL parallelRSPDoRspCycles(unsigned int cycles)
	{
#ifdef PRSP_SHADOW
		if (RSP::Shadow::enabled() && !(*RSP::rsp.SP_STATUS_REG & (SP_STATUS_HALT | SP_STATUS_BROKE)))
		{
			RSP::Shadow::begin();
			unsigned int r = run_rsp_task(RSP::cpu, cycles);
			RSP::Shadow::end();
			return r;
		}
#endif
		return run_rsp_task(RSP::cpu, cycles);
	}

	EXPORT m64p_error CALL parallelRSPPluginGetVersion(m64p_plugin_type *PluginType, int *PluginVersion,
	                                                   int *APIVersion, const char **PluginNamePtr, int *Capabilities)
	{
		/* set version info */
		if (PluginType != NULL)
			*PluginType = M64PLUGIN_RSP;

		if (PluginVersion != NULL)
			*PluginVersion = RSP_PARALLEL_VERSION;

		if (APIVersion != NULL)
			*APIVersion = RSP_PLUGIN_API_VERSION;

		if (Capabilities != NULL)
			*Capabilities = 0;

		return M64ERR_SUCCESS;
	}

	EXPORT void CALL parallelRSPRomClosed(void)
	{
		*RSP::rsp.SP_PC_REG = 0x00000000;
	}

	EXPORT void CALL parallelRSPInitiateRSP(RSP_INFO Rsp_Info, unsigned int *CycleCount)
	{
		if (CycleCount)
			*CycleCount = 0;

		if (Rsp_Info.DMEM == Rsp_Info.IMEM) /* usually dummy RSP data for testing */
			return; /* DMA is not executed just because plugin initiates. */

		RSP::rsp = Rsp_Info;
		*RSP::rsp.SP_PC_REG = 0x04001000 & 0x00000FFF; /* task init bug on Mupen64 */

		auto **cr = RSP::cpu.get_state().cp0.cr;
		cr[0x0] = RSP::rsp.SP_MEM_ADDR_REG;
		cr[0x1] = RSP::rsp.SP_DRAM_ADDR_REG;
		cr[0x2] = RSP::rsp.SP_RD_LEN_REG;
		cr[0x3] = RSP::rsp.SP_WR_LEN_REG;
		cr[0x4] = RSP::rsp.SP_STATUS_REG;
		cr[0x5] = RSP::rsp.SP_DMA_FULL_REG;
		cr[0x6] = RSP::rsp.SP_DMA_BUSY_REG;
		cr[0x7] = RSP::rsp.SP_SEMAPHORE_REG;
		cr[0x8] = RSP::rsp.DPC_START_REG;
		cr[0x9] = RSP::rsp.DPC_END_REG;
		cr[0xA] = RSP::rsp.DPC_CURRENT_REG;
		cr[0xB] = RSP::rsp.DPC_STATUS_REG;
		cr[0xC] = RSP::rsp.DPC_CLOCK_REG;
		cr[0xD] = RSP::rsp.DPC_BUFBUSY_REG;
		cr[0xE] = RSP::rsp.DPC_PIPEBUSY_REG;
		cr[0xF] = RSP::rsp.DPC_TMEM_REG;

		*cr[RSP::CP0_REGISTER_SP_STATUS] = SP_STATUS_HALT;
		RSP::cpu.get_state().cp0.irq = RSP::rsp.MI_INTR_REG;

		// From CXD4.
		RSP::SP_STATUS_TIMEOUT = 0x7fff;

		RSP::cpu.set_dmem(reinterpret_cast<uint32_t *>(Rsp_Info.DMEM));
		RSP::cpu.set_imem(reinterpret_cast<uint32_t *>(Rsp_Info.IMEM));
		RSP::cpu.set_rdram(reinterpret_cast<uint32_t *>(Rsp_Info.RDRAM));
#ifdef PRSP_SHADOW
		RSP::Shadow::init();
#endif
	}
}

#ifdef PRSP_NATIVE
// ---- native-routine hook (prsp_native.h) ----
extern "C"
{
	int RSP_MTC0(RSP::CPUState *rsp, unsigned rd, unsigned rt);
	int RSP_MFC0(RSP::CPUState *rsp, unsigned rt, unsigned rd);
	void prsp_native_cp0_host(const PrspNativeHost *host);

	void prsp_native_install(const PrspNativeHost *host)
	{
		prsp_native_cp0_host(host);
		RSP::cpu.native_install(host);
	}

	void prsp_native_set_pcs(const uint16_t *entries, unsigned n_entries, const uint16_t *splits, unsigned n_splits)
	{
		RSP::cpu.native_set_pcs(entries, n_entries, splits, n_splits);
	}

	void prsp_native_view(PrspNativeView *v)
	{
		RSP::CPUState &s = RSP::cpu.get_state();
		v->sr = s.sr;
		v->vr = reinterpret_cast<uint16_t(*)[8]>(&s.cp2.regs[0]);
		v->acc = s.cp2.acc.e;
		v->flags = reinterpret_cast<uint16_t(*)[16]>(&s.cp2.flags[0]);
		v->div_out = &s.cp2.div_out;
		v->div_in = &s.cp2.div_in;
		v->dp_flag = &s.cp2.dp_flag;
		v->dmem = s.dmem;
		v->imem = s.imem;
		v->rdram = s.rdram;
		for (int i = 0; i < 16; i++)
			v->cr[i] = s.cp0.cr[i];
		v->irq = s.cp0.irq;
		v->mfc0_count = RSP::MFC0_count;
		v->pc = &s.pc;
		v->has_delay_slot = &s.has_delay_slot;
		v->branch_target = &s.branch_target;
		v->dirty_blocks = &s.dirty_blocks;
	}

	void prsp_native_save(PrspNativeSnap *o)
	{
		RSP::CPUState &s = RSP::cpu.get_state();
		o->magic = PRSP_NATIVE_SNAP_MAGIC;
		o->size = sizeof(*o);
		o->pc = s.pc;
		o->has_delay_slot = s.has_delay_slot;
		o->branch_target = s.branch_target;
		o->irq = *s.cp0.irq;
		o->dirty_blocks = s.dirty_blocks;
		o->sp_status_timeout = RSP::SP_STATUS_TIMEOUT;
		memcpy(o->mfc0_count, RSP::MFC0_count, sizeof(o->mfc0_count));
		for (int i = 0; i < 16; i++)
			o->cr[i] = *s.cp0.cr[i];
		memcpy(o->sr, s.sr, sizeof(o->sr));
		for (int i = 0; i < 32; i++)
			memcpy(o->vr[i], s.cp2.regs[i].e, 16);
		for (int i = 0; i < 3; i++)
			memcpy(o->flags[i], s.cp2.flags[i].e, 32);
		memcpy(o->acc, s.cp2.acc.e, 48);
		o->div_out = s.cp2.div_out;
		o->div_in = s.cp2.div_in;
		o->dp_flag = s.cp2.dp_flag;
		o->pad_[0] = o->pad_[1] = o->pad_[2] = 0;
		memcpy(o->dmem, s.dmem, 4096);
		memcpy(o->imem, s.imem, 4096);
	}

	void prsp_native_load(const PrspNativeSnap *o)
	{
		RSP::CPUState &s = RSP::cpu.get_state();
		s.pc = o->pc;
		s.has_delay_slot = o->has_delay_slot;
		s.branch_target = o->branch_target;
		*s.cp0.irq = o->irq;
		s.dirty_blocks = o->dirty_blocks;
		RSP::SP_STATUS_TIMEOUT = o->sp_status_timeout;
		memcpy(RSP::MFC0_count, o->mfc0_count, sizeof(o->mfc0_count));
		for (int i = 0; i < 16; i++)
			*s.cp0.cr[i] = o->cr[i];
		memcpy(s.sr, o->sr, sizeof(o->sr));
		for (int i = 0; i < 32; i++)
			memcpy(s.cp2.regs[i].e, o->vr[i], 16);
		for (int i = 0; i < 3; i++)
			memcpy(s.cp2.flags[i].e, o->flags[i], 32);
		memcpy(s.cp2.acc.e, o->acc, 48);
		s.cp2.div_out = o->div_out;
		s.cp2.div_in = o->div_in;
		s.cp2.dp_flag = o->dp_flag;
		memcpy(s.dmem, o->dmem, 4096);
		memcpy(s.imem, o->imem, 4096);
	}

	void prsp_native_resync(void)
	{
		RSP::cpu.native_resync();
	}

	int prsp_native_oracle(unsigned entry_pc, unsigned stop_pc)
	{
		return RSP::cpu.native_oracle(entry_pc, stop_pc);
	}

	int prsp_native_mtc0(unsigned rd, uint32_t value)
	{
		RSP::CPUState &s = RSP::cpu.get_state();
		const uint32_t keep = s.sr[32];
		s.sr[32] = value;
		const int mode = RSP_MTC0(&s, rd, 32);
		s.sr[32] = keep;
		return mode;
	}

	uint32_t prsp_native_mfc0(unsigned rd, unsigned rt, int *mode)
	{
		RSP::CPUState &s = RSP::cpu.get_state();
		const int m = RSP_MFC0(&s, rt, rd);
		if (mode)
			*mode = m;
		return *s.cp0.cr[rd & 15];
	}
}
#endif
