#ifdef PRSP_SHADOW
#include "shadow.hpp"
#include "rsp_jit.hpp"
#include <new>
#include <stdarg.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>

#include "rsp_1.1.h"
#include "m64p_plugin.h"

extern "C" unsigned do_rsp_cycles_shadow(void *cpu, unsigned cycles); // parallel.cpp

namespace RSP
{
extern RSP_INFO rsp;
extern JIT::CPU cpu;
extern short MFC0_count[32];
extern int SP_STATUS_TIMEOUT;

namespace Shadow
{
namespace
{
const size_t RDRAM_BYTES = 0x800000;

struct RdpRec
{
	uint32_t cur, end, xbus;
	uint64_t hash;
	uint32_t post[16]; // CP0 registers after the RDP callback (main run), applied to the shadow run
};

struct DmaRec
{
	uint32_t addr, len;
	uint64_t hash;
	bool operator!=(const DmaRec &o) const
	{
		return addr != o.addr || len != o.len || hash != o.hash;
	}
};

struct Stats
{
	uint64_t tasks = 0, tasks_diff = 0;
	uint64_t rdp_lists = 0, rdp_diff = 0, dma_writes = 0, dma_diff = 0;
	uint64_t div_ops = 0, cfc2_ops = 0, imem_blocks = 0, imem_changed = 0;
	uint64_t shadow_imem_blocks = 0, shadow_imem_changed = 0;
	uint64_t checks = 0, stale_after_invalidate = 0, stale_not_dirty = 0;
	uint64_t diff_regs = 0, diff_cp2 = 0, diff_mem = 0, diff_cp0 = 0, diff_log = 0;
	unsigned reports = 0;
};

struct Ctx
{
	bool on = false, old_mode = false;
	Stats st;
	std::string out;

	// shadow machine
	JIT::CPU *scpu = nullptr;
	RSP_INFO sinfo;
	alignas(64) uint32_t sdmem[1024], simem[1024];
	uint32_t *srdram = nullptr;
	uint32_t scr[16], sirq;
	// start snapshot of the real run
	uint32_t pre_cr[16], pre_irq, pre_pc_reg;
	uint32_t pre_sr[33];
	CP2 pre_cp2;
	uint32_t pre_pc, pre_delay, pre_target;
	int pre_timeout;
	alignas(64) uint32_t pre_dmem[1024], pre_imem[1024];
	// logs
	bool in_shadow = false;
	std::vector<RdpRec> rdp_main, rdp_shadow;
	std::vector<DmaRec> dma_main, dma_shadow;
	size_t rdp_replay = 0;
	void (*real_process_rdp)(void) = nullptr;
	RSP_INFO saved_info;
	uint32_t shadow_pc_reg = 0;
};
Ctx *C;

uint64_t fnv(const void *p, size_t n, uint64_t h = 0xcbf29ce484222325ull)
{
	const uint8_t *b = static_cast<const uint8_t *>(p);
	for (size_t i = 0; i < n; i++)
		h = (h ^ b[i]) * 0x100000001b3ull;
	return h;
}

void report(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
void report(const char *fmt, ...)
{
	if (C->st.reports++ >= 30)
		return;
	char buf[512];
	va_list ap;
	va_start(ap, fmt);
	vsnprintf(buf, sizeof(buf), fmt, ap);
	va_end(ap);
	fprintf(stderr, "prsp shadow: task %llu: %s\n", (unsigned long long)C->st.tasks, buf);
}

RdpRec make_rdp_rec(CPUState &s, uint32_t *rdram, uint32_t *dmem)
{
	RdpRec r;
	r.cur = *s.cp0.cr[CP0_REGISTER_CMD_CURRENT];
	r.end = *s.cp0.cr[CP0_REGISTER_CMD_END];
	r.xbus = *s.cp0.cr[CP0_REGISTER_CMD_STATUS] & 1;
	r.hash = 0;
	if (r.end > r.cur && r.end - r.cur <= (1u << 20))
	{
		for (uint32_t a = r.cur; a < r.end; a += 4)
		{
			uint32_t w = r.xbus ? dmem[(a & 0xfff) >> 2] : rdram[(a & 0x7ffffc) >> 2];
			r.hash = fnv(&w, 4, r.hash);
		}
	}
	memset(r.post, 0, sizeof(r.post));
	return r;
}

void rdp_hook_main()
{
	CPUState &s = cpu.get_state();
	RdpRec r = make_rdp_rec(s, s.rdram, s.dmem);
	if (C->real_process_rdp)
		C->real_process_rdp();
	for (int i = 0; i < 16; i++)
		r.post[i] = *s.cp0.cr[i];
	C->rdp_main.push_back(r);
}

void rdp_hook_shadow()
{
	CPUState &s = C->scpu->get_state();
	RdpRec r = make_rdp_rec(s, s.rdram, s.dmem);
	size_t i = C->rdp_replay++;
	C->st.rdp_lists++;
	if (i >= C->rdp_main.size() || r.cur != C->rdp_main[i].cur || r.end != C->rdp_main[i].end ||
	    r.xbus != C->rdp_main[i].xbus || r.hash != C->rdp_main[i].hash)
	{
		C->st.rdp_diff++;
		report("RDP list %zu differs (shadow %08x-%08x xbus %u hash %016llx)", i, r.cur, r.end, r.xbus,
		       (unsigned long long)r.hash);
	}
	if (i < C->rdp_main.size())
		for (int k = 8; k < 16; k++)
			*s.cp0.cr[k] = C->rdp_main[i].post[k]; // DPC registers as the real RDP left them
	C->rdp_shadow.push_back(r);
}

void make_shadow()
{
	void *mem = nullptr;
	if (posix_memalign(&mem, 64, (sizeof(JIT::CPU) + 63) & ~size_t(63)))
		abort();
	C->scpu = new (mem) JIT::CPU;
#ifdef PRSP_VU_FAST
	if (C->old_mode)
		C->scpu->set_vu_fast(false);
#endif
#ifdef PRSP_VU_FAST
	C->scpu->set_div_inline(false);
#endif
	C->scpu->set_imem_skip(false);
	C->srdram = static_cast<uint32_t *>(malloc(RDRAM_BYTES));
	CPUState &s = C->scpu->get_state();
	C->scpu->set_dmem(C->sdmem);
	C->scpu->set_imem(C->simem);
	C->scpu->set_rdram(C->srdram);
	for (int i = 0; i < 16; i++)
		s.cp0.cr[i] = &C->scr[i];
	s.cp0.irq = &C->sirq;
	// RSP_INFO of the shadow run: the same plugin API pointing at the shadow registers.
	RSP_INFO &r = C->sinfo;
	r = rsp;
	r.DMEM = reinterpret_cast<unsigned char *>(C->sdmem);
	r.IMEM = reinterpret_cast<unsigned char *>(C->simem);
	r.RDRAM = reinterpret_cast<unsigned char *>(C->srdram);
	r.MI_INTR_REG = &C->sirq;
	r.SP_MEM_ADDR_REG = &C->scr[0];
	r.SP_DRAM_ADDR_REG = &C->scr[1];
	r.SP_RD_LEN_REG = &C->scr[2];
	r.SP_WR_LEN_REG = &C->scr[3];
	r.SP_STATUS_REG = &C->scr[4];
	r.SP_DMA_FULL_REG = &C->scr[5];
	r.SP_DMA_BUSY_REG = &C->scr[6];
	r.SP_SEMAPHORE_REG = &C->scr[7];
	r.DPC_START_REG = &C->scr[8];
	r.DPC_END_REG = &C->scr[9];
	r.DPC_CURRENT_REG = &C->scr[10];
	r.DPC_STATUS_REG = &C->scr[11];
	r.DPC_CLOCK_REG = &C->scr[12];
	r.DPC_BUFBUSY_REG = &C->scr[13];
	r.DPC_PIPEBUSY_REG = &C->scr[14];
	r.DPC_TMEM_REG = &C->scr[15];
	r.SP_PC_REG = &C->shadow_pc_reg;
	r.CheckInterrupts = [] {};
	r.ProcessRdpList = rdp_hook_shadow;
}

void write_summary()
{
	if (!C)
		return;
	Stats s = C->st;
	if (C->scpu)
	{
		s.checks += C->scpu->sh_checks;
		s.stale_after_invalidate += C->scpu->sh_stale_after_invalidate;
		s.stale_not_dirty += C->scpu->sh_stale_not_dirty;
	}
	s.checks += cpu.sh_checks;
	s.stale_after_invalidate += cpu.sh_stale_after_invalidate;
	s.stale_not_dirty += cpu.sh_stale_not_dirty;
	FILE *f = nullptr;
	const char *p = getenv("PRSP_SHADOW_OUT");
	if (p && *p)
		f = fopen(p, "w");
	if (!f)
		f = stderr;
	fprintf(f, "prsp shadow summary (shadow mode %s)\n", C->old_mode ? "old: helper-call code generation, IMEM always dirty" : "prev: no inlined VRCP/VRSQ/CFC2, IMEM always dirty");
	fprintf(f, "tasks compared %llu, tasks with a difference %llu\n", (unsigned long long)s.tasks, (unsigned long long)s.tasks_diff);
	fprintf(f, "  differences by kind: registers %llu, cp2 %llu, dmem/imem %llu, cp0 %llu, logs %llu\n", (unsigned long long)s.diff_regs,
	        (unsigned long long)s.diff_cp2, (unsigned long long)s.diff_mem, (unsigned long long)s.diff_cp0, (unsigned long long)s.diff_log);
	fprintf(f, "RDP lists compared %llu (differences %llu), RDRAM DMA writes compared %llu (differences %llu)\n", (unsigned long long)s.rdp_lists,
	        (unsigned long long)s.rdp_diff, (unsigned long long)s.dma_writes, (unsigned long long)s.dma_diff);
	fprintf(f, "real run: inlined VRCP/VRCPL/VRSQ/VRSQL executed %llu, inlined CFC2 executed %llu\n", (unsigned long long)s.div_ops,
	        (unsigned long long)s.cfc2_ops);
	fprintf(f, "real run: IMEM blocks covered by SP DMA %llu, changed %llu (skipped %llu); shadow run: covered %llu, changed %llu\n",
	        (unsigned long long)s.imem_blocks, (unsigned long long)s.imem_changed, (unsigned long long)(s.imem_blocks - s.imem_changed),
	        (unsigned long long)s.shadow_imem_blocks, (unsigned long long)s.shadow_imem_changed);
	fprintf(f, "compiled-region checks %llu, stale after invalidate_code %llu, stale but block not dirty %llu\n", (unsigned long long)s.checks,
	        (unsigned long long)s.stale_after_invalidate, (unsigned long long)s.stale_not_dirty);
	if (f != stderr)
		fclose(f);
}

struct Finalizer
{
	~Finalizer()
	{
		write_summary();
	}
} finalizer;
} // namespace

bool enabled()
{
	return C && C->on;
}

void init()
{
	if (!C)
		C = new Ctx;
	const char *m = getenv("PRSP_SHADOW_MODE");
	C->old_mode = m && !strcmp(m, "old");
	C->on = true;
	C->real_process_rdp = rsp.ProcessRdpList;
	rsp.ProcessRdpList = rdp_hook_main;
	if (!C->scpu)
		make_shadow();
}

void begin()
{
	CPUState &s = cpu.get_state();
	for (int i = 0; i < 16; i++)
		C->pre_cr[i] = *s.cp0.cr[i];
	C->pre_irq = *s.cp0.irq;
	C->pre_pc_reg = *rsp.SP_PC_REG;
	memcpy(C->pre_sr, s.sr, sizeof(C->pre_sr));
	C->pre_cp2 = s.cp2;
	C->pre_pc = s.pc;
	C->pre_delay = s.has_delay_slot;
	C->pre_target = s.branch_target;
	C->pre_timeout = SP_STATUS_TIMEOUT;
	memcpy(C->pre_dmem, s.dmem, 4096);
	memcpy(C->pre_imem, s.imem, 4096);
	memcpy(C->srdram, s.rdram, RDRAM_BYTES);
	C->rdp_main.clear();
	C->dma_main.clear();
	C->rdp_shadow.clear();
	C->dma_shadow.clear();
	// counters of the real run for this task
	s.sh_div_ops = s.sh_cfc2_ops = s.sh_imem_blocks = s.sh_imem_changed = 0;
}

void end()
{
	CPUState &m = cpu.get_state();
	Stats &st = C->st;
	st.div_ops += m.sh_div_ops;
	st.cfc2_ops += m.sh_cfc2_ops;
	st.imem_blocks += m.sh_imem_blocks;
	st.imem_changed += m.sh_imem_changed;

	// Final state of the real run (the scratch copies below keep it for the comparison).
	uint32_t main_cr[16];
	for (int i = 0; i < 16; i++)
		main_cr[i] = *m.cp0.cr[i];
	const uint32_t main_pc_reg = *rsp.SP_PC_REG, main_irq = *m.cp0.irq;
	const int main_timeout = SP_STATUS_TIMEOUT;
	short main_count[32];
	memcpy(main_count, MFC0_count, sizeof(main_count));

	// Start state for the shadow run.
	CPUState &s = C->scpu->get_state();
	for (int i = 0; i < 16; i++)
		C->scr[i] = C->pre_cr[i];
	C->sirq = C->pre_irq;
	C->shadow_pc_reg = C->pre_pc_reg;
	memcpy(s.sr, C->pre_sr, sizeof(C->pre_sr));
	s.cp2 = C->pre_cp2;
	s.pc = C->pre_pc;
	s.has_delay_slot = C->pre_delay;
	s.branch_target = C->pre_target;
	memcpy(C->sdmem, C->pre_dmem, 4096);
	memcpy(C->simem, C->pre_imem, 4096);
	s.sh_div_ops = s.sh_cfc2_ops = s.sh_imem_blocks = s.sh_imem_changed = 0;
	C->rdp_replay = 0;

	// Swap in the shadow plugin info, run, swap back.
	RSP_INFO saved = rsp;
	rsp = C->sinfo;
	SP_STATUS_TIMEOUT = C->pre_timeout;
	C->in_shadow = true;
	do_rsp_cycles_shadow(C->scpu, 0);
	C->in_shadow = false;
	rsp = saved;
	SP_STATUS_TIMEOUT = main_timeout;
	memcpy(MFC0_count, main_count, sizeof(main_count));

	st.shadow_imem_blocks += s.sh_imem_blocks;
	st.shadow_imem_changed += s.sh_imem_changed;
	st.tasks++;

	// Compare.
	bool diff = false;
	auto fail = [&](uint64_t &counter, const char *what) {
		counter++;
		diff = true;
		report("%s", what);
	};
	char buf[160];
	for (int i = 0; i < 16; i++)
		if (main_cr[i] != C->scr[i])
		{
			snprintf(buf, sizeof(buf), "CP0 register %d: %08x vs %08x", i, main_cr[i], C->scr[i]);
			fail(st.diff_cp0, buf);
		}
	if (main_pc_reg != C->shadow_pc_reg)
	{
		snprintf(buf, sizeof(buf), "SP_PC: %08x vs %08x", main_pc_reg, C->shadow_pc_reg);
		fail(st.diff_cp0, buf);
	}
	if ((main_irq & 1) != (C->sirq & 1))
		fail(st.diff_cp0, "interrupt flag differs");
	for (int i = 0; i < 32; i++)
		if (m.sr[i] != s.sr[i])
		{
			snprintf(buf, sizeof(buf), "sr[%d]: %08x vs %08x", i, m.sr[i], s.sr[i]);
			fail(st.diff_regs, buf);
		}
	if (m.pc != s.pc || m.has_delay_slot != s.has_delay_slot || m.branch_target != s.branch_target)
		fail(st.diff_regs, "pc / delay slot differs");
	for (int r = 0; r < 32; r++)
		if (memcmp(m.cp2.regs[r].e, s.cp2.regs[r].e, 16))
		{
			snprintf(buf, sizeof(buf), "vector register v%d differs", r);
			fail(st.diff_cp2, buf);
		}
	for (int r = 0; r < 3; r++)
		if (memcmp(m.cp2.flags[r].e, s.cp2.flags[r].e, 32))
			fail(st.diff_cp2, "flags differ");
	if (memcmp(m.cp2.acc.e, s.cp2.acc.e, 48))
		fail(st.diff_cp2, "accumulator differs");
	if (m.cp2.div_out != s.cp2.div_out || m.cp2.div_in != s.cp2.div_in || m.cp2.dp_flag != s.cp2.dp_flag)
		fail(st.diff_cp2, "divide state differs");
	if (memcmp(m.dmem, C->sdmem, 4096))
		fail(st.diff_mem, "DMEM differs");
	if (memcmp(m.imem, C->simem, 4096))
		fail(st.diff_mem, "IMEM differs");
	if (C->rdp_main.size() != C->rdp_shadow.size())
		fail(st.diff_log, "number of RDP lists differs");
	if (C->dma_main.size() != C->dma_shadow.size())
		fail(st.diff_log, "number of RDRAM DMA writes differs");
	else
		for (size_t i = 0; i < C->dma_main.size(); i++)
		{
			st.dma_writes++;
			if (C->dma_main[i] != C->dma_shadow[i])
			{
				st.dma_diff++;
				fail(st.diff_log, "RDRAM DMA write differs");
			}
		}
	if (diff)
		st.tasks_diff++;
	static time_t last;
	time_t now = time(nullptr);
	if (now != last && (now % 5 == 0 || diff))
	{
		last = now;
		write_summary();
	}
}
} // namespace Shadow
} // namespace RSP

// Called by the SP DMA write after it stored `bytes` at the RDRAM byte address; logs address, length and a hash of the bytes written.
extern "C" void prsp_shadow_dma_write(const void *state, unsigned addr, unsigned bytes)
{
	using namespace RSP::Shadow;
	if (!C || !C->on)
		return;
	const RSP::CPUState *s = static_cast<const RSP::CPUState *>(state);
	DmaRec r;
	r.addr = addr;
	r.len = bytes;
	r.hash = fnv(reinterpret_cast<const uint8_t *>(s->rdram) + addr, bytes);
	(C->in_shadow ? C->dma_shadow : C->dma_main).push_back(r);
}
#endif
