#include "../state.hpp"
#include "../prsp_hooks.h"
#include <string.h>

#ifdef PARALLEL_INTEGRATION
#include "../rsp_1.1.h"
#include "m64p_plugin.h"
namespace RSP
{
extern RSP_INFO rsp;
extern short MFC0_count[32];
extern int SP_STATUS_TIMEOUT;
} // namespace RSP
#endif

using namespace RSP;

extern "C"
{

#ifdef INTENSE_DEBUG
	void log_rsp_mem_parallel(void);
#endif
#ifdef PRSP_SHADOW
	void prsp_shadow_after_imem_write(RSP::CPUState *state, unsigned off, unsigned bytes);
	void prsp_shadow_dma_write(const void *state, unsigned rdram_byte_addr, unsigned bytes);
#endif

	int RSP_MFC0(RSP::CPUState *rsp, unsigned rt, unsigned rd)
	{
		rd &= 15;
		uint32_t res = *rsp->cp0.cr[rd];
		if (rt)
			rsp->sr[rt] = res;

#ifdef PARALLEL_INTEGRATION
		if (rd == CP0_REGISTER_SP_STATUS)
		{
			// Might be waiting for the CPU to set a signal bit on the STATUS register. Increment timeout
			RSP::MFC0_count[rt] += 1;
			if (RSP::MFC0_count[rt] >= RSP::SP_STATUS_TIMEOUT)
			{
				*RSP::rsp.SP_STATUS_REG |= SP_STATUS_HALT;
				return MODE_CHECK_FLAGS;
			}
		}
#endif

#if 0 // FIXME: this is broken with upstream mupen64plus-core
		if (rd == CP0_REGISTER_SP_SEMAPHORE)
		{
			if (*rsp->cp0.cr[CP0_REGISTER_SP_SEMAPHORE])
			{
#ifdef PARALLEL_INTEGRATION
				RSP::MFC0_count[rt] += 8; // Almost certainly waiting on the CPU. Timeout faster.
				if (RSP::MFC0_count[rt] >= RSP::SP_STATUS_TIMEOUT)
				{
					*RSP::rsp.SP_STATUS_REG |= SP_STATUS_HALT;
					return MODE_CHECK_FLAGS;
				}
#endif
			}
			else
				*rsp->cp0.cr[CP0_REGISTER_SP_SEMAPHORE] = 1;
		}
#endif

		//if (rd == 4) // SP_STATUS_REG
		//   fprintf(stderr, "READING STATUS REG!\n");

		return MODE_CONTINUE;
	}

#define RSP_HANDLE_STATUS_WRITE(flag) \
	switch (rt & (SP_SET_##flag | SP_CLR_##flag)) \
	{ \
		case SP_SET_##flag: status |= SP_STATUS_##flag; break; \
		case SP_CLR_##flag: status &= ~SP_STATUS_##flag; break; \
		default: break; \
	}

	static inline int rsp_status_write(RSP::CPUState *rsp, uint32_t rt)
	{
		//fprintf(stderr, "Writing 0x%x to status reg!\n", rt);

		uint32_t status = *rsp->cp0.cr[CP0_REGISTER_SP_STATUS];

		RSP_HANDLE_STATUS_WRITE(HALT)
		RSP_HANDLE_STATUS_WRITE(SSTEP)
		RSP_HANDLE_STATUS_WRITE(INTR_BREAK)
		RSP_HANDLE_STATUS_WRITE(SIG0)
		RSP_HANDLE_STATUS_WRITE(SIG1)
		RSP_HANDLE_STATUS_WRITE(SIG2)
		RSP_HANDLE_STATUS_WRITE(SIG3)
		RSP_HANDLE_STATUS_WRITE(SIG4)
		RSP_HANDLE_STATUS_WRITE(SIG5)
		RSP_HANDLE_STATUS_WRITE(SIG6)
		RSP_HANDLE_STATUS_WRITE(SIG7)

		switch (rt & (SP_SET_INTR | SP_CLR_INTR))
		{
			case SP_SET_INTR: *rsp->cp0.irq |= 1; break;
			case SP_CLR_INTR: *rsp->cp0.irq &= ~1; break;
			default: break;
		}

		if (rt & SP_CLR_BROKE)
			status &= ~SP_STATUS_BROKE;

		*rsp->cp0.cr[CP0_REGISTER_SP_STATUS] = status;
		return ((*rsp->cp0.irq & 1) || (status & SP_STATUS_HALT)) ? MODE_CHECK_FLAGS : MODE_CONTINUE;
	}

#ifdef PARALLEL_INTEGRATION
	// The only way the RSP itself can modify IMEM is an SP DMA read (stores address DMEM only). Copies `bytes` (a multiple
	// of 4) from `src` to IMEM at byte offset `off` (off + bytes <= 0x1000) and marks the code blocks dirty. A block is
	// marked (together with its predecessor, because a compiled region starting there can extend into it) only when the
	// copy actually changes one of its bytes: microcode that is reloaded unchanged (the boot ucode copies the same text to
	// IMEM for every task) then keeps its compiled blocks.
	static inline void copy_to_imem(RSP::CPUState *rsp, unsigned off, const uint32_t *src, unsigned bytes)
	{
		if (!bytes)
			return;
		const unsigned end = off + bytes;
		for (unsigned block = off / CODE_BLOCK_SIZE; block <= (end - 1) / CODE_BLOCK_SIZE; block++)
		{
			const unsigned b0 = block * CODE_BLOCK_SIZE > off ? block * CODE_BLOCK_SIZE : off;
			const unsigned b1 = (block + 1) * CODE_BLOCK_SIZE < end ? (block + 1) * CODE_BLOCK_SIZE : end;
			uint32_t *dst = rsp->imem + (b0 >> 2);
			const uint32_t *from = src + ((b0 - off) >> 2);
#ifdef PRSP_SHADOW
			rsp->sh_imem_blocks++;
#endif
			PRSP_HOOK_IMEM_DMA(1, memcmp(dst, from, b1 - b0) != 0);
			if (rsp->imem_skip_identical && !memcmp(dst, from, b1 - b0))
				continue;
#ifdef PRSP_SHADOW
			if (memcmp(dst, from, b1 - b0))
				rsp->sh_imem_changed++;
#endif
			rsp->dirty_blocks |= (0x3 << block) >> 1;
			memcpy(dst, from, b1 - b0);
		}
#ifdef PRSP_SHADOW
		prsp_shadow_after_imem_write(rsp, off, bytes);
#endif
	}

	static int rsp_dma_read(RSP::CPUState *rsp)
	{
		uint32_t length_reg = *rsp->cp0.cr[CP0_REGISTER_DMA_READ_LENGTH];
		uint32_t length = (length_reg & 0xFFF) + 1;
		uint32_t skip = (length_reg >> 20) & 0xFFF;
		unsigned count = (length_reg >> 12) & 0xFF;

		// Force alignment.
		length = (length + 0x7) & ~0x7;
		*rsp->cp0.cr[CP0_REGISTER_DMA_CACHE] &= ~0x3;
		*rsp->cp0.cr[CP0_REGISTER_DMA_DRAM] &= ~0x7;

		// Check length.
		if (((*rsp->cp0.cr[CP0_REGISTER_DMA_CACHE] & 0xFFF) + length) > 0x1000)
			length = 0x1000 - (*rsp->cp0.cr[CP0_REGISTER_DMA_CACHE] & 0xFFF);

		unsigned i = 0;
		uint32_t source = *rsp->cp0.cr[CP0_REGISTER_DMA_DRAM];
		uint32_t dest = *rsp->cp0.cr[CP0_REGISTER_DMA_CACHE];
		PRSP_HOOK_ONLY(const bool prsp_to_imem = (dest & 0x1000) != 0;)

#ifdef INTENSE_DEBUG
		fprintf(stderr, "DMA READ: (0x%x <- 0x%x) len %u, count %u, skip %u\n", dest & 0x1ffc, source & 0x7ffffc,
		        length, count + 1, skip);
#endif

		do
		{
			const uint32_t dest_start = dest & 0x1FFC;
			const uint32_t source_start = source & 0x7FFFFC;
			if ((dest_start & 0xfff) + length <= 0x1000 && source_start + length <= 0x800000)
			{
				// The row neither wraps in RDRAM nor straddles DMEM / IMEM: plain copy.
				if (dest_start & 0x1000)
					copy_to_imem(rsp, dest_start & 0xfff, rsp->rdram + (source_start >> 2), length);
				else
					memcpy(rsp->dmem + (dest_start >> 2), rsp->rdram + (source_start >> 2), length);
			}
			else
			{
				unsigned j = 0;
				do
				{
					uint32_t source_addr = (source + j) & 0x7FFFFC;
					uint32_t dest_addr = (dest + j) & 0x1FFC;
					uint32_t word = rsp->rdram[source_addr >> 2];

					if (dest_addr & 0x1000)
						copy_to_imem(rsp, dest_addr & 0xfff, &word, 4);
					else
						rsp->dmem[dest_addr >> 2] = word;

					j += 4;
				} while (j < length);
			}

			source += length + skip;
			dest += length;
		} while (++i <= count);

		*rsp->cp0.cr[CP0_REGISTER_DMA_DRAM] = source;
		*rsp->cp0.cr[CP0_REGISTER_DMA_CACHE] = dest;
		*rsp->cp0.cr[CP0_REGISTER_DMA_READ_LENGTH] = 0xff8;
		PRSP_HOOK_DMA(0, length * (count + 1), prsp_to_imem);

#ifdef INTENSE_DEBUG
		log_rsp_mem_parallel();
#endif
		return rsp->dirty_blocks ? MODE_CHECK_FLAGS : MODE_CONTINUE;
	}

	static void rsp_dma_write(RSP::CPUState *rsp)
	{
		uint32_t length_reg = *rsp->cp0.cr[CP0_REGISTER_DMA_WRITE_LENGTH];
		uint32_t length = (length_reg & 0xFFF) + 1;
		uint32_t skip = (length_reg >> 20) & 0xFFF;
		unsigned count = (length_reg >> 12) & 0xFF;

		// Force alignment.
		length = (length + 0x7) & ~0x7;
		*rsp->cp0.cr[CP0_REGISTER_DMA_CACHE] &= ~0x3;
		*rsp->cp0.cr[CP0_REGISTER_DMA_DRAM] &= ~0x7;

		// Check length.
		if (((*rsp->cp0.cr[CP0_REGISTER_DMA_CACHE] & 0xFFF) + length) > 0x1000)
			length = 0x1000 - (*rsp->cp0.cr[CP0_REGISTER_DMA_CACHE] & 0xFFF);

		uint32_t dest = *rsp->cp0.cr[CP0_REGISTER_DMA_DRAM];
		uint32_t source = *rsp->cp0.cr[CP0_REGISTER_DMA_CACHE];

#ifdef INTENSE_DEBUG
		fprintf(stderr, "DMA WRITE: (0x%x <- 0x%x) len %u, count %u, skip %u\n", dest & 0x7ffffc, source & 0x1ffc,
		        length, count + 1, skip);
#endif

		unsigned i = 0;
		do
		{
			const uint32_t source_start = source & 0x1FFC;
			const uint32_t dest_start = dest & 0x7FFFFC;
			if ((source_start & 0xfff) + length <= 0x1000 && dest_start + length <= 0x800000)
			{
				const uint32_t *src = (source_start & 0x1000) ? rsp->imem + ((source_start & 0xfff) >> 2)
				                                              : rsp->dmem + (source_start >> 2);
				memcpy(rsp->rdram + (dest_start >> 2), src, length);
#ifdef PRSP_SHADOW
				prsp_shadow_dma_write(rsp, dest_start, length);
#endif
			}
			else
			{
				unsigned j = 0;
				do
				{
					uint32_t source_addr = (source + j) & 0x1FFC;
					uint32_t dest_addr = (dest + j) & 0x7FFFFC;

					rsp->rdram[dest_addr >> 2] =
					    (source_addr & 0x1000) ? rsp->imem[(source_addr & 0xfff) >> 2] : rsp->dmem[source_addr >> 2];
#ifdef PRSP_SHADOW
					prsp_shadow_dma_write(rsp, dest_addr, 4);
#endif

					j += 4;
				} while (j < length);
			}

			source += length;
			dest += length + skip;
		} while (++i <= count);

		*rsp->cp0.cr[CP0_REGISTER_DMA_CACHE] = source;
		*rsp->cp0.cr[CP0_REGISTER_DMA_DRAM] = dest;
		*rsp->cp0.cr[CP0_REGISTER_DMA_WRITE_LENGTH] = 0xff8;
		PRSP_HOOK_DMA(1, length * (count + 1), 0);
#ifdef INTENSE_DEBUG
		log_rsp_mem_parallel();
#endif
	}
#endif

	int RSP_MTC0(RSP::CPUState *rsp, unsigned rd, unsigned rt)
	{
		uint32_t val = rsp->sr[rt];

		switch (static_cast<CP0Registers>(rd & 15))
		{
		case CP0_REGISTER_DMA_CACHE:
			*rsp->cp0.cr[CP0_REGISTER_DMA_CACHE] = val & 0x1fff;
			break;

		case CP0_REGISTER_DMA_DRAM:
			*rsp->cp0.cr[CP0_REGISTER_DMA_DRAM] = val & 0xffffff;
			break;

		case CP0_REGISTER_DMA_READ_LENGTH:
			*rsp->cp0.cr[CP0_REGISTER_DMA_READ_LENGTH] = val;
#ifdef PARALLEL_INTEGRATION
			return rsp_dma_read(rsp);
#else
			return MODE_DMA_READ;
#endif

		case CP0_REGISTER_DMA_WRITE_LENGTH:
			*rsp->cp0.cr[CP0_REGISTER_DMA_WRITE_LENGTH] = val;
#ifdef PARALLEL_INTEGRATION
			rsp_dma_write(rsp);
#endif
			break;

		case CP0_REGISTER_SP_STATUS:
			return rsp_status_write(rsp, val);

		case CP0_REGISTER_SP_SEMAPHORE:
			// Any write to the semaphore register, regardless of value, sets it to 0 for the next read
			*rsp->cp0.cr[CP0_REGISTER_SP_SEMAPHORE] = 0;
			break;

		case CP0_REGISTER_CMD_START:
#ifdef INTENSE_DEBUG
			fprintf(stderr, "CMD_START 0x%x\n", val & 0xfffffff8u);
#endif
			*rsp->cp0.cr[CP0_REGISTER_CMD_START] = *rsp->cp0.cr[CP0_REGISTER_CMD_CURRENT] =
			    *rsp->cp0.cr[CP0_REGISTER_CMD_END] = val & 0xfffffff8u;
			break;

		case CP0_REGISTER_CMD_END:
#ifdef INTENSE_DEBUG
			fprintf(stderr, "CMD_END 0x%x\n", val & 0xfffffff8u);
#endif
			*rsp->cp0.cr[CP0_REGISTER_CMD_END] = val & 0xfffffff8u;

#ifdef PARALLEL_INTEGRATION
			RSP::rsp.ProcessRdpList();
#endif
			break;

		case CP0_REGISTER_CMD_CLOCK:
			*rsp->cp0.cr[CP0_REGISTER_CMD_CLOCK] = val;
			break;

		case CP0_REGISTER_CMD_STATUS:
			*rsp->cp0.cr[CP0_REGISTER_CMD_STATUS] &= ~(!!(val & 0x1) << 0);
			*rsp->cp0.cr[CP0_REGISTER_CMD_STATUS] |= (!!(val & 0x2) << 0);
			*rsp->cp0.cr[CP0_REGISTER_CMD_STATUS] &= ~(!!(val & 0x4) << 1);
			*rsp->cp0.cr[CP0_REGISTER_CMD_STATUS] |= (!!(val & 0x8) << 1);
			*rsp->cp0.cr[CP0_REGISTER_CMD_STATUS] &= ~(!!(val & 0x10) << 2);
			*rsp->cp0.cr[CP0_REGISTER_CMD_STATUS] |= (!!(val & 0x20) << 2);
			*rsp->cp0.cr[CP0_REGISTER_CMD_TMEM_BUSY] &= !(val & 0x40) * -1;
			*rsp->cp0.cr[CP0_REGISTER_CMD_CLOCK] &= !(val & 0x200) * -1;
			break;

		case CP0_REGISTER_CMD_CURRENT:
		case CP0_REGISTER_CMD_BUSY:
		case CP0_REGISTER_CMD_PIPE_BUSY:
		case CP0_REGISTER_CMD_TMEM_BUSY:
			break;

		default:
			*rsp->cp0.cr[rd & 15] = val;
			break;
		}

		return MODE_CONTINUE;
	}
}
