// Hooks for a host that defers (coalesces) the RDP command lists the ucode submits with a CMD_END write, i.e. that swaps
// RSP_INFO::ProcessRdpList for a function that only remembers the range and hands it to the real one later. Everything
// here expands to nothing unless PRSP_RDP_DEFER is defined, so the normal build is unchanged (same object code). The
// host defines the two functions; they must run any list still pending, exactly as the ucode submitted it, because the
// event that follows would otherwise see the RDP in a different state than it has without deferral.
#ifndef PRSP_RDP_DEFER_H__
#define PRSP_RDP_DEFER_H__

#ifdef PRSP_RDP_DEFER
#include <stdint.h>
extern "C"
{
	// `why` is PRSP_RDP_FLUSH_*: before a write to CMD_START or CMD_STATUS (a new buffer / the XBUS bit changes), and when the
	// run loop of a task (parallelRSPDoRspCycles) has ended, before the interrupts are checked.
	enum { PRSP_RDP_FLUSH_START, PRSP_RDP_FLUSH_STATUS, PRSP_RDP_FLUSH_RUN_END };
	void prsp_rdp_defer_flush(int why);
	// before an SP DMA write copies rows to RDRAM: `dram` is the DMA_DRAM register (word aligned), `length` the aligned
	// row length in bytes, `skip` the gap between rows, `count` the number of rows minus one. Called with the register
	// values the DMA uses, before any byte is written.
	void prsp_rdp_defer_dma_write(uint32_t dram, uint32_t length, uint32_t skip, unsigned count);
}
#define PRSP_RDP_DEFER_FLUSH(why) prsp_rdp_defer_flush(why)
#define PRSP_RDP_DEFER_DMA_WRITE(dram, length, skip, count) prsp_rdp_defer_dma_write(dram, length, skip, count)
#else
#define PRSP_RDP_DEFER_FLUSH(why) ((void)0)
#define PRSP_RDP_DEFER_DMA_WRITE(dram, length, skip, count) ((void)0)
#endif
#endif
