// Differential test for the SP DMA fast path in rsp/cp0.cpp against the original word-by-word loops (tests/cp0_reference.cpp).
//   g++ -DPARALLEL_INTEGRATION -I. -I<m64p api dir> tests/dma_diff.cpp rsp/cp0.cpp -o dma_diff   (see tests/dma_diff.sh)
#include "state.hpp"
#include "m64p_plugin.h"
#include "rsp_1.1.h"
#include <random>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace RSP
{
RSP_INFO rsp;
short MFC0_count[32];
int SP_STATUS_TIMEOUT = 16;
} // namespace RSP

extern "C"
{
	int RSP_MTC0(RSP::CPUState *rsp, unsigned rd, unsigned rt);
	int RSP_MTC0_ref(RSP::CPUState *rsp, unsigned rd, unsigned rt);
}

static void fake_process_rdp_list() {}

struct M
{
	RSP::CPUState st;
	uint32_t cr[16] = {};
	uint32_t irq = 0;
	uint32_t *rdram;
	alignas(64) uint32_t dmem[1024], imem[1024];
	M()
	{
		rdram = static_cast<uint32_t *>(calloc(0x800000 / 4, 4));
		st.rdram = rdram;
		st.dmem = dmem;
		st.imem = imem;
		for (int i = 0; i < 16; i++)
			st.cp0.cr[i] = &cr[i];
		st.cp0.irq = &irq;
	}
};

int main(int argc, char **argv)
{
	unsigned iters = argc > 1 ? atoi(argv[1]) : 20000;
	std::mt19937 rnd(argc > 2 ? atoi(argv[2]) : 1);
	RSP::rsp.ProcessRdpList = fake_process_rdp_list;
	M a, b;
	unsigned bad = 0, imem_reads = 0, imem_clean = 0, imem_partial = 0;
	for (unsigned it = 0; it < iters; it++)
	{
		for (unsigned i = 0; i < 0x800000 / 4; i += 1 + rnd() % 4096)
			a.rdram[i] = rnd();
		for (int i = 0; i < 1024; i++)
			a.dmem[i] = rnd(), a.imem[i] = rnd();
		memcpy(b.rdram, a.rdram, 0x800000);
		memcpy(b.dmem, a.dmem, 4096);
		memcpy(b.imem, a.imem, 4096);
		a.st.dirty_blocks = b.st.dirty_blocks = 0;

		bool write = rnd() & 1;
		uint32_t cache = (rnd() % 3 == 0) ? (rnd() & 0x1fff) : ((rnd() & 0x1ff8) | (rnd() % 4 == 0 ? 4 : 0));
		uint32_t dram;
		switch (rnd() % 4)
		{
		case 0: dram = rnd() & 0xffffff; break;
		case 1: dram = 0x7fff00 - (rnd() % 64) * 8 + (rnd() % 4) * 0; break; // near the end of RDRAM
		default: dram = (rnd() & 0x7fffff) & ~7u; break;
		}
		unsigned len = rnd() % 4 == 0 ? (rnd() & 0xfff) : (rnd() & 0xff);
		unsigned count = rnd() % 3 == 0 ? rnd() % 16 : 0;
		unsigned skip = rnd() % 3 == 0 ? rnd() & 0xfff : 0;
		uint32_t lenreg = len | (count << 12) | (skip << 20);
		M *ms[2] = { &a, &b };
		for (M *m : ms)
		{
			m->cr[RSP::CP0_REGISTER_DMA_CACHE] = cache;
			m->cr[RSP::CP0_REGISTER_DMA_DRAM] = dram;
			m->st.sr[1] = lenreg;
		}
		// Make part of a read into IMEM write identical bytes (whole rows, single words, whole blocks), so that skipping the
		// invalidation of unchanged blocks is exercised with partial overlaps; both machines get the same RDRAM.
		if (!write && (cache & 0x1000) && rnd() % 4 != 0)
		{
			uint32_t l = ((len & 0xfff) + 1 + 7) & ~7u;
			uint32_t c = cache & ~3u, d = dram & ~7u;
			if ((c & 0xfff) + l > 0x1000)
				l = 0x1000 - (c & 0xfff);
			unsigned mode = rnd() % 4;
			for (unsigned r = 0; r <= count; r++)
			{
				uint32_t s = d + r * (l + skip), t = c + r * l;
				for (unsigned j = 0; j < l; j += 4)
				{
					uint32_t sa = (s + j) & 0x7FFFFC, da = (t + j) & 0x1FFC;
					bool same = mode == 0 ? true : mode == 1 ? ((da >> 8) & 1) : mode == 2 ? (rnd() % 16 != 0) : (rnd() % 2);
					if ((da & 0x1000) && same)
						a.rdram[sa >> 2] = a.imem[(da & 0xfff) >> 2];
				}
			}
			memcpy(b.rdram, a.rdram, 0x800000);
		}
		RSP_MTC0(&a.st, write ? RSP::CP0_REGISTER_DMA_WRITE_LENGTH : RSP::CP0_REGISTER_DMA_READ_LENGTH, 1);
		RSP_MTC0_ref(&b.st, write ? RSP::CP0_REGISTER_DMA_WRITE_LENGTH : RSP::CP0_REGISTER_DMA_READ_LENGTH, 1);
		if (!write && (cache & 0x1000))
		{
			imem_reads++;
			unsigned changed = 0;
			for (unsigned k = 0; k < 16; k++)
				changed += memcmp(a.imem + k * 64, b.imem + k * 64, 256) != 0;
			(void)changed;
			if (!a.st.dirty_blocks)
				imem_clean++;
			else if (__builtin_popcount(a.st.dirty_blocks) < 15)
				imem_partial++;
		}
		bool ok = !memcmp(a.rdram, b.rdram, 0x800000) && !memcmp(a.dmem, b.dmem, 4096) && !memcmp(a.imem, b.imem, 4096) &&
		          !memcmp(a.cr, b.cr, sizeof(a.cr)) && a.st.dirty_blocks == b.st.dirty_blocks;
		if (!ok)
		{
			bad++;
			if (bad < 10)
				fprintf(stderr, "MISMATCH it %u write %d cache %x dram %x lenreg %x dirty %x vs %x\n", it, write, cache, dram, lenreg,
				        a.st.dirty_blocks, b.st.dirty_blocks);
		}
	}
	fprintf(stderr, "dma_diff: %u iterations (%u reads into IMEM: %u left every block clean, %u dirtied only some blocks), %u mismatches\n", iters,
	        imem_reads, imem_clean, imem_partial, bad);
	return bad != 0;
}
