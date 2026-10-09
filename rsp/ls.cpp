#include "../state.hpp"

#ifdef TRACE_COP2
#include <stdio.h>
#define TRACE_LS(op) printf(#op " v%u, %u, 0x%x\n", rt, e, 0u)
#else
#define TRACE_LS(op) ((void)0)
#endif

#ifdef PRSP_LS_COUNT
extern "C" unsigned long rsp_ls_slow_calls[2][32]; // [store][op], JIT slow paths taken (differential test only)
unsigned long rsp_ls_slow_calls[2][32];
#define LS_SLOW_COUNT(store, op) (rsp_ls_slow_calls[store][op]++)
#else
#define LS_SLOW_COUNT(store, op) ((void)0)
#endif

extern "C"
{
	// Using mostly Ares' implementation as a base here

	static inline uint8_t byteFromHalfWords(const uint16_t *arr, unsigned i)
	{
		return (i & 1) ?
			(uint8_t)(arr[i >> 1] & 0xff) :
			(uint8_t)(arr[i >> 1] >> 8);
	}
	
	static inline void writeByteToHalfWords(uint16_t *arr, unsigned i, uint8_t b)
	{
		const unsigned n = i >> 1;
		if (i & 1)
			arr[n] = (arr[n] & 0xff00) | (uint16_t)b;
		else
			arr[n] = (arr[n] & 0xff) | ((uint16_t)b << 8);
	}
	
	// Fast paths. Whenever both the element and the DMEM address are even, the bytes involved are whole big-endian
	// 16-bit lanes, which the host stores as one 16-bit access each (HES), instead of a byte-wise read-modify-write
	// of the lane for every byte. Accesses wrap around at the end of DMEM, like the generic code.
	static inline uint16_t dmem_lane(const uint32_t *dmem, unsigned addr)
	{
		return READ_MEM_U16(dmem, addr & 0xfff);
	}

	static inline void dmem_store_lane(uint32_t *dmem, unsigned addr, uint16_t v)
	{
		WRITE_MEM_U16(dmem, addr & 0xfff, v);
	}

	// Load 8-bit
	void RSP_LBV(RSP::CPUState *rsp, unsigned rt, unsigned e, int offset, unsigned base)
	{
		TRACE_LS(LBV);
		unsigned addr = (rsp->sr[base] + offset * 1) & 0xfff;
		reinterpret_cast<uint8_t *>(rsp->cp2.regs[rt].e)[MES(e)] = READ_MEM_U8(rsp->dmem, addr);
	}

	// Store 8-bit
	void RSP_SBV(RSP::CPUState *rsp, unsigned rt, unsigned e, int offset, unsigned base)
	{
		TRACE_LS(SBV);
		unsigned addr = (rsp->sr[base] + offset * 1) & 0xfff;
		uint8_t v = reinterpret_cast<uint8_t *>(rsp->cp2.regs[rt].e)[MES(e)];

#ifdef INTENSE_DEBUG
		fprintf(stderr, "SBV: 0x%x (0x%x)\n", addr, v);
#endif

		WRITE_MEM_U8(rsp->dmem, addr, v);
	}

	// Load 16-bit
	static inline void ls_LSV(RSP::CPUState *rsp, unsigned rt, unsigned e, unsigned addr)
	{
		TRACE_LS(LSV);
		const unsigned end = (e > 14) ? 16 : (e + 2);
		if (((addr | e) & 1) == 0)
		{
			rsp->cp2.regs[rt].e[e >> 1] = dmem_lane(rsp->dmem, addr);
			return;
		}
		for (unsigned i = e; i < end; i++)
			writeByteToHalfWords(rsp->cp2.regs[rt].e, i & 0xf, READ_MEM_U8(rsp->dmem, addr++ & 0xfff));
	}

	void RSP_LSV(RSP::CPUState *rsp, unsigned rt, unsigned e, int offset, unsigned base)
	{
		ls_LSV(rsp, rt, e, rsp->sr[base] + offset * 2);
	}

	// Same operation with the effective address (base + offset * size) already computed by the JIT.
	void RSP_LSV_A(RSP::CPUState *rsp, unsigned rt, unsigned e, unsigned sum)
	{
		LS_SLOW_COUNT(0, 1);
		ls_LSV(rsp, rt, e, sum);
	}

	// Store 16-bit
	static inline void ls_SSV(RSP::CPUState *rsp, unsigned rt, unsigned e, unsigned addr)
	{
		TRACE_LS(SSV);
		if (((addr | e) & 1) == 0)
		{
			dmem_store_lane(rsp->dmem, addr, rsp->cp2.regs[rt].e[e >> 1]);
			return;
		}
		uint8_t v0 = reinterpret_cast<uint8_t *>(rsp->cp2.regs[rt].e)[MES(e)];
		uint8_t v1 = reinterpret_cast<uint8_t *>(rsp->cp2.regs[rt].e)[MES((e + 1) & 0xf)];

#ifdef INTENSE_DEBUG
		fprintf(stderr, "SSV: 0x%x (0x%x, 0x%x)\n", addr, v0, v1);
#endif

		WRITE_MEM_U8(rsp->dmem, addr, v0);
		WRITE_MEM_U8(rsp->dmem, (addr + 1) & 0xfff, v1);
	}

	void RSP_SSV(RSP::CPUState *rsp, unsigned rt, unsigned e, int offset, unsigned base)
	{
		ls_SSV(rsp, rt, e, (rsp->sr[base] + offset * 2) & 0xfff);
	}

	// Same operation with the effective address (base + offset * size) already computed by the JIT.
	void RSP_SSV_A(RSP::CPUState *rsp, unsigned rt, unsigned e, unsigned sum)
	{
		LS_SLOW_COUNT(1, 1);
		ls_SSV(rsp, rt, e, (sum) & 0xfff);
	}

	// Load 32-bit
	static inline void ls_LLV(RSP::CPUState *rsp, unsigned rt, unsigned e, unsigned addr)
	{
		TRACE_LS(LLV);
		const unsigned end = (e > 12) ? 16 : (e + 4);
		if (((addr | e) & 1) == 0)
		{
			auto *reg = rsp->cp2.regs[rt].e;
			for (unsigned i = e; i < end; i += 2)
				reg[i >> 1] = dmem_lane(rsp->dmem, addr + i - e);
			return;
		}
		for (unsigned i = e; i < end; i++)
			writeByteToHalfWords(rsp->cp2.regs[rt].e, i & 0xf, READ_MEM_U8(rsp->dmem, addr++ & 0xfff));
	}

	void RSP_LLV(RSP::CPUState *rsp, unsigned rt, unsigned e, int offset, unsigned base)
	{
		ls_LLV(rsp, rt, e, rsp->sr[base] + offset * 4);
	}

	// Same operation with the effective address (base + offset * size) already computed by the JIT.
	void RSP_LLV_A(RSP::CPUState *rsp, unsigned rt, unsigned e, unsigned sum)
	{
		LS_SLOW_COUNT(0, 2);
		ls_LLV(rsp, rt, e, sum);
	}

	// Store 32-bit
	static inline void ls_SLV(RSP::CPUState *rsp, unsigned rt, unsigned e, unsigned addr)
	{
		TRACE_LS(SLV);

#ifdef INTENSE_DEBUG
		fprintf(stderr, "SLV 0x%x, e = %u\n", addr, e);
#endif

		if (((addr | e) & 1) == 0)
		{
			const auto *reg = rsp->cp2.regs[rt].e;
			dmem_store_lane(rsp->dmem, addr, reg[e >> 1]);
			dmem_store_lane(rsp->dmem, addr + 2, reg[((e >> 1) + 1) & 7]);
			return;
		}

		for (unsigned i = e; i < e + 4; i++)
			WRITE_MEM_U8(rsp->dmem, addr++, byteFromHalfWords(rsp->cp2.regs[rt].e, i & 0xf));
	}

	void RSP_SLV(RSP::CPUState *rsp, unsigned rt, unsigned e, int offset, unsigned base)
	{
		ls_SLV(rsp, rt, e, (rsp->sr[base] + offset * 4) & 0xfff);
	}

	// Same operation with the effective address (base + offset * size) already computed by the JIT.
	void RSP_SLV_A(RSP::CPUState *rsp, unsigned rt, unsigned e, unsigned sum)
	{
		LS_SLOW_COUNT(1, 2);
		ls_SLV(rsp, rt, e, (sum) & 0xfff);
	}

	// Load 64-bit
	static inline void ls_LDV(RSP::CPUState *rsp, unsigned rt, unsigned e, unsigned addr)
	{
		TRACE_LS(LDV);
		const unsigned end = (e > 8) ? 16 : (e + 8);
		if (((addr | e) & 1) == 0)
		{
			auto *reg = rsp->cp2.regs[rt].e;
			for (unsigned i = e; i < end; i += 2)
				reg[i >> 1] = dmem_lane(rsp->dmem, addr + i - e);
			return;
		}
		for (unsigned i = e; i < end; i++)
			writeByteToHalfWords(rsp->cp2.regs[rt].e, i & 0xf, READ_MEM_U8(rsp->dmem, addr++ & 0xfff));
	}

	void RSP_LDV(RSP::CPUState *rsp, unsigned rt, unsigned e, int offset, unsigned base)
	{
		ls_LDV(rsp, rt, e, rsp->sr[base] + offset * 8);
	}

	// Same operation with the effective address (base + offset * size) already computed by the JIT.
	void RSP_LDV_A(RSP::CPUState *rsp, unsigned rt, unsigned e, unsigned sum)
	{
		LS_SLOW_COUNT(0, 3);
		ls_LDV(rsp, rt, e, sum);
	}

	// Store 64-bit
	static inline void ls_SDV(RSP::CPUState *rsp, unsigned rt, unsigned e, unsigned addr)
	{
		TRACE_LS(SDV);

#ifdef INTENSE_DEBUG
		fprintf(stderr, "SDV 0x%x, e = %u\n", addr, e);
#endif

		// Handle illegal scenario.
		if ((e > 8) || (e & 1) || (addr & 1))
		{
			for (unsigned i = 0; i < 8; i++)
			{
				WRITE_MEM_U8(rsp->dmem, (addr + i) & 0xfff,
				             reinterpret_cast<const uint8_t *>(rsp->cp2.regs[rt].e)[MES((e + i) & 0xf)]);
			}
		}
		else
		{
			e >>= 1;
			for (unsigned i = 0; i < 4; i++)
			{
				WRITE_MEM_U16(rsp->dmem, (addr + 2 * i) & 0xfff, rsp->cp2.regs[rt].e[e + i]);
			}
		}
	}

	void RSP_SDV(RSP::CPUState *rsp, unsigned rt, unsigned e, int offset, unsigned base)
	{
		ls_SDV(rsp, rt, e, (rsp->sr[base] + offset * 8) & 0xfff);
	}

	// Same operation with the effective address (base + offset * size) already computed by the JIT.
	void RSP_SDV_A(RSP::CPUState *rsp, unsigned rt, unsigned e, unsigned sum)
	{
		LS_SLOW_COUNT(1, 3);
		ls_SDV(rsp, rt, e, (sum) & 0xfff);
	}

	// Load 8x8-bit into high bits.
	static inline void ls_LPV(RSP::CPUState *rsp, unsigned rt, unsigned e, unsigned addr)
	{
		TRACE_LS(LPV);
		const unsigned index = (addr & 7) - e;
		addr &= ~7;

		auto *reg = rsp->cp2.regs[rt].e;
		for (unsigned i = 0; i < 8; i++)
			reg[i] = READ_MEM_U8(rsp->dmem, (addr + (i + index & 0xf)) & 0xfff) << 8;
	}

	void RSP_LPV(RSP::CPUState *rsp, unsigned rt, unsigned e, int offset, unsigned base)
	{
		ls_LPV(rsp, rt, e, (rsp->sr[base] + offset * 8) & 0xfff);
	}

	// Same operation with the effective address (base + offset * size) already computed by the JIT.
	void RSP_LPV_A(RSP::CPUState *rsp, unsigned rt, unsigned e, unsigned sum)
	{
		LS_SLOW_COUNT(0, 6);
		ls_LPV(rsp, rt, e, (sum) & 0xfff);
	}

	static inline void ls_SPV(RSP::CPUState *rsp, unsigned rt, unsigned e, unsigned addr)
	{
		TRACE_LS(SPV);
		auto *reg = rsp->cp2.regs[rt].e;

		for (unsigned i = e; i < e + 8; i++) {
			const unsigned shift = ((i & 0xf) < 8) ? 8 : 7;
			WRITE_MEM_U8(rsp->dmem, addr++ & 0xfff, int16_t(reg[i & 7]) >> shift);
		}
	}

	void RSP_SPV(RSP::CPUState *rsp, unsigned rt, unsigned e, int offset, unsigned base)
	{
		ls_SPV(rsp, rt, e, (rsp->sr[base] + offset * 8) & 0xfff);
	}

	// Same operation with the effective address (base + offset * size) already computed by the JIT.
	void RSP_SPV_A(RSP::CPUState *rsp, unsigned rt, unsigned e, unsigned sum)
	{
		LS_SLOW_COUNT(1, 6);
		ls_SPV(rsp, rt, e, (sum) & 0xfff);
	}

	// Load 8x8-bit into high bits, but shift by 7 instead of 8.
	// Was probably used for certain fixed point algorithms to get more headroom without
	// saturation, but weird nonetheless.
	static inline void ls_LUV(RSP::CPUState *rsp, unsigned rt, unsigned e, unsigned addr)
	{
		TRACE_LS(LUV);
		const unsigned index = (addr & 7) - e;
		addr &= ~7;

		auto *reg = rsp->cp2.regs[rt].e;
		for (unsigned i = 0; i < 8; i++)
			reg[i] = READ_MEM_U8(rsp->dmem, (addr + (i + index & 0xf)) & 0xfff) << 7;
	}

	void RSP_LUV(RSP::CPUState *rsp, unsigned rt, unsigned e, int offset, unsigned base)
	{
		ls_LUV(rsp, rt, e, (rsp->sr[base] + offset * 8) & 0xfff);
	}

	// Same operation with the effective address (base + offset * size) already computed by the JIT.
	void RSP_LUV_A(RSP::CPUState *rsp, unsigned rt, unsigned e, unsigned sum)
	{
		LS_SLOW_COUNT(0, 7);
		ls_LUV(rsp, rt, e, (sum) & 0xfff);
	}

	static inline void ls_SUV(RSP::CPUState *rsp, unsigned rt, unsigned e, unsigned addr)
	{
		TRACE_LS(SUV);
		auto *reg = rsp->cp2.regs[rt].e;

		for (unsigned i = e; i < e + 8; i++) {
			const unsigned shift = ((i & 0xf) < 8) ? 7 : 8;
			WRITE_MEM_U8(rsp->dmem, addr++ & 0xfff, int16_t(reg[i & 7]) >> shift);
		}
	}

	void RSP_SUV(RSP::CPUState *rsp, unsigned rt, unsigned e, int offset, unsigned base)
	{
		ls_SUV(rsp, rt, e, (rsp->sr[base] + offset * 8) & 0xfff);
	}

	// Same operation with the effective address (base + offset * size) already computed by the JIT.
	void RSP_SUV_A(RSP::CPUState *rsp, unsigned rt, unsigned e, unsigned sum)
	{
		LS_SLOW_COUNT(1, 7);
		ls_SUV(rsp, rt, e, (sum) & 0xfff);
	}

	// Load 8x8-bits into high bits, but shift by 7 instead of 8.
	// Seems to differ from LUV in that it loads every other byte instead of packed bytes.
	void RSP_LHV(RSP::CPUState *rsp, unsigned rt, unsigned e, int offset, unsigned base)
	{
		TRACE_LS(LHV);
		unsigned addr = rsp->sr[base] + offset * 16;
		const unsigned index = (addr & 7) - e;
		addr &= ~7;

		auto *reg = rsp->cp2.regs[rt].e;
		for (unsigned i = 0; i < 8; i++)
			reg[i] = (uint16_t)READ_MEM_U8(rsp->dmem, (addr + (index + i * 2 & 0xf)) & 0xfff) << 7;
	}

	void RSP_SHV(RSP::CPUState *rsp, unsigned rt, unsigned e, int offset, unsigned base)
	{
		TRACE_LS(SHV);
		unsigned addr = (rsp->sr[base] + offset * 16) & 0xfff;
		const unsigned index = addr & 7;
		addr &= ~7;

		const auto *reg = rsp->cp2.regs[rt].e;
		for (unsigned i = 0; i < 8; i++)
		{
			const unsigned b = e + (i << 1);
			const uint8_t byte = byteFromHalfWords(reg, b & 0xf) << 1 | byteFromHalfWords(reg, b + 1 & 0xf) >> 7;
			WRITE_MEM_U8(rsp->dmem, addr + (index + i * 2 & 0xf), byte);
		}
	}

	void RSP_LFV(RSP::CPUState *rsp, unsigned rt, unsigned e, int offset, unsigned base)
	{
		TRACE_LS(LFV);
		uint16_t temp[8];

		unsigned addr = rsp->sr[base] + offset * 16;
		const unsigned index = (addr & 7) - e;
		const unsigned end = (e > 8) ? 16 : (e + 8);
		addr &= ~7;

		for (unsigned i = 0; i < 4; i++)
		{
			temp[i] = (uint16_t)READ_MEM_U8(rsp->dmem, (addr + (index + i * 4 & 0xf)) & 0xfff) << 7;
			temp[i+4] = (uint16_t)READ_MEM_U8(rsp->dmem, (addr + (index + i * 4 + 8 & 0xf)) & 0xfff) << 7;
		}

		for (unsigned i = e; i < end; i++)
			writeByteToHalfWords(rsp->cp2.regs[rt].e, i, byteFromHalfWords(temp, i));
	}

#define RSP_SFV_CASE(a,b,c,d) \
	WRITE_MEM_U8(rsp->dmem, addr + base, int16_t(reg[a]) >> 7); \
	WRITE_MEM_U8(rsp->dmem, addr + 4 + base, int16_t(reg[b]) >> 7); \
	WRITE_MEM_U8(rsp->dmem, addr + (8 + base & 0xf), int16_t(reg[c]) >> 7); \
	WRITE_MEM_U8(rsp->dmem, addr + (12 + base & 0xf), int16_t(reg[d]) >> 7);

	void RSP_SFV(RSP::CPUState *rsp, unsigned rt, unsigned e, int offset, unsigned base)
	{
		TRACE_LS(SFV);
		unsigned addr = (rsp->sr[base] + offset * 16) & 0xfff;
		base = addr & 7;
		addr &= ~7;

		auto *reg = rsp->cp2.regs[rt].e;
		switch (e)
		{
		case 0:
		case 15:
			RSP_SFV_CASE(0,1,2,3)
			break;
		case 1:
			RSP_SFV_CASE(6,7,4,5)
			break;
		case 4:
			RSP_SFV_CASE(1,2,3,0)
			break;
		case 5:
			RSP_SFV_CASE(7,4,5,6)
			break;
		case 8:
			RSP_SFV_CASE(4,5,6,7)
			break;
		case 11:
			RSP_SFV_CASE(3,0,1,2)
			break;
		case 12:
			RSP_SFV_CASE(5,6,7,4)
			break;
		default:
			WRITE_MEM_U8(rsp->dmem, addr + base, 0);
			WRITE_MEM_U8(rsp->dmem, addr + 4 + base, 0);
			WRITE_MEM_U8(rsp->dmem, addr + (8 + base & 0xf), 0);
			WRITE_MEM_U8(rsp->dmem, addr + (12 + base & 0xf), 0);
			break;
		}
	}

	void RSP_LWV(RSP::CPUState *rsp, unsigned rt, unsigned e, int offset, unsigned base)
	{
		TRACE_LS(LWV);
		unsigned addr = rsp->sr[base] + offset * 16;
		for (unsigned i = 16 - e; i < 16 + e; i++)
		{
			writeByteToHalfWords(rsp->cp2.regs[rt].e, i & 0xf, READ_MEM_U8(rsp->dmem, addr & 0xfff));
			addr += 4;
		}
	}

	void RSP_SWV(RSP::CPUState *rsp, unsigned rt, unsigned e, int offset, unsigned base)
	{
		TRACE_LS(SWV);

		unsigned addr = (rsp->sr[base] + offset * 16) & 0xfff;
		base = addr & 7;
		addr &= ~7;

		for (unsigned i = e; i < e + 16; i++)
			WRITE_MEM_U8(rsp->dmem, addr + (base++ & 0xf), byteFromHalfWords(rsp->cp2.regs[rt].e, i & 0xf));
	}

	static inline void ls_LQV(RSP::CPUState *rsp, unsigned rt, unsigned e, unsigned addr)
	{
		TRACE_LS(LQV);
		unsigned end = 16 + e - (addr & 0xf);
		if (end > 16) end = 16;

		if (((addr | e) & 1) == 0)
		{
			auto *reg = rsp->cp2.regs[rt].e;
			for (unsigned i = e; i < end; i += 2)
				reg[i >> 1] = dmem_lane(rsp->dmem, addr + i - e);
			return;
		}

		for (unsigned i = e; i < end; i++)
			writeByteToHalfWords(rsp->cp2.regs[rt].e, i & 0xf, READ_MEM_U8(rsp->dmem, addr++ & 0xfff));
	}

	void RSP_LQV(RSP::CPUState *rsp, unsigned rt, unsigned e, int offset, unsigned base)
	{
		ls_LQV(rsp, rt, e, rsp->sr[base] + offset * 16);
	}

	// Same operation with the effective address (base + offset * size) already computed by the JIT.
	void RSP_LQV_A(RSP::CPUState *rsp, unsigned rt, unsigned e, unsigned sum)
	{
		LS_SLOW_COUNT(0, 4);
		ls_LQV(rsp, rt, e, sum);
	}

	static inline void ls_SQV(RSP::CPUState *rsp, unsigned rt, unsigned e, unsigned addr)
	{
		TRACE_LS(SQV);
		
		const unsigned end = e + (16 - (addr & 15));
		if (((addr | e) & 1) == 0)
		{
			const auto *reg = rsp->cp2.regs[rt].e;
			for (unsigned i = e; i < end; i += 2)
				dmem_store_lane(rsp->dmem, addr + i - e, reg[(i >> 1) & 7]);
			return;
		}
		for (unsigned i = e; i < end; i++)
			WRITE_MEM_U8(rsp->dmem, addr++, byteFromHalfWords(rsp->cp2.regs[rt].e, i & 15));
	}

	void RSP_SQV(RSP::CPUState *rsp, unsigned rt, unsigned e, int offset, unsigned base)
	{
		ls_SQV(rsp, rt, e, (rsp->sr[base] + offset * 16) & 0xfff);
	}

	// Same operation with the effective address (base + offset * size) already computed by the JIT.
	void RSP_SQV_A(RSP::CPUState *rsp, unsigned rt, unsigned e, unsigned sum)
	{
		LS_SLOW_COUNT(1, 4);
		ls_SQV(rsp, rt, e, (sum) & 0xfff);
	}

	void RSP_LRV(RSP::CPUState *rsp, unsigned rt, unsigned e, int offset, unsigned base)
	{
		TRACE_LS(LRV);
		unsigned addr = rsp->sr[base] + offset * 16;
		const unsigned start = 16 - ((addr & 0xf) - e);
		addr &= ~0xf;

		for (unsigned i = start; i < 16; i++)
			writeByteToHalfWords(rsp->cp2.regs[rt].e, i & 0xf, READ_MEM_U8(rsp->dmem, addr++ & 0xfff));
	}

	void RSP_SRV(RSP::CPUState *rsp, unsigned rt, unsigned e, int offset, unsigned base)
	{
		TRACE_LS(SRV);
		unsigned addr = (rsp->sr[base] + offset * 16) & 0xfff;
		const unsigned end = e + (addr & 0xf);
		base = 16 - (addr & 0xf);
		addr &= ~0xf;

		for (unsigned i = e; i < end; i++)
			WRITE_MEM_U8(rsp->dmem, addr++, byteFromHalfWords(rsp->cp2.regs[rt].e, i + base & 0xf));
	}

	void RSP_LTV(RSP::CPUState *rsp, unsigned rt, unsigned e, int offset, unsigned base)
	{
		TRACE_LS(LTV);
		unsigned addr = rsp->sr[base] + offset * 16;
		const unsigned start = addr & ~7;
		const unsigned vt0 = rt & ~7;
		addr = start + ((e + (addr & 8)) & 0xf);
		unsigned j = e >> 1;

		for (unsigned i = 0; i < 16; j++)
		{
			j &= 7;
			writeByteToHalfWords(rsp->cp2.regs[vt0+j].e, i++, READ_MEM_U8(rsp->dmem, addr++ & 0xfff));
			if (addr == start + 16) addr = start;
			writeByteToHalfWords(rsp->cp2.regs[vt0+j].e, i++, READ_MEM_U8(rsp->dmem, addr++ & 0xfff));
			if (addr == start + 16) addr = start;
		}
	}

	void RSP_STV(RSP::CPUState *rsp, unsigned rt, unsigned e, int offset, unsigned base)
	{
		TRACE_LS(STV);
		e &= ~1;
		rt &= ~7;

		unsigned addr = (rsp->sr[base] + offset * 16) & 0xfff;
		unsigned element = 16 - e;
		base = (addr & 7) - e;
		addr &= ~7;

		for (unsigned i = rt; i < rt + 8; i++ )
		{
			WRITE_MEM_U8(rsp->dmem, addr + (base++ & 0xf), byteFromHalfWords(rsp->cp2.regs[i].e, element++ & 0xf));
			WRITE_MEM_U8(rsp->dmem, addr + (base++ & 0xf), byteFromHalfWords(rsp->cp2.regs[i].e, element++ & 0xf));
		}
	}
}
