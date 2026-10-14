// Differential test for the aarch64 JIT: runs the same random RSP programs through two JIT instances, one using the original
// RSP_Vxxx / RSP_Lxx helper-call code generation (the reference, "slow") and one using the optimized code generation ("fast"),
// and compares the complete architectural state (scalar registers, vector registers, accumulator, flags, divide state, DMEM,
// PC and delay slot state) after every program.
//
//   jit_diff [programs] [seed] [mode]      mode: all (default) | single | random | vu | div | scalar | link
//
// Mode `scalar` is different: it does not compare the two JITs with each other but each of them with a small C++ model of the scalar
// load/store instructions (LB LBU LH LHU LW LWU SB SH SW) and the ALU forms used to build addresses, so a code generation change that
// both JITs share (the Lightning backend, the addressing sequence) is caught as well.
#include "rsp_jit.hpp"
#include <new>
#include <random>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>

using namespace RSP;

#ifdef PRSP_LS_COUNT
extern "C" unsigned long rsp_ls_slow_calls[2][32];
#endif

#ifdef PRSP_VU_FAST
struct Machine
{
	JIT::CPU cpu;
	alignas(64) uint32_t dmem[1024];
	alignas(64) uint32_t imem[1024];
	uint32_t cr_storage[16] = {};
	uint32_t irq = 0;
	bool fast;

	Machine(bool fast_) : fast(fast_)
	{
		auto &st = cpu.get_state();
		cpu.set_dmem(dmem);
		cpu.set_imem(imem);
		for (int i = 0; i < 16; i++)
			st.cp0.cr[i] = &cr_storage[i];
		st.cp0.irq = &irq;
#ifdef PRSP_VU_FAST
		cpu.set_vu_fast(fast);
		cpu.set_clobber_slow_paths(fast);
#endif
	}
};

static uint32_t cr_init[16];

static void copy_state(Machine &dst, const CPUState &src, const uint32_t *dmem, const uint32_t *imem)
{
	memcpy(dst.cr_storage, cr_init, sizeof(cr_init));
	auto &d = dst.cpu.get_state();
	memcpy(d.sr, src.sr, sizeof(d.sr));
	memcpy(&d.cp2, &src.cp2, sizeof(d.cp2));
	d.pc = src.pc;
	d.has_delay_slot = 0;
	d.branch_target = 0;
	d.dirty_blocks = 0;
	memcpy(dst.dmem, dmem, sizeof(dst.dmem));
	memcpy(dst.imem, imem, sizeof(dst.imem));
	dst.cpu.invalidate_imem();
}

static bool compare_state(const CPUState &a, const CPUState &b, const uint32_t *da, const uint32_t *db, const uint32_t *cr_a,
                          const uint32_t *cr_b, std::string &why)
{
	char buf[256];
	for (int i = 0; i < 32; i++)
		if (a.sr[i] != b.sr[i])
		{
			snprintf(buf, sizeof(buf), "sr[%d]: %08x vs %08x", i, a.sr[i], b.sr[i]);
			why = buf;
			return false;
		}
	for (int r = 0; r < 32; r++)
		if (memcmp(a.cp2.regs[r].e, b.cp2.regs[r].e, 16))
		{
			snprintf(buf, sizeof(buf), "v%d differs", r);
			why = buf;
			return false;
		}
	for (int r = 0; r < 3; r++)
		if (memcmp(a.cp2.flags[r].e, b.cp2.flags[r].e, 32))
		{
			snprintf(buf, sizeof(buf), "flags[%d] differ", r);
			why = buf;
			return false;
		}
	if (memcmp(a.cp2.acc.e, b.cp2.acc.e, 48))
	{
		why = "acc differs";
		return false;
	}
	if (a.cp2.div_out != b.cp2.div_out || a.cp2.div_in != b.cp2.div_in || a.cp2.dp_flag != b.cp2.dp_flag)
	{
		why = "divide state differs";
		return false;
	}
	if (a.pc != b.pc || a.has_delay_slot != b.has_delay_slot || a.branch_target != b.branch_target)
	{
		why = "pc/delay slot differs";
		return false;
	}
	if (memcmp(cr_a, cr_b, 64))
	{
		why = "cp0 registers differ";
		return false;
	}
	if (memcmp(da, db, 4096))
	{
		why = "dmem differs";
		return false;
	}
	return true;
}

static std::mt19937 rnd;


// ---- scalar load/store model (mode `scalar`) ----
struct ScalarModel
{
	uint32_t r[32];
	uint32_t dmem[1024];

	uint32_t get8(uint32_t a) const { a &= 0xfff; return (dmem[a >> 2] >> (24 - 8 * (a & 3))) & 0xff; }
	void put8(uint32_t a, uint32_t v)
	{
		a &= 0xfff;
		unsigned sh = 24 - 8 * (a & 3);
		dmem[a >> 2] = (dmem[a >> 2] & ~(0xffu << sh)) | ((v & 0xff) << sh);
	}
	void set(unsigned i, uint32_t v) { if (i) r[i] = v; }

	// Runs until BREAK.
	void run(const uint32_t *imem)
	{
		for (unsigned pc = 0; pc < 1024; pc++)
		{
			uint32_t in = imem[pc];
			uint32_t op = in >> 26, rs = (in >> 21) & 31, rt = (in >> 16) & 31, rd = (in >> 11) & 31, sh = (in >> 6) & 31;
			int32_t simm = int16_t(in);
			uint32_t uimm = in & 0xffff;
			uint32_t a = (r[rs] + uint32_t(simm)) & 0xfff;
			switch (op)
			{
			case 000:
				switch (in & 63)
				{
				case 000: set(rd, r[rt] << sh); break;
				case 002: set(rd, r[rt] >> sh); break;
				case 015: return;
				case 041: set(rd, r[rs] + r[rt]); break;
				case 043: set(rd, r[rs] - r[rt]); break;
				case 045: set(rd, r[rs] | r[rt]); break;
				}
				break;
			case 010: case 011: set(rt, r[rs] + uint32_t(simm)); break;
			case 014: set(rt, r[rs] & uimm); break;
			case 015: set(rt, r[rs] | uimm); break;
			case 017: set(rt, uimm << 16); break;
			case 040: set(rt, uint32_t(int32_t(int8_t(get8(a))))); break;
			case 044: set(rt, get8(a)); break;
			case 041: set(rt, uint32_t(int32_t(int16_t((get8(a) << 8) | get8(a + 1))))); break;
			case 045: set(rt, (get8(a) << 8) | get8(a + 1)); break;
			case 043: case 047: set(rt, (get8(a) << 24) | (get8(a + 1) << 16) | (get8(a + 2) << 8) | get8(a + 3)); break;
			case 050: put8(a, r[rt]); break;
			case 051: put8(a, r[rt] >> 8); put8(a + 1, r[rt]); break;
			case 053: put8(a, r[rt] >> 24); put8(a + 1, r[rt] >> 16); put8(a + 2, r[rt] >> 8); put8(a + 3, r[rt]); break;
			}
		}
	}
};

// A program of address arithmetic and loads/stores. Registers 1..9 form a small pool so that results feed later addresses and
// stored values; constants near the end of DMEM (the 4 KB wrap) and bases of $0 are common.
static void gen_scalar_program(uint32_t *imem, unsigned len)
{
	for (unsigned i = 0; i < 1024; i++)
		imem[i] = 0;
	auto pool = [] { return 1 + rnd() % 9; };
	for (unsigned i = 0; i < len; i++)
	{
		unsigned k = rnd() % 100, rs = pool(), rt = pool(), rd = pool();
		if (rnd() % 5 == 0)
			rs = 0;
		uint32_t imm;
		switch (rnd() % 6)
		{
		case 0: imm = rnd() & 15; break;
		case 1: imm = 0xfff0u + (rnd() & 15); break;
		case 2: imm = (0u - (rnd() & 15)) & 0xffff; break;
		case 3: imm = rnd() & 0xfff; break;
		case 4: imm = (rnd() & 0xff) << 2; break;
		default: imm = rnd() & 0xffff; break;
		}
		if (k < 14) // build a constant
		{
			imem[i] = (017u << 26) | (rt << 16) | ((rnd() & 3) ? 0 : (rnd() & 0xffff));
		}
		else if (k < 26)
			imem[i] = (015u << 26) | (rs << 21) | (rt << 16) | imm;
		else if (k < 34)
			imem[i] = (((rnd() & 1) ? 011u : 010u) << 26) | (rs << 21) | (rt << 16) | imm;
		else if (k < 38)
			imem[i] = (014u << 26) | (rs << 21) | (rt << 16) | (imm | 0xf000);
		else if (k < 42)
		{
			static const unsigned f[] = { 041, 043, 045 };
			imem[i] = (rs << 21) | (pool() << 16) | (rd << 11) | f[rnd() % 3];
		}
		else if (k < 45)
			imem[i] = (pool() << 16) | (rd << 11) | ((rnd() % 32) << 6) | ((rnd() & 1) ? 0 : 2);
		else if (k < 75)
		{
			static const unsigned t[] = { 040, 041, 043, 044, 045, 047 };
			imem[i] = (t[rnd() % 6] << 26) | (rs << 21) | (rt << 16) | imm;
		}
		else
		{
			static const unsigned t[] = { 050, 051, 053 };
			imem[i] = (t[rnd() % 3] << 26) | (rs << 21) | (rt << 16) | imm;
		}
	}
	for (unsigned i = 0; i < 8; i++)
		imem[len + i] = 0x0000000d;
}

static uint32_t rnd_flag_vec_word()
{
	return (rnd() & 1) ? 0xffffffffu : 0u;
}

static void randomize_state(CPUState &st, uint32_t *dmem)
{
	for (int i = 1; i < 32; i++)
	{
		switch (rnd() % 4)
		{
		case 0: st.sr[i] = rnd(); break;
		case 1: st.sr[i] = (rnd() & 0xff0) ; break;
		case 2: st.sr[i] = rnd() & 0xfff; break;
		default: st.sr[i] = 0x100 + (rnd() & 0x7f) * 16; break;
		}
	}
	st.sr[0] = 0;
	st.sr[32] = 0;
	uint16_t *p = reinterpret_cast<uint16_t *>(&st.cp2);
	(void)p;
	for (int r = 0; r < 32; r++)
		for (int i = 0; i < 8; i++)
		{
			unsigned k = rnd() % 8;
			uint16_t v = rnd();
			if (k == 0) v = 0x8000; else if (k == 1) v = 0x7fff; else if (k == 2) v = 0; else if (k == 3) v = 0xffff;
			st.cp2.regs[r].e[i] = v;
		}
	bool canonical = rnd() % 4 != 0;
	for (int f = 0; f < 3; f++)
		for (int i = 0; i < 16; i++)
			st.cp2.flags[f].e[i] = canonical ? ((rnd() & 1) ? 0xffff : 0) : uint16_t(rnd());
	for (int i = 0; i < 24; i++)
	{
		unsigned k = rnd() % 8;
		uint16_t v = rnd();
		if (k == 0) v = 0x8000; else if (k == 1) v = 0x7fff; else if (k == 2) v = 0; else if (k == 3) v = 0xffff;
		st.cp2.acc.e[i] = v;
	}
	st.cp2.div_out = rnd();
	st.cp2.div_in = rnd();
	st.cp2.dp_flag = rnd() & 1;
	(void)rnd_flag_vec_word;
	for (int i = 0; i < 1024; i++)
		dmem[i] = rnd();
	for (int i = 0; i < 16; i++)
		cr_init[i] = rnd();
	st.pc = 0;
}

static uint32_t vu_instr(unsigned op, unsigned vd, unsigned vs, unsigned vt, unsigned e)
{
	return 0x4a000000u | (e << 21) | (vt << 16) | (vs << 11) | (vd << 6) | op;
}

static uint32_t rnd_vu()
{
	// Bias towards implemented ops, but include everything (reserved opcodes included).
	unsigned op = rnd() % 64;
	unsigned vd = rnd() % 32, vs = rnd() % 32, vt = rnd() % 32;
	switch (rnd() % 6)
	{
	case 0: vs = vd; break;
	case 1: vt = vd; break;
	case 2: vs = vt; break;
	case 3: vs = vt = vd; break;
	default: break;
	}
	return vu_instr(op, vd, vs, vt, rnd() % 16);
}

static uint32_t rnd_scalar()
{
	unsigned rs = rnd() % 32, rt = rnd() % 32, rd = rnd() % 32, sh = rnd() % 32;
	uint32_t imm = rnd() & 0xffff;
	switch (rnd() % 14)
	{
	case 0: { static const unsigned f[] = { 0, 2, 3, 4, 6, 7, 040, 041, 042, 043, 044, 045, 046, 047, 052, 053 };
		return (rs << 21) | (rt << 16) | (rd << 11) | (sh << 6) | f[rnd() % 16]; }
	case 1: return (010u << 26) | (rs << 21) | (rt << 16) | imm;
	case 2: return (011u << 26) | (rs << 21) | (rt << 16) | imm;
	case 3: return (014u << 26) | (rs << 21) | (rt << 16) | imm;
	case 4: return (015u << 26) | (rs << 21) | (rt << 16) | imm;
	case 5: return (017u << 26) | (rt << 16) | imm;
	case 6: { static const unsigned t[] = { 040, 041, 043, 044, 045, 047 }; return (t[rnd() % 6] << 26) | (rs << 21) | (rt << 16) | (rnd() & 0xff); }
	case 7: { static const unsigned t[] = { 050, 051, 053 }; return (t[rnd() % 3] << 26) | (rs << 21) | (rt << 16) | (rnd() & 0xff); }
	case 8: return (012u << 26) | (rs << 21) | (rt << 16) | imm;
	case 9: return (016u << 26) | (rs << 21) | (rt << 16) | imm;
	case 10: // forward branches
	{
		static const unsigned t[] = { 004, 005, 006, 007 };
		return (t[rnd() % 4] << 26) | (rs << 21) | (rt << 16) | (1 + rnd() % 4);
	}
	case 11: return (001u << 26) | (rs << 21) | (((rnd() & 1) ? 0 : 1u) << 16) | (1 + rnd() % 4);
	default: return (013u << 26) | (rs << 21) | (rt << 16) | imm;
	}
}

static uint32_t rnd_lwc2(bool store)
{
	unsigned base = rnd() % 32, vt = rnd() % 32, op = rnd() % 12, e = rnd() % 16;
	if (!store && op == 10) op = 11; // no load op 10 (LWV)
	if (rnd() % 2 == 0) e = 0;
	else if (rnd() % 2 == 0) e &= ~1u;
	int off = int(rnd() % 128) - 64;
	return ((store ? 072u : 062u) << 26) | (base << 21) | (vt << 16) | (op << 11) | (e << 7) | (off & 0x7f);
}

static uint32_t rnd_cop2move()
{
	unsigned rt = rnd() % 32, rd = rnd() % 32, imm = rnd() % 16;
	static const unsigned rs[] = { 0, 2, 4, 6 };
	return (022u << 26) | (rs[rnd() % 4] << 21) | (rt << 16) | (rd << 11) | (imm << 7);
}

static uint32_t rnd_cop0()
{
	unsigned rt = rnd() % 32, rd = rnd() % 32;
	if (rnd() & 1)
		return (020u << 26) | (0u << 21) | (rt << 16) | (rd << 11); // MFC0
	static const unsigned mt[] = { 0, 1, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22 }; // no DMA length / status / RDP writes
	rd = mt[rnd() % 16];
	return (020u << 26) | (4u << 21) | (rt << 16) | (rd << 11); // MTC0
}

static const uint32_t BREAK = 0x0000000d;

// Programs of several segments far enough apart to be separate JIT regions, joined by j / jal / jr / conditional branches, so that the
// code that leaves a region (the block lookup, the enter thunk) runs. The segments are visited in increasing order: no loops.
static void gen_link_program(uint32_t *imem)
{
	const unsigned SEG = 160, NSEG = 5;
	for (unsigned i = 0; i < 1024; i++)
		imem[i] = 0;
	for (unsigned k = 0; k < NSEG; k++)
	{
		unsigned at = k * SEG, len = 12 + rnd() % 14; // entered at +0..+8: never in the terminal sequence
		for (unsigned i = 0; i < len; i++)
		{
			unsigned r = rnd() % 100;
			if (r < 30) imem[at + i] = rnd_vu();
			else if (r < 45) imem[at + i] = rnd_lwc2(false);
			else if (r < 55) imem[at + i] = rnd_lwc2(true);
			else imem[at + i] = rnd_scalar();
		}
		unsigned t = at + len + 4; // after some padding (a short forward branch may land in it)
		unsigned next = (k + 1) * SEG + (rnd() % 3) * 4;
		if (k == NSEG - 1)
		{
			imem[t] = BREAK;
			continue;
		}
		unsigned rr = 1 + rnd() % 30;
		switch (rnd() % 5)
		{
		case 0: // j next ; delay slot
			imem[t] = (002u << 26) | next;
			imem[t + 1] = rnd_scalar() & ~(077u << 26) | (015u << 26); // ori in the delay slot
			break;
		case 1: // jal next
			imem[t] = (003u << 26) | next;
			imem[t + 1] = 0;
			break;
		case 2: // lui / ori the target, jr
			imem[t] = (015u << 26) | (0u << 21) | (rr << 16) | ((next << 2) & 0xffff);
			imem[t + 1] = (000u << 26) | (rr << 21) | 010;
			imem[t + 2] = rnd_scalar() & ~(077u << 26) | (015u << 26);
			break;
		case 3: // beq r0, r0 next (always taken), then the fall-through goes on to the next-but-one segment
			imem[t] = (004u << 26) | (next - (t + 1));
			imem[t + 1] = 0;
			imem[t + 2] = (002u << 26) | (next + 4);
			break;
		default: // bne rs, rt: taken or not, then j to the next segment
			imem[t] = (005u << 26) | ((rnd() % 32) << 21) | ((rnd() % 32) << 16) | ((next + 8 - (t + 1)) & 0xffff);
			imem[t + 1] = 0;
			imem[t + 2] = (002u << 26) | next;
			imem[t + 3] = 0;
			break;
		}
		// the last segment may be entered at +0..+8 words: make sure there is something there
		if (k + 1 == NSEG - 1)
			for (unsigned i = 0; i < 12; i++)
				if (!imem[(k + 1) * SEG + i])
					imem[(k + 1) * SEG + i] = BREAK;
	}
	imem[NSEG * SEG - 1] = BREAK;
}

// ---- reciprocal / reciprocal square root (VRCP, VRCPL, VRCPH, VRSQ, VRSQL, VRSQH) and CFC2 / CTC2 ----
enum { VRCP = 060, VRCPL = 061, VRCPH = 062, VMOV = 063, VRSQ = 064, VRSQL = 065, VRSQH = 066 };

static uint16_t edge16()
{
	static const uint16_t edge[] = { 0, 1, 2, 3, 0x7fff, 0x8000, 0x8001, 0xffff, 0xfffe, 0xfffd, 0x4000, 0xc000, 0x0100, 0xff00,
		                             0x00ff, 0x7ffe, 0x2000, 0x5555, 0xaaaa, 0x8100, 0x0080, 0xff80, 0x0200, 0xfe00 };
	return edge[rnd() % (sizeof(edge) / sizeof(edge[0]))];
}

// Random state whose vector lanes, divide input and flag lanes are biased towards the corner cases of the divide unit.
static void randomize_state_edge(CPUState &st, uint32_t *dmem)
{
	randomize_state(st, dmem);
	for (int r = 0; r < 32; r++)
		for (int i = 0; i < 8; i++)
			if (rnd() % 3)
				st.cp2.regs[r].e[i] = edge16();
	st.cp2.div_in = (rnd() % 4) ? edge16() : uint16_t(rnd());
	st.cp2.div_out = rnd();
	st.cp2.dp_flag = rnd() & 1;
}

static uint32_t cop2_move(unsigned rs, unsigned rt, unsigned rd)
{
	return (022u << 26) | (rs << 21) | (rt << 16) | (rd << 11);
}

static uint32_t rnd_div_op()
{
	static const unsigned ops[] = { VRCP, VRCPL, VRCPH, VRSQ, VRSQL, VRSQH, VRCPL, VRSQL };
	unsigned vd = rnd() % 32, vs = rnd() % 32, vt = rnd() % 32;
	switch (rnd() % 6)
	{
	case 0: vd = vt; break;
	case 1: vs = (vs & ~7u) | (rnd() % 8); break;
	default: break;
	}
	return vu_instr(ops[rnd() % 8], vd, vs, vt, rnd() % 16);
}

static void gen_program(uint32_t *imem, unsigned len, int mode)
{
	for (unsigned i = 0; i < 1024; i++)
		imem[i] = 0;
	for (unsigned i = 0; i < len; i++)
	{
		unsigned k = rnd() % 100;
		if (mode == 1) // single op mode, set by the caller
			continue;
		if (k < 45) imem[i] = rnd_vu();
		else if (k < 60) imem[i] = rnd_lwc2(false);
		else if (k < 70) imem[i] = rnd_lwc2(true);
		else if (k < 73) imem[i] = rnd_cop2move();
		else if (k < 76) imem[i] = rnd_cop0();
		else imem[i] = rnd_scalar();
	}
	// Branches may skip over the first BREAK.
	for (unsigned i = 0; i < 8; i++)
		imem[len + i] = BREAK;
}

#endif // PRSP_VU_FAST

int main(int argc, char **argv)
{
	unsigned programs = argc > 1 ? atoi(argv[1]) : 2000;
	(void)programs;
	unsigned seed = argc > 2 ? atoi(argv[2]) : 1;
	std::string mode = argc > 3 ? argv[3] : "all";
	rnd.seed(seed);

#ifndef PRSP_VU_FAST
	fprintf(stderr, "This build has no optimized path (PRSP_VU_FAST undefined); nothing to compare.\n");
	return 0;
#else
	// BREAK makes CPU::run() print the register file in the non-integration build.
	if (!freopen("/dev/null", "w", stdout))
		return 2;

	auto aligned_new = [](size_t size) {
		void *p = nullptr;
		if (posix_memalign(&p, 64, (size + 63) & ~size_t(63)))
			abort();
		return p;
	};
	auto *fast = new (aligned_new(sizeof(Machine))) Machine(true);
	auto *slow = new (aligned_new(sizeof(Machine))) Machine(false);
	auto *init = new (aligned_new(sizeof(CPUState))) CPUState;
	alignas(64) static uint32_t dmem0[1024], imem0[1024];

	unsigned failures = 0, runs = 0;
	auto run_one = [&](const char *what) {
		copy_state(*fast, *init, dmem0, imem0);
		copy_state(*slow, *init, dmem0, imem0);
		if (getenv("JD_TRACE"))
		{
			fprintf(stderr, "run %u (%s):", runs + 1, what);
			for (unsigned i = 0; i < 1024; i++)
				if (imem0[i])
					fprintf(stderr, " %u:%08x", i, imem0[i]);
			fprintf(stderr, "\n");
		}
		auto m1 = slow->cpu.run();
		if (getenv("JD_TRACE"))
			fprintf(stderr, "  slow done\n");
		auto m2 = fast->cpu.run();
		runs++;
		std::string why;
		if (m1 != m2)
			why = "return mode differs";
		else
			compare_state(slow->cpu.get_state(), fast->cpu.get_state(), slow->dmem, fast->dmem, slow->cr_storage, fast->cr_storage, why) ? (void)0 : (void)0;
		bool ok = why.empty() && compare_state(slow->cpu.get_state(), fast->cpu.get_state(), slow->dmem, fast->dmem, slow->cr_storage, fast->cr_storage, why);
		if (!ok)
		{
			failures++;
			if (failures < 20)
			{
				fprintf(stderr, "MISMATCH (%s) run %u: %s\n  program:", what, runs, why.c_str());
				for (unsigned i = 0; i < 1024 && imem0[i] != BREAK; i++)
					fprintf(stderr, " %08x", imem0[i]);
				fprintf(stderr, "\n");
			}
		}
	};

	if (mode == "all" || mode == "single")
	{
		// Every VU opcode x every element, aliasing variants, random state: one instruction (plus a following dependent one).
		for (unsigned op = 0; op < 64; op++)
			for (unsigned e = 0; e < 16; e++)
				for (unsigned rep = 0; rep < (programs / 64 / 16 + 1); rep++)
				{
					randomize_state(*init, dmem0);
					memset(imem0, 0, sizeof(imem0));
					unsigned vd = rnd() % 32, vs = rnd() % 32, vt = rnd() % 32;
					switch (rep % 5) { case 1: vs = vd; break; case 2: vt = vd; break; case 3: vs = vt; break; case 4: vs = vt = vd; break; default: break; }
					imem0[0] = vu_instr(op, vd, vs, vt, e);
					imem0[1] = rnd_vu();
					imem0[2] = vu_instr(op, rnd() % 32, vd, rnd() % 32, rnd() % 16);
					imem0[3] = BREAK;
					run_one("single");
				}
	}
	if (mode == "vu")
	{
		// Chains of VU ops only (accumulator / flag dependencies), many random and corner-case states.
		for (unsigned p = 0; p < programs; p++)
		{
			randomize_state(*init, dmem0);
			memset(imem0, 0, sizeof(imem0));
			unsigned len = 2 + rnd() % 8;
			for (unsigned i = 0; i < len; i++)
				imem0[i] = rnd_vu();
			for (unsigned i = 0; i < 8; i++)
				imem0[len + i] = BREAK;
			run_one("vu");
		}
	}
	if (mode == "all" || mode == "div")
	{
		// (a) Exhaustive sweep of the single-precision input: every 16-bit value in every lane, for all 8 lane selections of
		//     VRCP / VRSQ (writing every destination element), followed by CFC2 of all three flag registers.
		for (unsigned base = 0; base < 65536; base += 8)
		{
			randomize_state_edge(*init, dmem0);
			memset(imem0, 0, sizeof(imem0));
			for (unsigned i = 0; i < 8; i++)
				init->cp2.regs[2].e[i] = uint16_t(base + i);
			unsigned n = 0;
			for (unsigned i = 0; i < 8; i++)
			{
				imem0[n++] = vu_instr(VRCP, 1, i, 2, 8 + i);
				imem0[n++] = vu_instr(VRSQ, 3, i, 2, 8 + i);
			}
			imem0[n++] = vu_instr(VRCP, 4, 5, 2, 0); // e < 8: the element index is e & 7 as well
			imem0[n++] = vu_instr(VRSQ, 6, 7, 2, 7);
			for (unsigned f = 0; f < 4; f++)
				imem0[n++] = cop2_move(2, 8 + f, f);
			imem0[n++] = BREAK;
			run_one("div-sweep");
		}
		// (b) Double precision: VRCPH / VRSQH loads div_in from a lane, VRCPL / VRSQL combines it with the next lane. High and low
		//     halves from the corner-case set and fully random, mixed single / double / stale dp_flag sequences.
		for (unsigned p = 0; p < programs * 8; p++)
		{
			randomize_state_edge(*init, dmem0);
			memset(imem0, 0, sizeof(imem0));
			unsigned n = 0, pairs = 1 + rnd() % 3;
			for (unsigned k = 0; k < pairs; k++)
			{
				bool rsq = rnd() & 1;
				unsigned vt = rnd() % 32, e = rnd() % 16;
				if (rnd() % 2)
					init->cp2.regs[vt].e[e & 7] = (rnd() % 3) ? edge16() : uint16_t(rnd());
				if (rnd() % 4)
					imem0[n++] = vu_instr(rsq ? VRSQH : VRCPH, rnd() % 32, rnd() % 32, vt, e);
				unsigned vt2 = (rnd() % 3) ? vt : rnd() % 32, e2 = (rnd() % 3) ? e : rnd() % 16;
				imem0[n++] = vu_instr(rsq ? VRSQL : VRCPL, rnd() % 32, rnd() % 32, vt2, e2);
				if (rnd() % 3 == 0)
					imem0[n++] = vu_instr(rsq ? VRSQL : VRCPL, rnd() % 32, rnd() % 32, rnd() % 32, rnd() % 16); // dp_flag already cleared
			}
			imem0[n++] = cop2_move(2, 1 + rnd() % 31, rnd() % 4);
			imem0[n++] = BREAK;
			run_one("div-pairs");
		}
		// (c) Random programs of divide ops mixed with other VU ops, CTC2 / CFC2 / MTC2 / MFC2 and scalar code.
		for (unsigned p = 0; p < programs; p++)
		{
			randomize_state_edge(*init, dmem0);
			memset(imem0, 0, sizeof(imem0));
			unsigned len = 4 + rnd() % 40;
			for (unsigned i = 0; i < len; i++)
			{
				unsigned k = rnd() % 100;
				if (k < 40) imem0[i] = rnd_div_op();
				else if (k < 55) imem0[i] = rnd_vu();
				else if (k < 70) imem0[i] = cop2_move(2, rnd() % 32, rnd() % 32);                    // CFC2
				else if (k < 80) imem0[i] = cop2_move(6, rnd() % 32, rnd() % 32);                    // CTC2
				else if (k < 90) imem0[i] = rnd_cop2move();
				else imem0[i] = rnd_scalar();
			}
			for (unsigned i = 0; i < 8; i++)
				imem0[len + i] = BREAK;
			run_one("div-random");
		}
	}
	if (mode == "all" || mode == "random")
	{
		for (unsigned p = 0; p < programs; p++)
		{
			randomize_state(*init, dmem0);
			gen_program(imem0, 8 + rnd() % 120, 0);
			run_one("random");
		}
	}

	if (mode == "all" || mode == "link")
	{
		for (unsigned p = 0; p < programs; p++)
		{
			randomize_state(*init, dmem0);
			gen_link_program(imem0);
			run_one("link");
			run_one("link, regions already compiled"); // the same IMEM: every region exit now finds its block in blocks[]
		}
	}
	if (mode == "prog")
	{
		// Replays one scalar program given in $JD_PROG (hex words) with the registers of $JD_REGS (31 hex words, default 0) through the
		// fast JIT and the model; PRSP_DUMP_JIT=file keeps the generated code.
		static ScalarModel model;
		randomize_state(*init, dmem0);
		memset(imem0, 0, sizeof(imem0));
		unsigned n = 0;
		for (const char *p = getenv("JD_PROG"); p && *p && n < 1000;)
		{
			char *e;
			imem0[n++] = strtoul(p, &e, 16);
			p = e;
		}
		for (unsigned i = 0; i < 8; i++)
			imem0[n + i] = BREAK;
		const char *rp = getenv("JD_REGS");
		for (int i = 1; i < 32; i++)
		{
			char *e = nullptr;
			init->sr[i] = rp ? strtoul(rp, &e, 16) : 0;
			if (rp)
				rp = e;
		}
		model.r[0] = 0;
		for (int i = 1; i < 32; i++)
			model.r[i] = init->sr[i];
		memcpy(model.dmem, dmem0, sizeof(model.dmem));
		model.run(imem0);
		copy_state(*fast, *init, dmem0, imem0);
		fast->cpu.run();
		runs++;
		for (int i = 0; i < 32; i++)
			if (fast->cpu.get_state().sr[i] != model.r[i])
			{
				failures++;
				fprintf(stderr, "r%d: jit %08x model %08x\n", i, fast->cpu.get_state().sr[i], model.r[i]);
			}
		if (memcmp(fast->dmem, model.dmem, 4096))
		{
			failures++;
			fprintf(stderr, "dmem differs\n");
		}
	}
	if (mode == "all" || mode == "scalar")
	{
		// Both JITs against the C++ model: scalar registers and DMEM after loads, stores and address arithmetic.
		static ScalarModel model;
		for (unsigned p = 0; p < programs; p++)
		{
			randomize_state(*init, dmem0);
			gen_scalar_program(imem0, 4 + rnd() % 60);
			model.r[0] = 0;
			for (int i = 1; i < 32; i++)
				model.r[i] = init->sr[i];
			memcpy(model.dmem, dmem0, sizeof(model.dmem));
			model.run(imem0);
			for (int variant = 0; variant < 2; variant++)
			{
				Machine &m = variant ? *fast : *slow;
				copy_state(m, *init, dmem0, imem0);
				m.cpu.run();
				runs++;
				const char *why = nullptr;
				int bad = -1;
				for (int i = 0; i < 32 && !why; i++)
					if (m.cpu.get_state().sr[i] != model.r[i])
						why = "register", bad = i;
				if (!why && memcmp(m.dmem, model.dmem, 4096))
					why = "dmem";
				if (why)
				{
					failures++;
					if (failures < 20)
					{
						// Shortest failing prefix.
						uint32_t full[1024];
						memcpy(full, imem0, sizeof(full));
						unsigned n = 0;
						while (n < 1024 && full[n] != BREAK)
							n++;
						unsigned shortest = n;
						for (unsigned len = 1; len <= n; len++)
						{
							memcpy(imem0, full, sizeof(full));
							for (unsigned i = 0; i < 8; i++)
								imem0[len + i] = BREAK;
							static ScalarModel m2;
							for (int i = 0; i < 32; i++)
								m2.r[i] = i ? init->sr[i] : 0;
							memcpy(m2.dmem, dmem0, sizeof(m2.dmem));
							m2.run(imem0);
							copy_state(m, *init, dmem0, imem0);
							m.cpu.run();
							bool differs = memcmp(m.dmem, m2.dmem, 4096) != 0;
							for (int i = 0; i < 32; i++)
								differs |= m.cpu.get_state().sr[i] != m2.r[i];
							if (differs)
							{
								shortest = len;
								break;
							}
						}
						fprintf(stderr, "MISMATCH (scalar, %s JIT) run %u: %s %d\n  shortest failing program (%u of %u instructions):", variant ? "fast" : "slow",
						        runs, why, bad, shortest, n);
						for (unsigned i = 0; i < shortest; i++)
							fprintf(stderr, " %08x", full[i]);
						fprintf(stderr, "\n  initial registers:");
						for (int i = 1; i < 32; i++)
							fprintf(stderr, " %08x", init->sr[i]);
						fprintf(stderr, "\n");
						memcpy(imem0, full, sizeof(full));
					}
				}
			}
		}
	}
#ifdef PRSP_LS_COUNT
	{
		static const char *names[2][32] = {};
		(void)names;
		fprintf(stderr, "LS slow-path calls (fast machine and reference both count; reference calls the RSP_xxx helpers, not _A):\n");
		for (int st = 0; st < 2; st++)
			for (int op = 0; op < 12; op++)
				if (rsp_ls_slow_calls[st][op])
					fprintf(stderr, "  %s op %d: %lu\n", st ? "store" : "load ", op, rsp_ls_slow_calls[st][op]);
	}
#endif
	fprintf(stderr, "jit_diff: %u runs, %u mismatches\n", runs, failures);
	return failures ? 1 : 0;
#endif
}
