// Register-convention VU op bodies for the aarch64 JIT (see rsp_jit.cpp, jit_emit_vu_fast).
//
// Each VU_STUB(X) function is the body of RSP_VX from vfunctions.cpp, split from its operand fetch / result store: the JIT loads vs and
// the (element-shuffled) vt into v0 / v1 itself, copies this machine code into its own code region (either inline or as a shared
// stub it BLs to) and stores the returned v0 into vd. The CPUState pointer lives in x19 for the whole JIT, exactly where the
// JIT keeps it (JIT_REGISTER_STATE), so the bodies reach the accumulator and flag registers at fixed offsets.
//
// The machine code is copied, so it must be position independent and must only touch registers the JIT does not hold live. That is
// enforced two ways: this file is compiled with -ffixed-xN for every integer register the JIT uses (CMakeLists.txt), and rsp_jit.cpp
// validates the copied code at start-up (no branches except the final ret, no pc-relative addressing, no stack access, no
// register outside the allowed set); an op whose body fails validation falls back to the RSP_Vxxx helper call.
//
// The bodies are the same SSE2NEON-based templates as vfunctions.cpp, so the results are identical by construction.

#if defined(__aarch64__) && defined(__ARM_NEON)

#include "../state.hpp"
#include "../rsp_op.hpp"
#include "rsp_impl.h"
#include "vu_inline.h"
#include "vu_neon.h"

register RSP::CPUState *g_vu_state asm("x19");

#define VU_STUB(name) extern "C" __attribute__((noinline, used, visibility("hidden"))) rsp_vect_t RSP_VUS_##name(rsp_vect_t vsv, rsp_vect_t vtv)

VU_STUB(VMULF)
	{
		uint16_t *acc = g_vu_state->cp2.acc.e;
		rsp_vect_t acc_lo, acc_md, acc_hi, result;

		result = rsp_vmulf_vmulu<false>(vsv, vtv, rsp_vzero(), &acc_lo, &acc_md, &acc_hi);

		write_acc_lo(acc, acc_lo);
		write_acc_md(acc, acc_md);
		write_acc_hi(acc, acc_hi);
		return result;
	}

VU_STUB(VMULU)
	{
		uint16_t *acc = g_vu_state->cp2.acc.e;
		rsp_vect_t acc_lo, acc_md, acc_hi, result;

		result = rsp_vmulf_vmulu<true>(vsv, vtv, rsp_vzero(), &acc_lo, &acc_md, &acc_hi);

		write_acc_lo(acc, acc_lo);
		write_acc_md(acc, acc_md);
		write_acc_hi(acc, acc_hi);
		return result;
	}

VU_STUB(VMUDL)
	{
		uint16_t *acc = g_vu_state->cp2.acc.e;
		using namespace vun;
		u16x8 acc_lo = vld1q_u16(acc + 16), acc_md = vld1q_u16(acc + 8), acc_hi = vld1q_u16(acc);
		u16x8 result = vmadl_vmudl<false>(vreinterpretq_u16_s32(vsv), vreinterpretq_u16_s32(vtv), acc_lo, acc_md, acc_hi);
		vst1q_u16(acc + 16, acc_lo);
		vst1q_u16(acc + 8, acc_md);
		vst1q_u16(acc, acc_hi);
		return vreinterpretq_s32_u16(result);
	}

VU_STUB(VMUDM)
	{
		uint16_t *acc = g_vu_state->cp2.acc.e;
		using namespace vun;
		u16x8 acc_lo = vld1q_u16(acc + 16), acc_md = vld1q_u16(acc + 8), acc_hi = vld1q_u16(acc);
		u16x8 result = vmadm_vmudm<false>(vreinterpretq_u16_s32(vsv), vreinterpretq_u16_s32(vtv), acc_lo, acc_md, acc_hi);
		vst1q_u16(acc + 16, acc_lo);
		vst1q_u16(acc + 8, acc_md);
		vst1q_u16(acc, acc_hi);
		return vreinterpretq_s32_u16(result);
	}

VU_STUB(VMUDN)
	{
		uint16_t *acc = g_vu_state->cp2.acc.e;
		using namespace vun;
		u16x8 acc_lo = vld1q_u16(acc + 16), acc_md = vld1q_u16(acc + 8), acc_hi = vld1q_u16(acc);
		u16x8 result = vmadn_vmudn<false>(vreinterpretq_u16_s32(vsv), vreinterpretq_u16_s32(vtv), acc_lo, acc_md, acc_hi);
		vst1q_u16(acc + 16, acc_lo);
		vst1q_u16(acc + 8, acc_md);
		vst1q_u16(acc, acc_hi);
		return vreinterpretq_s32_u16(result);
	}

VU_STUB(VMUDH)
	{
		uint16_t *acc = g_vu_state->cp2.acc.e;
		using namespace vun;
		u16x8 acc_lo = vld1q_u16(acc + 16), acc_md = vld1q_u16(acc + 8), acc_hi = vld1q_u16(acc);
		u16x8 result = vmadh_vmudh<false>(vreinterpretq_u16_s32(vsv), vreinterpretq_u16_s32(vtv), acc_lo, acc_md, acc_hi);
		vst1q_u16(acc + 16, acc_lo);
		vst1q_u16(acc + 8, acc_md);
		vst1q_u16(acc, acc_hi);
		return vreinterpretq_s32_u16(result);
	}

VU_STUB(VMACF)
	{
		uint16_t *acc = g_vu_state->cp2.acc.e;
		using namespace vun;
		u16x8 acc_lo = vld1q_u16(acc + 16), acc_md = vld1q_u16(acc + 8), acc_hi = vld1q_u16(acc);
		u16x8 result = vmacf_vmacu<false>(vreinterpretq_u16_s32(vsv), vreinterpretq_u16_s32(vtv), acc_lo, acc_md, acc_hi);
		vst1q_u16(acc + 16, acc_lo);
		vst1q_u16(acc + 8, acc_md);
		vst1q_u16(acc, acc_hi);
		return vreinterpretq_s32_u16(result);
	}

VU_STUB(VMACU)
	{
		uint16_t *acc = g_vu_state->cp2.acc.e;
		using namespace vun;
		u16x8 acc_lo = vld1q_u16(acc + 16), acc_md = vld1q_u16(acc + 8), acc_hi = vld1q_u16(acc);
		u16x8 result = vmacf_vmacu<true>(vreinterpretq_u16_s32(vsv), vreinterpretq_u16_s32(vtv), acc_lo, acc_md, acc_hi);
		vst1q_u16(acc + 16, acc_lo);
		vst1q_u16(acc + 8, acc_md);
		vst1q_u16(acc, acc_hi);
		return vreinterpretq_s32_u16(result);
	}

VU_STUB(VMADL)
	{
		uint16_t *acc = g_vu_state->cp2.acc.e;
		using namespace vun;
		u16x8 acc_lo = vld1q_u16(acc + 16), acc_md = vld1q_u16(acc + 8), acc_hi = vld1q_u16(acc);
		u16x8 result = vmadl_vmudl<true>(vreinterpretq_u16_s32(vsv), vreinterpretq_u16_s32(vtv), acc_lo, acc_md, acc_hi);
		vst1q_u16(acc + 16, acc_lo);
		vst1q_u16(acc + 8, acc_md);
		vst1q_u16(acc, acc_hi);
		return vreinterpretq_s32_u16(result);
	}

VU_STUB(VMADM)
	{
		uint16_t *acc = g_vu_state->cp2.acc.e;
		using namespace vun;
		u16x8 acc_lo = vld1q_u16(acc + 16), acc_md = vld1q_u16(acc + 8), acc_hi = vld1q_u16(acc);
		u16x8 result = vmadm_vmudm<true>(vreinterpretq_u16_s32(vsv), vreinterpretq_u16_s32(vtv), acc_lo, acc_md, acc_hi);
		vst1q_u16(acc + 16, acc_lo);
		vst1q_u16(acc + 8, acc_md);
		vst1q_u16(acc, acc_hi);
		return vreinterpretq_s32_u16(result);
	}

VU_STUB(VMADN)
	{
		uint16_t *acc = g_vu_state->cp2.acc.e;
		using namespace vun;
		u16x8 acc_lo = vld1q_u16(acc + 16), acc_md = vld1q_u16(acc + 8), acc_hi = vld1q_u16(acc);
		u16x8 result = vmadn_vmudn<true>(vreinterpretq_u16_s32(vsv), vreinterpretq_u16_s32(vtv), acc_lo, acc_md, acc_hi);
		vst1q_u16(acc + 16, acc_lo);
		vst1q_u16(acc + 8, acc_md);
		vst1q_u16(acc, acc_hi);
		return vreinterpretq_s32_u16(result);
	}

VU_STUB(VMADH)
	{
		uint16_t *acc = g_vu_state->cp2.acc.e;
		using namespace vun;
		u16x8 acc_lo = vld1q_u16(acc + 16), acc_md = vld1q_u16(acc + 8), acc_hi = vld1q_u16(acc);
		u16x8 result = vmadh_vmudh<true>(vreinterpretq_u16_s32(vsv), vreinterpretq_u16_s32(vtv), acc_lo, acc_md, acc_hi);
		vst1q_u16(acc + 16, acc_lo);
		vst1q_u16(acc + 8, acc_md);
		vst1q_u16(acc, acc_hi);
		return vreinterpretq_s32_u16(result);
	}

VU_STUB(VADD)
	{
		uint16_t *acc = g_vu_state->cp2.acc.e;
		rsp_vect_t carry, acc_lo;

		carry = read_vco_lo(g_vu_state->cp2.flags[RSP::RSP_VCO].e);
		rsp_vect_t result = rsp_vadd(vsv, vtv, carry, &acc_lo);

		write_vco_hi(g_vu_state->cp2.flags[RSP::RSP_VCO].e, rsp_vzero());
		write_vco_lo(g_vu_state->cp2.flags[RSP::RSP_VCO].e, rsp_vzero());
		write_acc_lo(acc, acc_lo);
		return result;
	}

VU_STUB(VSUB)
	{
		uint16_t *acc = g_vu_state->cp2.acc.e;
		rsp_vect_t carry, acc_lo;

		carry = read_vco_lo(g_vu_state->cp2.flags[RSP::RSP_VCO].e);

		rsp_vect_t result = rsp_vsub(vsv, vtv, carry, &acc_lo);

		write_vco_hi(g_vu_state->cp2.flags[RSP::RSP_VCO].e, rsp_vzero());
		write_vco_lo(g_vu_state->cp2.flags[RSP::RSP_VCO].e, rsp_vzero());
		write_acc_lo(acc, acc_lo);
		return result;
	}

VU_STUB(VABS)
	{
		uint16_t *acc = g_vu_state->cp2.acc.e;
		rsp_vect_t acc_lo;
		rsp_vect_t result = rsp_vabs(vsv, vtv, &acc_lo);
		write_acc_lo(acc, acc_lo);
		return result;
	}

VU_STUB(VADDC)
	{
		uint16_t *acc = g_vu_state->cp2.acc.e;
		rsp_vect_t sn;

		rsp_vect_t result = rsp_vaddc(vsv, vtv, rsp_vzero(), &sn);
		write_vco_hi(g_vu_state->cp2.flags[RSP::RSP_VCO].e, rsp_vzero()); // TODO: Confirm.
		write_vco_lo(g_vu_state->cp2.flags[RSP::RSP_VCO].e, sn);
		write_acc_lo(acc, result);
		return result;
	}

VU_STUB(VSUBC)
	{
		uint16_t *acc = g_vu_state->cp2.acc.e;
		rsp_vect_t eq, sn;

		rsp_vect_t result = rsp_vsubc(vsv, vtv, rsp_vzero(), &eq, &sn);

		write_vco_hi(g_vu_state->cp2.flags[RSP::RSP_VCO].e, eq);
		write_vco_lo(g_vu_state->cp2.flags[RSP::RSP_VCO].e, sn);
		write_acc_lo(acc, result);
		return result;
	}

VU_STUB(VLT)
	{
		uint16_t *acc = g_vu_state->cp2.acc.e;
		rsp_vect_t le, eq, sign;

		eq = read_vco_hi(g_vu_state->cp2.flags[RSP::RSP_VCO].e);
		sign = read_vco_lo(g_vu_state->cp2.flags[RSP::RSP_VCO].e);

		rsp_vect_t result = rsp_vlt(vsv, vtv, rsp_vzero(), &le, eq, sign);

		write_vcc_hi(g_vu_state->cp2.flags[RSP::RSP_VCC].e, rsp_vzero());
		write_vcc_lo(g_vu_state->cp2.flags[RSP::RSP_VCC].e, le);
		write_vco_hi(g_vu_state->cp2.flags[RSP::RSP_VCO].e, rsp_vzero());
		write_vco_lo(g_vu_state->cp2.flags[RSP::RSP_VCO].e, rsp_vzero());
		write_acc_lo(acc, result);
		return result;
	}

VU_STUB(VEQ)
	{
		uint16_t *acc = g_vu_state->cp2.acc.e;
		rsp_vect_t le, eq, sign;

		eq = read_vco_hi(g_vu_state->cp2.flags[RSP::RSP_VCO].e);
		sign = read_vco_lo(g_vu_state->cp2.flags[RSP::RSP_VCO].e);

		rsp_vect_t result = rsp_veq(vsv, vtv, rsp_vzero(), &le, eq, sign);

		write_vcc_hi(g_vu_state->cp2.flags[RSP::RSP_VCC].e, rsp_vzero());
		write_vcc_lo(g_vu_state->cp2.flags[RSP::RSP_VCC].e, le);
		write_vco_hi(g_vu_state->cp2.flags[RSP::RSP_VCO].e, rsp_vzero());
		write_vco_lo(g_vu_state->cp2.flags[RSP::RSP_VCO].e, rsp_vzero());
		write_acc_lo(acc, result);
		return result;
	}

VU_STUB(VNE)
	{
		uint16_t *acc = g_vu_state->cp2.acc.e;
		rsp_vect_t le, eq, sign;

		eq = read_vco_hi(g_vu_state->cp2.flags[RSP::RSP_VCO].e);
		sign = read_vco_lo(g_vu_state->cp2.flags[RSP::RSP_VCO].e);

		rsp_vect_t result = rsp_vne(vsv, vtv, rsp_vzero(), &le, eq, sign);

		write_vcc_hi(g_vu_state->cp2.flags[RSP::RSP_VCC].e, rsp_vzero());
		write_vcc_lo(g_vu_state->cp2.flags[RSP::RSP_VCC].e, le);
		write_vco_hi(g_vu_state->cp2.flags[RSP::RSP_VCO].e, rsp_vzero());
		write_vco_lo(g_vu_state->cp2.flags[RSP::RSP_VCO].e, rsp_vzero());
		write_acc_lo(acc, result);
		return result;
	}

VU_STUB(VGE)
	{
		uint16_t *acc = g_vu_state->cp2.acc.e;
		rsp_vect_t le, eq, sign;

		eq = read_vco_hi(g_vu_state->cp2.flags[RSP::RSP_VCO].e);
		sign = read_vco_lo(g_vu_state->cp2.flags[RSP::RSP_VCO].e);

		rsp_vect_t result = rsp_vge(vsv, vtv, rsp_vzero(), &le, eq, sign);

		write_vcc_hi(g_vu_state->cp2.flags[RSP::RSP_VCC].e, rsp_vzero());
		write_vcc_lo(g_vu_state->cp2.flags[RSP::RSP_VCC].e, le);
		write_vco_hi(g_vu_state->cp2.flags[RSP::RSP_VCO].e, rsp_vzero());
		write_vco_lo(g_vu_state->cp2.flags[RSP::RSP_VCO].e, rsp_vzero());
		write_acc_lo(acc, result);
		return result;
	}

VU_STUB(VCL)
	{
		uint16_t *acc = g_vu_state->cp2.acc.e;
		rsp_vect_t ge, le, eq, sign, vce;

		ge = read_vcc_hi(g_vu_state->cp2.flags[RSP::RSP_VCC].e);
		le = read_vcc_lo(g_vu_state->cp2.flags[RSP::RSP_VCC].e);
		eq = read_vco_hi(g_vu_state->cp2.flags[RSP::RSP_VCO].e);
		sign = read_vco_lo(g_vu_state->cp2.flags[RSP::RSP_VCO].e);
		vce = read_vce(g_vu_state->cp2.flags[RSP::RSP_VCE].e);

		rsp_vect_t result = rsp_vcl(vsv, vtv, rsp_vzero(), &ge, &le, eq, sign, vce);

		write_vcc_hi(g_vu_state->cp2.flags[RSP::RSP_VCC].e, ge);
		write_vcc_lo(g_vu_state->cp2.flags[RSP::RSP_VCC].e, le);
		write_vco_hi(g_vu_state->cp2.flags[RSP::RSP_VCO].e, rsp_vzero());
		write_vco_lo(g_vu_state->cp2.flags[RSP::RSP_VCO].e, rsp_vzero());
		write_vce(g_vu_state->cp2.flags[RSP::RSP_VCE].e, rsp_vzero());
		write_acc_lo(acc, result);
		return result;
	}

VU_STUB(VCH)
	{
		uint16_t *acc = g_vu_state->cp2.acc.e;
		rsp_vect_t ge, le, sign, eq, vce;

		rsp_vect_t result = rsp_vch(vsv, vtv, rsp_vzero(), &ge, &le, &eq, &sign, &vce);

		write_vcc_hi(g_vu_state->cp2.flags[RSP::RSP_VCC].e, ge);
		write_vcc_lo(g_vu_state->cp2.flags[RSP::RSP_VCC].e, le);
		write_vco_hi(g_vu_state->cp2.flags[RSP::RSP_VCO].e, eq);
		write_vco_lo(g_vu_state->cp2.flags[RSP::RSP_VCO].e, sign);
		write_vce(g_vu_state->cp2.flags[RSP::RSP_VCE].e, vce);
		write_acc_lo(acc, result);
		return result;
	}

VU_STUB(VCR)
	{
		uint16_t *acc = g_vu_state->cp2.acc.e;
		rsp_vect_t ge, le;

		rsp_vect_t result = rsp_vcr(vsv, vtv, rsp_vzero(), &ge, &le);


		write_vcc_hi(g_vu_state->cp2.flags[RSP::RSP_VCC].e, ge);
		write_vcc_lo(g_vu_state->cp2.flags[RSP::RSP_VCC].e, le);
		write_vco_hi(g_vu_state->cp2.flags[RSP::RSP_VCO].e, rsp_vzero());
		write_vco_lo(g_vu_state->cp2.flags[RSP::RSP_VCO].e, rsp_vzero());
		write_vce(g_vu_state->cp2.flags[RSP::RSP_VCE].e, rsp_vzero());
		write_acc_lo(acc, result);
		return result;
	}

VU_STUB(VMRG)
	{
		uint16_t *acc = g_vu_state->cp2.acc.e;
		rsp_vect_t le;

		le = read_vcc_lo(g_vu_state->cp2.flags[RSP::RSP_VCC].e);
		rsp_vect_t result = rsp_vmrg(vsv, vtv, le);
		write_vco_hi(g_vu_state->cp2.flags[RSP::RSP_VCO].e, rsp_vzero());
		write_vco_lo(g_vu_state->cp2.flags[RSP::RSP_VCO].e, rsp_vzero());
		write_acc_lo(acc, result);
		return result;
	}

VU_STUB(VAND)
	{
		uint16_t *acc = g_vu_state->cp2.acc.e;
		rsp_vect_t result = rsp_vand(vsv, vtv);
		write_acc_lo(acc, result);
		return result;
	}

VU_STUB(VNAND)
	{
		uint16_t *acc = g_vu_state->cp2.acc.e;
		rsp_vect_t result = rsp_vnand(vsv, vtv);
		write_acc_lo(acc, result);
		return result;
	}

VU_STUB(VOR)
	{
		uint16_t *acc = g_vu_state->cp2.acc.e;

		rsp_vect_t result = rsp_vor(vsv, vtv);

		write_acc_lo(acc, result);
		return result;
	}

VU_STUB(VNOR)
	{
		uint16_t *acc = g_vu_state->cp2.acc.e;

		rsp_vect_t result = rsp_vnor(vsv, vtv);

		write_acc_lo(acc, result);
		return result;
	}

VU_STUB(VXOR)
	{
		uint16_t *acc = g_vu_state->cp2.acc.e;

		rsp_vect_t result = rsp_vxor(vsv, vtv);

		write_acc_lo(acc, result);
		return result;
	}

VU_STUB(VNXOR)
	{
		uint16_t *acc = g_vu_state->cp2.acc.e;

		rsp_vect_t result = rsp_vnxor(vsv, vtv);

		write_acc_lo(acc, result);
		return result;
	}

	// Reserved opcodes: acc_lo = vs + vt, result 0.
	VU_STUB(RESERVED)
	{
		write_acc_lo(g_vu_state->cp2.acc.e, _mm_add_epi16(vsv, vtv));
		return rsp_vzero();
	}

#define X(name) extern "C" rsp_vect_t RSP_VUS_##name(rsp_vect_t, rsp_vect_t);
RSP_VU_STUB_LIST(X)
#undef X

extern "C" const void *const rsp_vu_stub_fn[RSP_VU_STUB_COUNT] = {
#define X(name) reinterpret_cast<const void *>(&RSP_VUS_##name),
RSP_VU_STUB_LIST(X)
#undef X
};

extern "C" const char *const rsp_vu_stub_name[RSP_VU_STUB_COUNT] = {
#define X(name) #name,
RSP_VU_STUB_LIST(X)
#undef X
};

#endif
