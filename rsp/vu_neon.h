// Native NEON versions of the accumulating VU multiply ops for the register-convention bodies in vu_inline.cpp.
//
// They compute exactly what the SSE-style templates in arch/simd/rsp/*.h compute (those remain the reference: the helper-call
// path in vfunctions.cpp still uses them, and tests/jit_diff.cpp compares the two). The savings come from
//  * unsigned carry detection with one cmhi (sum < addend  <=>  carry) instead of uqadd / cmeq / mvn,
//  * the clamps written as "no clamp iff hi == md >> 15" (see uclamp / sclamp below),
//  * zip + sqxtn written directly instead of through SSE2NEON's unpack / pack emulation.
#ifndef RSP_VU_NEON_H
#define RSP_VU_NEON_H

#include <arm_neon.h>

namespace vun
{
using u16x8 = uint16x8_t;
using s16x8 = int16x8_t;

static inline u16x8 mulhi_u(u16x8 a, u16x8 b)
{
	uint32x4_t l = vmull_u16(vget_low_u16(a), vget_low_u16(b));
	uint32x4_t h = vmull_high_u16(a, b);
	return vuzp2q_u16(vreinterpretq_u16_u32(l), vreinterpretq_u16_u32(h));
}

static inline u16x8 mulhi_s(u16x8 a, u16x8 b)
{
	int32x4_t l = vmull_s16(vget_low_s16(vreinterpretq_s16_u16(a)), vget_low_s16(vreinterpretq_s16_u16(b)));
	int32x4_t h = vmull_high_s16(vreinterpretq_s16_u16(a), vreinterpretq_s16_u16(b));
	return vuzp2q_u16(vreinterpretq_u16_s32(l), vreinterpretq_u16_s32(h));
}

static inline u16x8 vcltzq_s16_u(u16x8 a)
{
	return vcltzq_s16(vreinterpretq_s16_u16(a));
}

// Signed saturation of the 32-bit value (hi:md) to 16 bits (rsp_sclamp_acc_tomd).
static inline u16x8 sclamp_tomd(u16x8 md, u16x8 hi)
{
	s16x8 m = vreinterpretq_s16_u16(md), h = vreinterpretq_s16_u16(hi);
	int32x4_t l = vreinterpretq_s32_s16(vzip1q_s16(m, h));
	int32x4_t u = vreinterpretq_s32_s16(vzip2q_s16(m, h));
	return vreinterpretq_u16_s16(vqmovn_high_s32(vqmovn_s32(l), u));
}

// rsp_uclamp_acc(lo, md, hi): lo when the 48-bit accumulator fits in a signed 32-bit value (hi == sign of md), otherwise 0xffff
// for a positive and 0 for a negative accumulator.
static inline u16x8 uclamp(u16x8 lo, u16x8 md, u16x8 hi)
{
	s16x8 h = vreinterpretq_s16_u16(hi);
	u16x8 fits = vceqq_s16(h, vshrq_n_s16(vreinterpretq_s16_u16(md), 15));
	return vbslq_u16(fits, lo, vcgezq_s16(h));
}

// a + b with the carry out of the 16-bit add as an all-ones mask.
static inline u16x8 add_carry(u16x8 a, u16x8 b, u16x8 &carry_mask)
{
	u16x8 s = vaddq_u16(a, b);
	carry_mask = vcgtq_u16(a, s);
	return s;
}

// VMADN / VMUDN
template <bool ACCUMULATE>
static inline u16x8 vmadn_vmudn(u16x8 vs, u16x8 vt, u16x8 &acc_lo, u16x8 &acc_md, u16x8 &acc_hi)
{
	u16x8 lo = vmulq_u16(vs, vt);
	u16x8 hi = mulhi_u(vs, vt);
	// unsigned vs * signed vt: subtract vs from the high half when vt is negative
	hi = vsubq_u16(hi, vandq_u16(vs, vcltzq_s16_u(vt)));
	if (ACCUMULATE)
	{
		u16x8 c;
		acc_lo = add_carry(acc_lo, lo, c);
		hi = vsubq_u16(hi, c);
		acc_md = add_carry(acc_md, hi, c);
		acc_hi = vreinterpretq_u16_s16(vsraq_n_s16(vreinterpretq_s16_u16(acc_hi), vreinterpretq_s16_u16(hi), 15));
		acc_hi = vsubq_u16(acc_hi, c);
		return uclamp(acc_lo, acc_md, acc_hi);
	}
	acc_lo = lo;
	acc_md = hi;
	acc_hi = vreinterpretq_u16_s16(vshrq_n_s16(vreinterpretq_s16_u16(hi), 15));
	return lo;
}

// VMADM / VMUDM
template <bool ACCUMULATE>
static inline u16x8 vmadm_vmudm(u16x8 vs, u16x8 vt, u16x8 &acc_lo, u16x8 &acc_md, u16x8 &acc_hi)
{
	u16x8 lo = vmulq_u16(vs, vt);
	u16x8 hi = mulhi_u(vs, vt);
	// signed vs * unsigned vt (VMADM multiplies the signed vs by the unsigned vt)
	hi = vsubq_u16(hi, vandq_u16(vt, vcltzq_s16_u(vs)));
	if (ACCUMULATE)
	{
		u16x8 c;
		acc_lo = add_carry(acc_lo, lo, c);
		hi = vsubq_u16(hi, c);
		acc_md = add_carry(acc_md, hi, c);
		acc_hi = vreinterpretq_u16_s16(vsraq_n_s16(vreinterpretq_s16_u16(acc_hi), vreinterpretq_s16_u16(hi), 15));
		acc_hi = vsubq_u16(acc_hi, c);
		return sclamp_tomd(acc_md, acc_hi);
	}
	acc_lo = lo;
	acc_md = hi;
	acc_hi = vreinterpretq_u16_s16(vshrq_n_s16(vreinterpretq_s16_u16(hi), 15));
	return hi;
}

// VMADH / VMUDH
template <bool ACCUMULATE>
static inline u16x8 vmadh_vmudh(u16x8 vs, u16x8 vt, u16x8 &acc_lo, u16x8 &acc_md, u16x8 &acc_hi)
{
	u16x8 lo = vmulq_u16(vs, vt);
	u16x8 hi = mulhi_s(vs, vt);
	if (ACCUMULATE)
	{
		u16x8 c;
		acc_md = add_carry(acc_md, lo, c);
		hi = vsubq_u16(hi, c);
		acc_hi = vaddq_u16(acc_hi, hi);
	}
	else
	{
		acc_lo = vdupq_n_u16(0);
		acc_md = lo;
		acc_hi = hi;
	}
	return sclamp_tomd(acc_md, acc_hi);
}

// VMADL / VMUDL
template <bool ACCUMULATE>
static inline u16x8 vmadl_vmudl(u16x8 vs, u16x8 vt, u16x8 &acc_lo, u16x8 &acc_md, u16x8 &acc_hi)
{
	u16x8 hi = mulhi_u(vs, vt);
	if (ACCUMULATE)
	{
		u16x8 c, c2;
		acc_lo = add_carry(acc_lo, hi, c);
		// the carry adds 1 to md; md can only overflow when it was 0xffff
		u16x8 md = vsubq_u16(acc_md, c);
		c2 = vcgtq_u16(acc_md, md);
		acc_md = md;
		acc_hi = vsubq_u16(acc_hi, c2);
		return uclamp(acc_lo, acc_md, acc_hi);
	}
	acc_lo = hi;
	acc_md = vdupq_n_u16(0);
	acc_hi = vdupq_n_u16(0);
	return hi;
}

// VMACF / VMACU
template <bool VMACU>
static inline u16x8 vmacf_vmacu(u16x8 vs, u16x8 vt, u16x8 &acc_lo, u16x8 &acc_md, u16x8 &acc_hi)
{
	u16x8 lo = vmulq_u16(vs, vt);
	u16x8 hi = mulhi_s(vs, vt);

	// (vs * vt) << 1 as a 48-bit value: md = (hi << 1) | (lo >> 15), hi = sign of hi, lo = lo << 1
	u16x8 md = vsraq_n_u16(vshlq_n_u16(hi, 1), lo, 15);
	hi = vreinterpretq_u16_s16(vshrq_n_s16(vreinterpretq_s16_u16(hi), 15));
	lo = vshlq_n_u16(lo, 1);

	u16x8 c;
	acc_lo = add_carry(acc_lo, lo, c);
	md = vsubq_u16(md, c);
	// md wrapped around to zero by the carry: carry on into hi
	hi = vsubq_u16(hi, vandq_u16(vceqzq_u16(md), c));

	acc_md = add_carry(acc_md, md, c);
	acc_hi = vsubq_u16(vaddq_u16(acc_hi, hi), c);

	if (VMACU)
	{
		u16x8 hi_neg = vcltzq_s16_u(acc_hi);
		u16x8 md_neg = vcltzq_s16_u(acc_md);
		u16x8 r = vbicq_u16(vorrq_u16(md_neg, acc_md), hi_neg);
		return vorrq_u16(vcgtzq_s16(vreinterpretq_s16_u16(acc_hi)), r);
	}
	return sclamp_tomd(acc_md, acc_hi);
}
} // namespace vun

#endif
