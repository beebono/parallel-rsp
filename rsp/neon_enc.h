#ifndef RSP_NEON_ENC_H
#define RSP_NEON_ENC_H
// Minimal AArch64 NEON / load-store encoders for the raw instruction words the JIT emits next to the
// copied VU bodies (rsp_jit.cpp). Checked against GNU as in tools/ (see parallel-rsp/tests/neon_enc_check.sh).
#include <stdint.h>

namespace RSP
{
namespace NeonEnc
{
// ldr/str qT, [xN, #byte_offset]  (offset multiple of 16, < 65536)
static inline uint32_t ldr_q(unsigned t, unsigned n, unsigned off) { return 0x3dc00000u | ((off >> 4) << 10) | (n << 5) | t; }
static inline uint32_t str_q(unsigned t, unsigned n, unsigned off) { return 0x3d800000u | ((off >> 4) << 10) | (n << 5) | t; }
// ldr/str dT, [xN, #byte_offset] (offset multiple of 8)
static inline uint32_t ldr_d(unsigned t, unsigned n, unsigned off) { return 0xfd400000u | ((off >> 3) << 10) | (n << 5) | t; }
static inline uint32_t str_d(unsigned t, unsigned n, unsigned off) { return 0xfd000000u | ((off >> 3) << 10) | (n << 5) | t; }
// mov vD.16b, vN.16b
static inline uint32_t mov16b(unsigned d, unsigned n) { return 0x4ea01c00u | (n << 16) | (n << 5) | d; }
// movi vD.4s, #0
static inline uint32_t movi_zero(unsigned d) { return 0x4f000400u | d; }
// trn1/trn2 vD.8h, vN.8h, vM.8h and .4s
static inline uint32_t trn1_8h(unsigned d, unsigned n, unsigned m) { return 0x4e402800u | (m << 16) | (n << 5) | d; }
static inline uint32_t trn2_8h(unsigned d, unsigned n, unsigned m) { return 0x4e406800u | (m << 16) | (n << 5) | d; }
static inline uint32_t trn1_4s(unsigned d, unsigned n, unsigned m) { return 0x4e802800u | (m << 16) | (n << 5) | d; }
static inline uint32_t trn2_4s(unsigned d, unsigned n, unsigned m) { return 0x4e806800u | (m << 16) | (n << 5) | d; }
// dup vD.8h, vN.h[idx]
static inline uint32_t dup_8h(unsigned d, unsigned n, unsigned idx) { return 0x4e000400u | (((idx << 2) | 2u) << 16) | (n << 5) | d; }

// Register-offset forms: [xN, xM] (no extend / shift), used with the DMEM base in x20 and a byte address register.
static inline uint32_t ldr_d_r(unsigned t, unsigned n, unsigned m) { return 0xfc606800u | (m << 16) | (n << 5) | t; }
static inline uint32_t str_d_r(unsigned t, unsigned n, unsigned m) { return 0xfc206800u | (m << 16) | (n << 5) | t; }
static inline uint32_t ldr_q_r(unsigned t, unsigned n, unsigned m) { return 0x3ce06800u | (m << 16) | (n << 5) | t; }
static inline uint32_t str_q_r(unsigned t, unsigned n, unsigned m) { return 0x3ca06800u | (m << 16) | (n << 5) | t; }
static inline uint32_t ldr_w_r(unsigned t, unsigned n, unsigned m) { return 0xb8606800u | (m << 16) | (n << 5) | t; }
static inline uint32_t str_w_r(unsigned t, unsigned n, unsigned m) { return 0xb8206800u | (m << 16) | (n << 5) | t; }
static inline uint32_t ldrh_r(unsigned t, unsigned n, unsigned m) { return 0x78606800u | (m << 16) | (n << 5) | t; }
static inline uint32_t strh_r(unsigned t, unsigned n, unsigned m) { return 0x78206800u | (m << 16) | (n << 5) | t; }
static inline uint32_t ldrb_r(unsigned t, unsigned n, unsigned m) { return 0x38606800u | (m << 16) | (n << 5) | t; }
static inline uint32_t strb_r(unsigned t, unsigned n, unsigned m) { return 0x38206800u | (m << 16) | (n << 5) | t; }
// Immediate forms with a scaled offset.
static inline uint32_t ldr_w(unsigned t, unsigned n, unsigned off) { return 0xb9400000u | ((off >> 2) << 10) | (n << 5) | t; }
static inline uint32_t str_w(unsigned t, unsigned n, unsigned off) { return 0xb9000000u | ((off >> 2) << 10) | (n << 5) | t; }
static inline uint32_t ldrh(unsigned t, unsigned n, unsigned off) { return 0x79400000u | ((off >> 1) << 10) | (n << 5) | t; }
static inline uint32_t strh(unsigned t, unsigned n, unsigned off) { return 0x79000000u | ((off >> 1) << 10) | (n << 5) | t; }
static inline uint32_t ldrb(unsigned t, unsigned n, unsigned off) { return 0x39400000u | (off << 10) | (n << 5) | t; }
static inline uint32_t strb(unsigned t, unsigned n, unsigned off) { return 0x39000000u | (off << 10) | (n << 5) | t; }
// add xD, xN, #imm12
static inline uint32_t add_imm(unsigned d, unsigned n, unsigned imm) { return 0x91000000u | (imm << 10) | (n << 5) | d; }
// ror wD, wN, #16 (EXTR wD, wN, wN, #16)
static inline uint32_t ror16_w(unsigned d, unsigned n) { return 0x13800000u | (n << 16) | (16u << 10) | (n << 5) | d; }
// rev32 vD.4h / vD.8h / vD.8b, vN
static inline uint32_t rev32_4h(unsigned d, unsigned n) { return 0x2e600800u | (n << 5) | d; }
static inline uint32_t rev32_8h(unsigned d, unsigned n) { return 0x6e600800u | (n << 5) | d; }
static inline uint32_t rev32_8b(unsigned d, unsigned n) { return 0x2e200800u | (n << 5) | d; }
// shll vD.8h, vN.8b, #8 ; ushll vD.8h, vN.8b, #7
static inline uint32_t shll_8h_8(unsigned d, unsigned n) { return 0x2e213800u | (n << 5) | d; }
static inline uint32_t ushll_8h_7(unsigned d, unsigned n) { return 0x2f0fa400u | (n << 5) | d; }
// shrn vD.8b, vN.8h, #shift (shift 7 or 8)
static inline uint32_t shrn_8b(unsigned d, unsigned n, unsigned shift) { return 0x0f088400u | ((8u - shift) << 16) | (n << 5) | d; }
// ext vD.16b, vN.16b, vM.16b, #imm
static inline uint32_t ext_16b(unsigned d, unsigned n, unsigned m, unsigned imm) { return 0x6e000000u | (m << 16) | (imm << 11) | (n << 5) | d; }
// Byte-granular helpers for unaligned vector loads (rev32/rev16 on .16b, tbl, add, cmhi, bsl, movi, dup from a W register).
static inline uint32_t rev32_16b(unsigned d, unsigned n) { return 0x6e200800u | (n << 5) | d; }
static inline uint32_t rev16_16b(unsigned d, unsigned n) { return 0x4e201800u | (n << 5) | d; }
static inline uint32_t dup_16b_w(unsigned d, unsigned wn) { return 0x4e010c00u | (wn << 5) | d; }
static inline uint32_t add_16b(unsigned d, unsigned n, unsigned m) { return 0x4e208400u | (m << 16) | (n << 5) | d; }
static inline uint32_t tbl_16b(unsigned d, unsigned n, unsigned m) { return 0x4e000000u | (m << 16) | (n << 5) | d; }
static inline uint32_t cmhi_16b(unsigned d, unsigned n, unsigned m) { return 0x6e203400u | (m << 16) | (n << 5) | d; }
static inline uint32_t bsl_16b(unsigned d, unsigned n, unsigned m) { return 0x6e601c00u | (m << 16) | (n << 5) | d; }
static inline uint32_t movi_16b(unsigned d, unsigned imm8) { return 0x4f00e400u | ((imm8 >> 5) << 16) | ((imm8 & 31) << 5) | d; }

// Scalar (W register) helpers used by the inlined VRCP / VRSQ sequences.
// orr wD, wN, wM, lsl #sh ; eor / sub wD, wN, wM
static inline uint32_t orr_w_lsl(unsigned d, unsigned n, unsigned m, unsigned sh) { return 0x2a000000u | (m << 16) | (sh << 10) | (n << 5) | d; }
static inline uint32_t eor_w(unsigned d, unsigned n, unsigned m) { return 0x4a000000u | (m << 16) | (n << 5) | d; }
static inline uint32_t sub_w(unsigned d, unsigned n, unsigned m) { return 0x4b000000u | (m << 16) | (n << 5) | d; }
// sxth wD, wN ; clz wD, wN ; lsl / lsr wD, wN, wM
static inline uint32_t sxth_w(unsigned d, unsigned n) { return 0x13003c00u | (n << 5) | d; }
static inline uint32_t clz_w(unsigned d, unsigned n) { return 0x5ac01000u | (n << 5) | d; }
static inline uint32_t lslv_w(unsigned d, unsigned n, unsigned m) { return 0x1ac02000u | (m << 16) | (n << 5) | d; }
static inline uint32_t lsrv_w(unsigned d, unsigned n, unsigned m) { return 0x1ac02400u | (m << 16) | (n << 5) | d; }
// asr / lsr / lsl wD, wN, #sh (1 <= sh <= 31)
static inline uint32_t asr_w_imm(unsigned d, unsigned n, unsigned sh) { return 0x13007c00u | (sh << 16) | (n << 5) | d; }
static inline uint32_t lsr_w_imm(unsigned d, unsigned n, unsigned sh) { return 0x53007c00u | (sh << 16) | (n << 5) | d; }
static inline uint32_t lsl_w_imm(unsigned d, unsigned n, unsigned sh) { return 0x53000000u | (((32u - sh) & 31u) << 16) | ((31u - sh) << 10) | (n << 5) | d; }
// ubfx wD, wN, #lsb, #width ; bfi wD, wN, #0, #width
static inline uint32_t ubfx_w(unsigned d, unsigned n, unsigned lsb, unsigned width) { return 0x53000000u | (lsb << 16) | ((lsb + width - 1) << 10) | (n << 5) | d; }
static inline uint32_t bfi0_w(unsigned d, unsigned n, unsigned width) { return 0x33000000u | ((width - 1) << 10) | (n << 5) | d; }
// csel wD, wN, wM, cond (0 = eq, 1 = ne, 12 = gt)
static inline uint32_t csel_w(unsigned d, unsigned n, unsigned m, unsigned cond) { return 0x1a800000u | (m << 16) | (cond << 12) | (n << 5) | d; }
// cmn wN, #imm12, lsl #12 ; cmp wN, #0
static inline uint32_t cmn_w_lsl12(unsigned n, unsigned imm) { return 0x31400000u | (imm << 10) | (n << 5) | 31u; }
static inline uint32_t cmp_w_0(unsigned n) { return 0x7100001fu | (n << 5); }
// and / orr / eor / ands wD, wN, #imm given as the (immr, imms) bitmask encoding with N = 0
static inline uint32_t and_w_bm(unsigned d, unsigned n, unsigned immr, unsigned imms) { return 0x12000000u | (immr << 16) | (imms << 10) | (n << 5) | d; }
static inline uint32_t orr_w_bm(unsigned d, unsigned n, unsigned immr, unsigned imms) { return 0x32000000u | (immr << 16) | (imms << 10) | (n << 5) | d; }
static inline uint32_t eor_w_bm(unsigned d, unsigned n, unsigned immr, unsigned imms) { return 0x52000000u | (immr << 16) | (imms << 10) | (n << 5) | d; }
static inline uint32_t ands_w_bm(unsigned d, unsigned n, unsigned immr, unsigned imms) { return 0x72000000u | (immr << 16) | (imms << 10) | (n << 5) | d; }
// ldr xT, [xN, #off] (offset multiple of 8) ; ldrh wT, [xN, wM, uxtw #1]
static inline uint32_t ldr_x(unsigned t, unsigned n, unsigned off) { return 0xf9400000u | ((off >> 3) << 10) | (n << 5) | t; }
static inline uint32_t ldrh_uxtw1(unsigned t, unsigned n, unsigned m) { return 0x78605800u | (m << 16) | (n << 5) | t; }
// add wD, wN, #imm12
static inline uint32_t add_w_imm(unsigned d, unsigned n, unsigned imm) { return 0x11000000u | (imm << 10) | (n << 5) | d; }
// NEON: sqxtn vD.8b, vN.8h ; sqxtn2 vD.16b, vN.8h ; sshr vD.16b, vN.16b, #7 ; and / zip1 vD.16b ; addv hD, vN.8h ; smov wD, vN.h[0]
static inline uint32_t sqxtn_8b(unsigned d, unsigned n) { return 0x0e214800u | (n << 5) | d; }
static inline uint32_t sqxtn2_16b(unsigned d, unsigned n) { return 0x4e214800u | (n << 5) | d; }
static inline uint32_t sshr_16b_7(unsigned d, unsigned n) { return 0x4f090400u | (n << 5) | d; }
static inline uint32_t and_16b(unsigned d, unsigned n, unsigned m) { return 0x4e201c00u | (m << 16) | (n << 5) | d; }
static inline uint32_t zip1_16b(unsigned d, unsigned n, unsigned m) { return 0x4e003800u | (m << 16) | (n << 5) | d; }
static inline uint32_t addv_h(unsigned d, unsigned n) { return 0x4e71b800u | (n << 5) | d; }
static inline uint32_t smov_w_h0(unsigned d, unsigned n) { return 0x0e022c00u | (n << 5) | d; }
}
}
#endif
