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
}
}
#endif
