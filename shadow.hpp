// Shadow verification for the Mupen64Plus plugin (build with -DPRSP_SHADOW, e.g. PRSP_SHADOW=1 ./build_m64p_plugin.sh aarch64).
//
// Every DoRspCycles() call is run twice from the same starting state: once by the normal JIT (the real run, including the RDP
// callbacks), then again on a private copy of DMEM / IMEM / registers / RDRAM by a second JIT instance that is configured with the
// *previous* code paths: no inlined VRCP / VRSQ / CFC2 (PRSP_SHADOW_MODE=prev, the default) or the helper-call code generation for
// all VU and load / store ops (PRSP_SHADOW_MODE=old), and IMEM blocks dirtied by every SP DMA. Afterwards the two runs are compared:
// scalar and vector registers, accumulator, flags, divide state, PC, CP0 / DPC registers, DMEM, IMEM, the RDP command lists handed to
// the RDP (start, end, hash of the bytes), and every RDRAM range the task wrote with an SP DMA (address, length, hash).
// PRSP_SHADOW_OUT=<file> receives the summary (default stderr) when the plugin is unloaded.
#ifndef PRSP_SHADOW_HPP__
#define PRSP_SHADOW_HPP__
#ifdef PRSP_SHADOW
#include <stdint.h>

namespace RSP
{
namespace Shadow
{
void init();
bool enabled();
void begin(); // snapshot the state a DoRspCycles() call starts from
void end(); // replay the task on the shadow JIT and compare
} // namespace Shadow
} // namespace RSP

extern "C" void prsp_shadow_dma_write(const void *state, unsigned rdram_byte_addr, unsigned bytes);
#endif
#endif
