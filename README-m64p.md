# parallel-rsp as a Mupen64Plus RSP plugin (aarch64 work)

`mupen64plus-rsp-parallel.so` (CMake option `PARALLEL_RSP_M64P_PLUGIN`, `./build_m64p_plugin.sh [aarch64]`) is the
parallel-rsp Lightning JIT behind the Mupen64Plus RSP plugin API. It runs standalone (LLE graphics and audio) or as
`RspFallback` of `mupen64plus-rsp-hle` (HLE audio, LLE graphics). Pairing results with the foir RDP plugin:
`mupen64plus-rdp-foir/README.md`, section "RSP pairing".

Build flags are unchanged (Release, `-O3`, generic armv8-a; `-mcpu` / `-march` made no difference).

## What changed for aarch64 (all in the JIT, behaviour is bit-identical to the old code generation)

The original JIT called a C helper for every vector (VU/COP2) instruction and every vector load/store. Profiling Super Mario 64
showed ~46 % of the emulation thread in those helpers plus their call overhead; the helpers themselves were already tight NEON.
What was lost was call overhead, argument marshalling, flushing the scalar register cache around every call, address
decoding inside the helpers, a hash-table lookup on every block transition and mostly 4 KiB of padding between code regions.

| Change | Where |
|---|---|
| **VU ops with a register calling convention.** Each hot VU op body (`rsp/vu_inline.cpp`, the same SSE2NEON-derived code as `vfunctions.cpp`, or native NEON in `rsp/vu_neon.h` for the multiply-accumulate family) is a C function taking `vs` in `v0`, the element-shuffled `vt` in `v1` and returning `vd` in `v0`; the CPUState pointer is `x19` (a global register variable, which is where the JIT keeps it). At start-up `CPU::init_vu_stubs` validates the compiled machine code (straight-line, no branches / pc-relative / stack access, only scratch registers), copies it into the JIT code region and the JIT emits `ldr q0 / ldr q1 / [dup or 2x trn] / bl stub / str q0` (small bodies are copied inline). The element shuffles (`0q 1q 0h..3h 0w..7w`) are `trn1/trn2/dup`, no table. Nothing but x0-x8/x16/x17 and v-registers is clobbered, so **the scalar register cache is no longer flushed around VU ops** (the file is compiled with `-ffixed-x9..x15,x18,x20..x28`). A body that fails validation falls back to the old helper call. `v0` is tracked across straight-line code so back-to-back dependent ops skip the reload. | `rsp_jit.cpp` (`jit_emit_vu_fast`), `rsp/vu_inline.cpp`, `rsp/vu_neon.h`, `rsp/neon_enc.h` |
| **Cheaper accumulate / clamp sequences.** Carry out of a 16-bit add is `cmhi(a, a+b)` instead of `uqadd/cmeq/mvn`; the unsigned / signed clamps of the 48-bit accumulator are `hi == (md >> 15) ? ... : ...` (4 instructions instead of 7-10 with the emulated pack/unpack). VMADN 30 -> 24 instructions, VMADH 22 -> 15, VMADM 34 -> 23, VMADL 24 -> 17, VMACF 38 -> 26. | `rsp/vu_neon.h` |
| **Vector loads / stores inline.** LBV/SBV, LSV/SSV, LLV/SLV, LDV/SDV, LQV/SQV, LPV/LUV/SPV/SUV (the element/alignment combinations ucode actually uses) are a handful of instructions: address arithmetic in JIT registers, a single alignment test, `ldr d/q [dmem, addr]`, `rev32` (the DMEM word swap), store. Anything else branches to an **out-of-line slow path** that spills only the live caller-save registers, calls the original helper through a new `RSP_xxx_A(rsp, rt, e, addr)` entry (same code, address already computed) and reloads. Unaligned LQV (`e == 0`) is done in NEON with `tbl`. | `rsp_jit.cpp` (`jit_emit_ls_fast`, `jit_emit_slow_paths`), `rsp/ls.cpp` |
| MTC2/MFC2, VMOV, VRCPH/VRSQH, VSAR, VNOP, MFC0 (not SP_STATUS), MTC0 DMA_CACHE/DMA_DRAM | emitted inline (plain loads/stores/NEON, no helper) | `rsp_jit.cpp` |
| **Block dispatch.** The enter thunk looks the target up in `CPU::blocks[]` itself and only calls into C for blocks that are not compiled yet. | `CPU::init_jit_thunks` |
| **Code layout.** Regions were page aligned (about 6x padding: 516 KiB of pages for 88 KiB of code in SM64, and every region entry on the same L1I sets). They are now packed back to back, 64-byte aligned, with the size estimate trimmed after emission. L1I refills -35 %, I-TLB refills -50 % in `perf stat`. | `jit_allocator.cpp`, `CPU::jit_region` |
| **SP DMA** word loops became `memcpy` per row (wrapping / straddling rows keep the old loop). | `rsp/cp0.cpp` |
| Lightning: `jit_raw(word)` emits a raw 32-bit instruction (a `jit_skip` node with a payload; aarch64 backend only). | `lightning/` |

The x86 / non-NEON build is unchanged (`PRSP_VU_FAST` is only defined for aarch64 with NEON; `-DPARALLEL_RSP_NO_VU_FAST` turns it off).

## RSP exit handling (`parallelRSPDoRspCycles`)

When the JIT returns with `SP_STATUS_HALT` set but without `SP_STATUS_BROKE` there are two different causes:

* the ucode halted itself without a BREAK (the "RSP hang when setting HALT but not BREAK" fix): `DoRspCycles` returns and leaves HALT set, the CPU restarts the RSP;
* `RSP_MFC0` forced the HALT because the ucode polled `SP_STATUS` more than `SP_STATUS_TIMEOUT` times (it is waiting for the CPU to set a signal bit).
  This is the CXD4 `WAIT_FOR_CPU_HOST` path: HALT has to be **cleared again** before returning, so that the core sees a running RSP, schedules the SP
  interrupt and calls `DoRspCycles` again once the CPU has run.

The first version of the HALT fix treated both the same, so a task that idles on `SP_STATUS` waiting for the CPU was never resumed. Gauntlet Legends
(graphics ucode, task type 1) is the known victim: it sent 10 RDP lists and a black screen, with parallel-rsp as the fallback of rsp-hle or as the only RSP
(cxd4 rendered it). `DoRspCycles` now checks the per-register `MFC0_count[]` against the time-out to tell the two cases apart. SM64, Banjo-Kazooie and
Ocarina of Time are bit-identical (frames 300-1200, `Core[RandomizeInterrupt]=False`) with and without the change.

## Verifying bit-exactness

* `tests/jit_diff.cpp` (`-DPARALLEL_RSP_JIT_DIFF=ON`, built for aarch64 and run with `qemu-aarch64 -L /usr/aarch64-linux-gnu`): the same random
  RSP programs (VU ops incl. reserved opcodes, all elements and aliasing, vector loads/stores with random alignment, COP2 / COP0 moves,
  scalar ALU, branches) run through a JIT using the new code generation and one using the old helper calls
  (`CPU::set_vu_fast(false)`); full state (scalar and vector registers, accumulator, flags, divide state, DMEM, PC, delay slot, CP0 registers)
  is compared after every program. Modes `single` (every opcode x element), `vu` (chains of VU ops), `random`, `all`.
  The test was checked to catch deliberately wrong code generation (one-bit mutations in the shuffle, the clamp, ...).
* `tests/dma_diff.sh`: the DMA fast path against the original loops (`tests/cp0_reference.cpp`), random registers and rows that wrap / straddle.
* `tests/neon_enc_check.sh`: every raw instruction encoder in `rsp/neon_enc.h` against `aarch64-linux-gnu-as`.
* End to end on the device: `mupen64plus-rdp-foir/tools/prsp/prsp_cmp.sh` (screenshots at VI frames 300/900/1500/2100/2700 and a hash of DMEM+IMEM after every
  `DoRspCycles` call, new vs old `.so`, `Core[RandomizeInterrupt]=False`) and `prsp_cmp_audio.sh` (LLE audio stream of the standalone plugin).

## Debug / tuning environment variables

| Variable | Effect |
|---|---|
| `PRSP_NO_VU_FAST=1` | use the helper-call code generation for VU ops, vector loads/stores and COP0/COP2 moves (A/B comparisons) |
| `PRSP_INLINE_WORDS=n` | copy VU bodies of up to n instructions inline instead of calling the shared copy (default 8) |
| `PRSP_DEBUG_VU=1` | print the validation result for every VU body at start-up |
| `PRSP_DUMP_JIT=file` | write an image of the code region (use with `objdump -D -b binary -m aarch64` and `perf script`) |
| build with `PRSP_PROFILE=1 ./build_m64p_plugin.sh aarch64` | dynamic instruction mix, VU / LS run lengths, slow-path histograms (`PRSP_PROF_OUT=file`) |
