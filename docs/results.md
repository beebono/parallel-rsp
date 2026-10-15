# Measurements and verification results

Detail behind the summary in the [README](../README.md): what the profile showed, the effect of each change, and the checks
that show the new code generation is bit-identical to the old one. Device: RK3566 handheld (4x Cortex-A55), `performance` governor
(reports 1992 MHz, really about 1.58 GHz).

## Why the VU / load-store fast paths

Profiling Super Mario 64 showed about 46 % of the emulation thread in the C helpers for vector (VU/COP2) instructions and vector
load/stores, plus their call overhead; the helpers themselves were already tight NEON. What was lost was call overhead, argument
marshalling, flushing the scalar register cache around every call, address decoding inside the helpers, a hash-table lookup on every
block transition, and mostly 4 KiB of padding between code regions.

Details of the changes:

- **VU ops with a register calling convention.** Each hot VU op body (`rsp/vu_inline.cpp`, the SSE2NEON-derived code of `vfunctions.cpp`,
  or native NEON in `rsp/vu_neon.h` for the multiply-accumulate family) is a C function taking `vs` in `v0` and the element-shuffled
  `vt` in `v1` and returning `vd` in `v0`; the CPUState pointer is `x19` (a global register variable, which is where the JIT keeps it).
  At start-up `CPU::init_vu_stubs` validates the compiled machine code (straight-line, no branches / pc-relative / stack access, only
  scratch registers), copies it into the JIT code region, and the JIT emits `ldr q0 / ldr q1 / [dup or 2x trn] / bl stub / str q0`
  (small bodies are copied inline). The element shuffles (`0q 1q 0h..3h 0w..7w`) are `trn1/trn2/dup`, no table. Only x0-x8/x16/x17 and
  v-registers are clobbered, so the scalar register cache is no longer flushed around VU ops (the file is compiled with
  `-ffixed-x9..x15,x18,x20..x28`). A body that fails validation falls back to the helper call. `v0` is tracked across straight-line code
  so back-to-back dependent ops skip the reload.
- **Cheaper accumulate / clamp sequences.** Carry out of a 16-bit add is `cmhi(a, a+b)` instead of `uqadd/cmeq/mvn`; the unsigned and signed
  clamps of the 48-bit accumulator are `hi == (md >> 15) ? ... : ...` (4 instructions instead of 7-10 with the emulated pack/unpack).
  VMADN 30 -> 24 instructions, VMADH 22 -> 15, VMADM 34 -> 23, VMADL 24 -> 17, VMACF 38 -> 26.
- **Vector loads / stores inline.** LBV/SBV, LSV/SSV, LLV/SLV, LDV/SDV, LQV/SQV, LPV/LUV/SPV/SUV (the element/alignment combinations ucode
  actually uses) are a few instructions: address arithmetic in JIT registers, one alignment test, `ldr d/q [dmem, addr]`, `rev32` (the DMEM
  word swap), store. Anything else branches to an out-of-line slow path that spills only the live caller-save registers, calls the original
  helper through the new `RSP_xxx_A(rsp, rt, e, addr)` entry (same code, address already computed) and reloads. Unaligned LQV (`e == 0`) is
  done in NEON with `tbl`.
- **Other inline ops:** MTC2/MFC2, VMOV, VRCPH/VRSQH, VSAR, VNOP, MFC0 (not SP_STATUS), MTC0 DMA_CACHE/DMA_DRAM.
- **Block dispatch.** The enter thunk looks the target up in `CPU::blocks[]` itself and only calls into C for blocks that are not compiled yet.
- **Code layout.** Regions were page aligned (about 6x padding: 516 KiB of pages for 88 KiB of code in SM64, and every region entry on the
  same L1I sets). They are now packed back to back, 64-byte aligned, with the size estimate trimmed after emission. L1I refills -35 %,
  I-TLB refills -50 % in `perf stat`.
- **SP DMA** word loops became `memcpy` per row (wrapping / straddling rows keep the old loop).
- **Lightning:** `jit_raw(word)` emits a raw 32-bit instruction (a `jit_skip` node with a payload; aarch64 backend only).

The x86 / non-NEON build does not get the VU / load-store changes (`PRSP_VU_FAST` is only defined for aarch64 with NEON;
`-DPARALLEL_RSP_NO_VU_FAST` turns it off). Release flags are unchanged (`-O3`, generic armv8-a; `-mcpu` / `-march` made no difference).

## Gauntlet Legends HALT fix

When the JIT returns with `SP_STATUS_HALT` set but without `SP_STATUS_BROKE` there are two causes:

- the ucode halted itself without a BREAK (the "RSP hang when setting HALT but not BREAK" case): `DoRspCycles` returns and leaves HALT set,
  the CPU restarts the RSP;
- `RSP_MFC0` forced the HALT because the ucode polled `SP_STATUS` more than `SP_STATUS_TIMEOUT` times (it is waiting for the CPU to set a
  signal bit). This is the CXD4 `WAIT_FOR_CPU_HOST` path: HALT has to be cleared again before returning, so that the core sees a running RSP,
  schedules the SP interrupt and calls `DoRspCycles` again once the CPU has run.

The early return for the first case also caught the second, so a task that idles on `SP_STATUS` waiting for the CPU was never resumed.
Gauntlet Legends (graphics ucode, task type 1) sent 10 RDP lists and showed a black screen, both as the fallback of rsp-hle and as the only
RSP (cxd4 rendered it). `DoRspCycles` now checks the per-register `MFC0_count[]` against the time-out to tell the cases apart. SM64,
Banjo-Kazooie and Ocarina of Time are bit-identical (frames 300-1200, `Core[RandomizeInterrupt]=False`) with and without the change.

## IMEM invalidation filter and inlined VRCP / VRSQ / CFC2

Both changes come from a profile of the graphics tasks (rsp-hle's `RspFallback`, foir as RDP; "JIT proper" = JIT code + helpers + DMA +
block lookup, without the synchronous `ProcessRdpList` callback). Both are on by default; `PRSP_NO_IMEM_SKIP=1` / `PRSP_NO_DIV_INLINE=1`
switch them off for A/B runs.

- **SP DMA into IMEM only dirties the code blocks whose bytes change.** The only way the RSP modifies its own IMEM is an SP DMA read (stores
  address DMEM only). Every IMEM DMA used to dirty every block it covered, which cleared `blocks[]` and forced `get_jit_block()` (analyze,
  hash, hash-map) again although the ucode text was identical (the boot ucode copies the same text to IMEM for every task). The copy is now
  done per 256-byte block with a `memcmp`; a block is dirtied (together with its predecessor, because a compiled region starting there extends
  into it) only when the bytes differ, so a partial change still invalidates exactly the changed blocks plus the block that can span them, and
  an unchanged DMA no longer leaves the region (`MODE_CHECK_FLAGS`). The wrapping / straddling slow path uses the same helper per word.
  Writes by the CPU between tasks are unchanged: `invalidate_imem()` compares all 16 blocks with the copy the compiled code was made from at
  every `DoRspCycles`. (`rsp/cp0.cpp` `copy_to_imem`, `state.hpp` `imem_skip_identical`.)
- **VRCP, VRCPL, VRSQ, VRSQL and CFC2 are emitted inline** (VRCPH / VRSQH already were). The divide ops are a branch-free sequence of 30-35
  raw instructions (x0-x7, w16/w17, v1 only, so the scalar register cache is not flushed): sign / magnitude with `csel`, `clz`, table index
  with `ubfx` (VRSQ: `orr`/`bfi` for the shift parity), the ROM lookup through a pointer kept in `CPUState` (`recip_rom`, reachable from x19),
  `lsr`, and `csel`s for the `data == 0` (0x7fffffff) and `input == -32768` (0xffff0000) cases; `dp_flag` / `div_in` / `div_out` are read and
  written exactly like the helpers, and `VRCPL` / `VRSQL` select the 32-bit input with the flag. CFC2 is `sqxtn` + `sshr` + weight `and` +
  `addv` of the flag lanes (a movemask that works for non-canonical lane values too) straight into the cached MIPS register. There is no
  fallback path: every input is handled. (`rsp_jit.cpp` `jit_emit_vu_divide` and the CFC2 case, `rsp/neon_enc.h`, `state.hpp` `recip_rom`,
  `flag_weights`.)

### Timing results

Setup: rsp-hle + parallel-rsp as `RspFallback` + foir (3 workers), scripted input, 75 s per run, window from 24 s after the start (VI frame
1440 on), per-frame values = per-second rates x foir's ms/frame. The `rspprof` instrumentation is in foir's tree (its output is not
tracked). 3 rounds, builds and configurations alternated within each session, medians (min-max) of the 3 rounds; the scenes of the scripted
runs are repeatable (tasks per frame identical within 0.2 %). HEAD = before the two changes.

**JIT proper, ms per frame:**

| game | HEAD | IMEM filter only | inline VRCP/VRSQ/CFC2 only | both | both vs HEAD |
|---|---|---|---|---|---|
| Banjo-Kazooie | 5.15 (5.15-5.17) | 5.07 (5.06-5.08) | 5.01 (5.00-5.06) | 4.99 (4.98-5.00) | -0.16 (-3.2 %) |
| Donkey Kong 64 | 4.53 (4.51-4.54) | 4.53 (4.51-4.53) | 4.39 (4.39-4.39) | 4.37 (4.37-4.42) | -0.16 (-3.6 %) |
| Ocarina of Time | 2.12 (2.12-2.12) | 2.12 (2.11-2.13) | 2.08 (2.07-2.08) | 2.09 (2.08-2.09) | -0.03 (-1.4 %) |
| Mario Kart 64 | 3.66 (3.63-3.67) | 3.64 (3.62-3.66) | 3.57 (3.57-3.63) | 3.57 (3.57-3.58) | -0.08 (-2.2 %) |
| Paper Mario | 2.81 (2.81-2.85) | 2.80 (2.79-2.80) | 2.77 (2.75-2.77) | 2.75 (2.73-2.78) | -0.06 (-2.1 %) |
| Pokemon Snap | 0.17 (0.17-0.17) | 0.17 (0.17-0.17) | 0.17 (0.17-0.18) | 0.17 (0.17-0.17) | -0.00 (-2.9 %) |
| Star Fox 64 | 3.75 (3.74-3.77) | 3.74 (3.72-3.75) | 3.74 (3.71-3.75) | 3.69 (3.66-3.70) | -0.06 (-1.6 %) |
| Super Mario 64 | 3.44 (3.41-3.46) | 3.36 (3.34-3.40) | 3.35 (3.34-3.36) | 3.30 (3.25-3.31) | -0.14 (-4.0 %) |
| Toy Story 2 | 0.59 (0.58-0.59) | 0.56 (0.56-0.57) | 0.58 (0.58-0.58) | 0.56 (0.55-0.56) | -0.03 (-5.3 %) |

**Emulation thread, ms per frame** (foir's own CPU time of the thread that runs the core, the RSP and the RDP callback; the frame time is
16.7-16.9 ms in all runs):

| game | HEAD | IMEM filter only | inline VRCP/VRSQ/CFC2 only | both | both vs HEAD |
|---|---|---|---|---|---|
| Banjo-Kazooie | 12.37 (12.34-12.38) | 12.30 (12.29-12.34) | 12.30 (12.25-12.32) | 12.26 (12.26-12.33) | -0.11 (-0.9 %) |
| Donkey Kong 64 | 14.89 (14.87-14.93) | 14.90 (14.86-14.92) | 14.75 (14.74-14.77) | 14.79 (14.78-14.79) | -0.11 (-0.7 %) |
| Ocarina of Time | 7.01 (7.01-7.01) | 6.96 (6.96-7.07) | 6.93 (6.89-6.94) | 6.94 (6.93-6.96) | -0.07 (-1.0 %) |
| Mario Kart 64 | 9.99 (9.95-10.00) | 9.98 (9.97-9.99) | 9.99 (9.99-10.01) | 9.91 (9.90-9.93) | -0.08 (-0.8 %) |
| Paper Mario | 9.56 (9.54-9.63) | 9.64 (9.58-9.66) | 9.44 (9.41-9.52) | 9.54 (9.52-9.65) | -0.02 (-0.2 %) |
| Pokemon Snap | 4.90 (4.89-4.91) | 4.90 (4.88-4.93) | 4.93 (4.90-4.94) | 4.90 (4.90-4.91) | +0.01 (+0.2 %) |
| Star Fox 64 | 8.33 (8.32-8.34) | 8.33 (8.24-8.34) | 8.31 (8.30-8.34) | 8.31 (8.20-8.33) | -0.02 (-0.3 %) |
| Super Mario 64 | 8.99 (8.99-9.02) | 8.91 (8.88-8.94) | 8.90 (8.90-8.92) | 8.89 (8.89-8.89) | -0.10 (-1.1 %) |
| Toy Story 2 | 5.55 (5.54-5.61) | 5.56 (5.55-5.57) | 5.57 (5.56-5.57) | 5.51 (5.50-5.55) | -0.04 (-0.7 %) |

The RDP callback time (in the order of the table: 3.7 / 5.5 / 1.9 / 2.6 / 3.3 / 2.3 / 1.8 / 2.4 / 2.1 ms) is identical between the builds
within 0.1 ms. The emulation-thread differences are one to two tenths of a millisecond, close to the noise of the other work on that thread;
the JIT-proper numbers are the reliable ones.

Where the time went (per gfx task, median of 3 rounds, HEAD -> IMEM filter):

- blocks cleared by invalidations: 60 -> 44 (Banjo), 25 -> 13 (DK64), 100 -> 78 (SM64), 20 -> 7 (Paper Mario, Snap, Toy Story 2), 76 -> 76
  (OoT: its overlays really change);
- `rsp_enter()` block lookups: 221 -> 150 (Banjo), 331 -> 241 (SM64), 72 -> 30 (Toy Story 2);
- time in `get_jit_block()`: 226 -> 134 us (Banjo), 332 -> 208 us (SM64), 82 -> 38 us (Toy Story 2).

The IMEM filter alone is worth 0.08-0.09 ms/frame in Banjo-Kazooie and Super Mario 64 and nothing measurable where the microcode really
swaps overlays (Ocarina of Time) or the per-task cost is small; the inlined divides are worth 0.04-0.14 ms/frame (DK64 -0.14, Banjo -0.14,
SM64 -0.09) and are roughly additive with it. Summed over the nine games JIT proper goes from 26.22 to 25.49 ms/frame (-2.8 %).

## Exactness checks

All run against the code being replaced, with 0 differences unless a mutation is stated.

- **`tests/jit_diff.cpp`** (`-DPARALLEL_RSP_JIT_DIFF=ON`, built for aarch64, run with `qemu-aarch64 -L /usr/aarch64-linux-gnu` or on the
  device): the same random RSP programs (VU ops including reserved opcodes, all elements and aliasing, vector loads/stores with random
  alignment, COP2 / COP0 moves, scalar ALU, branches) run through a JIT with the new code generation and one with the old helper calls
  (`CPU::set_vu_fast(false)`); full state (scalar and vector registers, accumulator, flags, divide state, DMEM, PC, delay slot, CP0
  registers) is compared after every program. Modes `single` (every opcode x element), `vu` (chains of VU ops), `random`, `div`, `all`.
  The test was checked to catch deliberately wrong code generation (one-bit mutations in the shuffle, the clamp, ...).
  - Mode `div` (inlined divides and CFC2 against the helper-call JIT): (a) exhaustive sweep of all 65536 16-bit inputs in every lane for
    VRCP and VRSQ (all 8 element selections, all destination elements) plus CFC2 of VCO / VCC / VCE; (b) VRCPH / VRSQH + VRCPL / VRSQL pairs
    with corner-case halves (0, +-1, 0x7fff, 0x8000, 0x8001, 0xffff, ...), stale and cleared `dp_flag`, 16000 programs; (c) random programs
    mixing the divide ops with other VU ops, CTC2 / CFC2 (random and non-canonical flag lanes), MTC2 / MFC2 and scalar code.
    26192 runs (`jit_diff 2000 <seed> div`) under qemu and 62192 runs (`6000 4 div`) on the Cortex-A55 device: 0 mismatches. The other modes
    (`all`: 52288 runs, `vu`: 4000 runs, several seeds) pass as well, under qemu and on the device.
  - Mutation check (26192-run invocation): a table index off by one (24363 runs differ), the double-precision select inverted (17554), the
    VRSQ parity bit dropped (13274), the `input == -32768` case dropped (1620), the `data == 0` case dropped (2383), `div_out` not stored
    (26143), CFC2 lo / hi lanes swapped (25961) and CFC2 without the sign normalisation (6543) are all caught; the code was reverted after each.
- **`tests/dma_diff.cpp` / `tests/cp0_reference.cpp`** (`tests/dma_diff.sh [iterations] [seed]`): the DMA against a word-by-word reference
  that dirties a block only when a word changes; the generator makes rows, blocks and single words of the source equal to the current IMEM
  (and random registers and rows that wrap / straddle). 30000 iterations x 3 seeds: about 7600 reads into IMEM per seed, 1900 leave every
  block clean, 5400 dirty only some blocks, 0 mismatches. Mutations (the comparison ignores the last word of a block, the predecessor block
  is not dirtied) give 13824 and 34533 mismatches in 200000 iterations.
- **`tests/neon_enc_check.sh`**: 122 encoders of `rsp/neon_enc.h` against `aarch64-linux-gnu-as`, 0 mismatches. `PRSP_DEBUG_VU=1`: all 32 VU
  stubs validate.
- **Shadow verification** (`PRSP_SHADOW=1 ./build_m64p_plugin.sh aarch64`, `shadow.cpp`): every `DoRspCycles` call is replayed from the same
  start state on a second JIT with the previous code paths (`PRSP_SHADOW_MODE=prev`, default: helper VRCP/VRSQ/CFC2, IMEM always dirty;
  `old`: also the helper-call code generation of all VU and load/store ops) on a private copy of DMEM, IMEM, registers and RDRAM. Compared after
  each task: scalar and vector registers, accumulator, flags, divide state, PC, all CP0 / DPC registers, DMEM, IMEM, the RDP command lists
  handed to the RDP (start, end, hash of the bytes) and every RDRAM range written by an SP DMA (address, length, hash). The build also records
  the IMEM range and hash every compiled region was made from and checks, after every IMEM DMA (the region must be covered by a dirty bit) and
  after `invalidate_code()` (no stale region may survive), that nothing reachable through `blocks[]` is stale. Summary in `PRSP_SHADOW_OUT`.
  - On the RK3566 device (scripted input, 60 s per game): 39969 tasks compared, 0 differences; 18.9 M RDP lists and 18.7 M RDRAM DMA writes
    identical; 79.3 M inlined VRCP/VRCPL/VRSQ/VRSQL and 33.9 M inlined CFC2 executed; 760150 IMEM block writes, of which 432373 changed (the
    rest skipped); 24.7 M compiled-region checks, 0 stale. Banjo-Kazooie, Donkey Kong 64, Ocarina of Time, Mario Kart 64, Paper Mario, Pokemon
    Snap, Star Fox 64, Super Mario 64, Toy Story 2 and Gauntlet Legends against `prev`; Banjo-Kazooie, Donkey Kong 64, Ocarina of Time, Paper
    Mario, Star Fox 64 and Super Mario 64 also against `old`.
  - Mutation check: skipping the dirty bit of a changed block 9 is reported by the region checks (SM64: 3735 regions stale at the DMA, 7615
    surviving `invalidate_code()`); a variant that ignored changes in the second half of a block and also dropped the copy crashed the plugin
    after 6 tasks in DK64 (1 of 6 tasks differing); a variant that only forgot to dirty such blocks was not triggered in 40 s of DK64 / SM64
    (no compiled region overlapped the changed words), and the block-9 variant did not trigger in DK64 either; a table index off by one in the
    inline divide is reported as 277 differing RDP lists in the first 4 tasks of SM64. Reverted afterwards.
- **End to end on the device** (scripts kept with the foir RDP plugin, `tools/prsp/prsp_cmp.sh` there; screenshots at VI frames
  300/900/1500/2100/2700 and a hash of DMEM+IMEM after every `DoRspCycles` call, new vs old `.so`, `Core[RandomizeInterrupt]=False`) and
  `prsp_cmp_audio.sh` (LLE audio stream of the standalone plugin).

## Debug / tuning environment variables

| Variable | Effect |
|---|---|
| `PRSP_NO_VU_FAST=1` | helper-call code generation for VU ops, vector loads/stores and COP0/COP2 moves (A/B comparisons) |
| `PRSP_NO_DIV_INLINE=1` | call the VRCP / VRCPL / VRSQ / VRSQL / CFC2 helpers again (A/B comparisons) |
| `PRSP_NO_IMEM_SKIP=1` | dirty the IMEM code blocks on every SP DMA again, whether or not the bytes change (A/B comparisons) |
| `PRSP_INLINE_WORDS=n` | copy VU bodies of up to n instructions inline instead of calling the shared copy (default 8) |
| `PRSP_DEBUG_VU=1` | print the validation result for every VU body at start-up |
| `PRSP_DUMP_JIT=file` | write an image of the code region (use with `objdump -D -b binary -m aarch64` and `perf script`) |
| build with `PRSP_SHADOW=1 ./build_m64p_plugin.sh aarch64` | shadow verification, see above (`PRSP_SHADOW_MODE=prev\|old`, `PRSP_SHADOW_OUT=file`); slower than real time in the heavy games, not for timing |
| build with `PRSP_PROFILE=1 ./build_m64p_plugin.sh aarch64` | dynamic instruction mix, VU / LS run lengths, slow-path histograms (`PRSP_PROF_OUT=file`) |


## Scalar memory path, register cache retention, inline region exits (2026-10-13)

Six commits on `m64p-aarch64` after `f4edbfe` (run `git log f4edbfe..` for the exact list); each one was checked with `tests/jit_diff.cpp` (a deliberate error in
the new code gives mismatches, see the commit messages) and is bit-identical on the device.

| variant | commit | what |
|---|---|---|
| v0 | `f4edbfe` | before |
| v1 | `077eb2d` | LB is one `LDRSB`; unaligned LH/LHU/LW/SH/SW out of line; loads write the destination directly; `v0` kept across scalar loads/stores |
| v2 | `5e92142` | + Lightning `ldxi` value-tracking fix, `$0`-base accesses with the static address |
| v3 | `3fa01e7` | + register cache kept on the not-taken path of a conditional branch |
| v4 | `0b9dd6f` | + inline `blocks[]` lookup at every region exit (`PRSP_NO_INLINE_DISPATCH=1` = off) |
| v5 | `b54207c` | + unaligned LW as a funnel shift |

**Equivalence** (rgdsplus; `tools/equiv`-style runs of arMLEt built with each variant, `RSPT_HASH`, three screenshots at frames 400 / 1500 / 2400, pad script): the
per-call RSP state hash stream and the screenshots of v4 are identical to v0 in Super Mario 64, Ocarina of Time, Banjo-Kazooie, Donkey Kong 64, Mario Kart 64, Paper Mario, Pokemon Snap,
Star Fox 64 and Toy Story 2; v5 is identical to v0 in SM64 and OoT.

**Time** (rgdsplus, performance governor, 1.58 GHz real; arMLEt with rsp-hle audio and parallel-rsp for graphics, foir with 3 workers; RSP time per VI frame over
frames 1440-2160 from the shim's marks; median of three runs, the variants alternated in a different order in each repetition). The RSP time includes the video
plugin's synchronous callback (about 1.9-3.7 ms in these scenes, the same for every variant), so the change in the JIT proper is larger than the percentages.

| game | v0 | v1 | v2 | v3 | v4 | v5 |
|---|---|---|---|---|---|---|
| Banjo-Kazooie | 6.472 | 6.330 (-2.2 %) | 6.315 (-2.4 %) | 6.278 (-3.0 %) | 6.315 (-2.4 %) | |
| Super Mario 64 (fast-mode runs, see below) | 4.16 (4 runs) | 4.034 (1 run, -3.0 %) | 4.017 (-3.4 %) | 4.018 (-3.4 %) | 4.01 (5 runs, -3.6 %) | 4.04 (2 runs, -2.9 %) |
| Ocarina of Time (second set) | 2.786 | | | | 2.725 (-2.2 %) | 2.713 (-2.6 %) |

SM64 runs in one of two modes: most runs take 4.0-4.18 ms (emulation thread 7.06-7.29 ms), some 4.26-4.39 ms (7.9-8.1 ms). The slow mode shows up in every variant
that was run more than twice (v0 two of six runs, v1 two of three, v4 one of six, v5 one of three), so it is a property of the run (probably where the video plugin's
worker threads land), not of a build; the SM64 row uses the fast-mode runs only. Banjo-Kazooie and OoT did not show it. The medians of all runs
(v0 4.176 / 4.167, v1 4.255, v4 4.007 / 4.021, v5 4.047) are what `tim.py`-style summaries give and are dominated by that mode.

Emulation-thread CPU time per frame moves by the same absolute amount (Banjo-Kazooie -0.9 %, SM64 fast mode -1.6 %, OoT -0.8 % at v5) with a larger spread.

**Items that may be detrimental (kept; for a later bisect):**
- `0b9dd6f` (inline region exits): Banjo-Kazooie 6.278 -> 6.315 ms (+0.6 %, the ranges 6.270-6.308 and 6.303-6.344 just do not overlap), SM64 -0.3 %. The exits grow from 2 to
  about 11 instructions; the gain of per-site indirect branches did not show. `PRSP_NO_INLINE_DISPATCH=1` is the A/B switch.
- Nothing else looks detrimental: v1 and v5 in SM64 have only one or two fast-mode runs, which is too few to separate them from v2 / v4 (+0.4 %, +0.7 % over v4), and v5 gains 0.4 % in OoT
  (where unaligned LW is common) and nothing measurable elsewhere.

**Counts behind the items** (eight `--prof3` runs, rgdsplus, `test-output/session/rgdsplus/F.*`, and `PRSP_PROFILE` counters):
- scalar loads/stores are 11-15 % of the executed RSP instructions; with a `$0` base 14.8 % of them, with a base built by `lui`/`ori`/`addiu` in the same block only 0.4 % (so only `$0` is specialised);
- the cache reset at block starts costs an estimated 26 % of the executed instructions in register reloads, half at branch targets and half after a branch with its delay slot (v3 takes
  the second half for conditional branches; a fixed entry state at branch targets was not attempted);
- the enter thunks are 1.6 % of the emulation thread's samples in SM64 (4 % of the JIT proper);
- unaligned scalar helper calls per 1100 frames: SM64 LW 97.8k, SW 97.8k, LHU 15k; OoT LW 312k, LHU 39k, SW 31k; Banjo-Kazooie LHU 51k only (0.2 %, 0.4 %, 2.2 % and 0.2 % of the scalar loads/stores);
- no task in the eight scenes reaches the 32767 `SP_STATUS` polls of the initial time-out (largest COP0 count of a task: 69k in DK64 over all COP0 instructions), so the first poll-out
  burning one task per session was not observed and the initial limit is unchanged.

## Inline region exits (`0b9dd6f`) on the rg35xxsp, A/B (2026-10-15)

Question: keep the inline `blocks[]` lookup at region exits? A = default, B = `PRSP_NO_INLINE_DISPATCH=1`, same plugin binary (arMLEt `--prof` build, fork `m64p-aarch64` at the
parent's pinned commit, glibc 2.17 toolchain build), rg35xxsp (H700, 4x Cortex-A53), `cpu_perf.sh on` (1512 MHz) and restored afterwards, `tools/session/host.sh`, one
session per game: 3 alternating pairs (A B, B A, A B), scripted pad input, VI frames 1440-2160, rsp-hle audio + parallel-rsp graphics + foir (3 workers).
JIT proper = profiler level 1 (`report_time.py` definition: JIT code, helpers, SP DMA, block lookup; without the video plugin's callback). RSP = the shim's RSP time per frame
(includes the callback); emu = emulation-thread CPU time per frame. Values are the mean of 3 runs, (range of the 3 runs); delta = A vs B, negative = inline faster.

| game | JIT proper A ms | JIT proper B ms | delta | RSP delta | emu delta |
|---|---|---|---|---|---|
| Banjo-Kazooie | 4.300 (0.041) | 4.319 (0.029) | -0.4 % | -0.5 % | -0.3 % |
| Donkey Kong 64 | 3.485 (0.024) | 3.502 (0.043) | -0.5 % | +0.1 % | +0.1 % |
| Ocarina of Time | 1.398 (0.028) | 1.412 (0.018) | -1.0 % | +1.8 % | +0.9 % |
| Mario Kart 64 | 0.436 (0.009) | 0.439 (0.012) | -0.5 % | -0.4 % | -0.4 % |
| Paper Mario | 2.508 (0.020) | 2.500 (0.019) | +0.3 % | +0.7 % | +0.3 % |
| Pokemon Snap | 0.151 (0.004) | 0.157 (0.006) | -3.4 % | +1.1 % | +0.8 % |
| Star Fox 64 | 2.001 (0.019) | 2.012 (0.010) | -0.5 % | +0.2 % | +0.4 % |
| Super Mario 64 | 3.103 (0.041) | 3.120 (0.009) | -0.5 % | -0.4 % | -0.1 % |
| Toy Story 2 | 0.443 (0.044) | 0.422 (0.041) | +5.0 % | -0.1 % | 0.0 % |

The run-to-run range is 0.5-1.5 % of the JIT proper (up to 10 % in the two tiny cases, Snap and Toy Story 2, at 0.15 and 0.43 ms), 1-3 % of the RSP time. Every JIT-proper
difference is inside or at the edge of that range; the sign is A faster in 7 of 9 games, by 0.4-1.0 % (0.01-0.02 ms) in the six substantial ones. RSP and emu-thread
times, which add the callback and its worker-thread noise, scatter around zero (-0.5 % .. +1.8 %). No frame-rate effect (the games run at the VI rate).

Equivalence (plain arMLEt build, `RSPT_HASH`, screenshots at frames 400 and 1000, A vs B): the per-call RSP state hash stream (484 / 966 / 1451 / 1934 lines) and the
screenshots are identical in Banjo-Kazooie, Super Mario 64, Ocarina of Time and Paper Mario.

Verdict: on the A53 the inline lookup is neutral to marginally positive (about -0.5 % of the JIT proper, 0.01-0.02 ms per frame), the opposite sign from the Banjo-Kazooie
+0.6 % on the A55 (rgdsplus). Neither device shows an effect beyond the noise, so the change is not worth a revert on speed grounds; it adds code (exits 2 -> about 11
instructions) for no measurable gain. Raw data: `test-output/session/rg35xxsp/I.*` and `eq` (not tracked).
