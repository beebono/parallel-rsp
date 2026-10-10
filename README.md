# parallel-rsp (aarch64 / Mupen64Plus fork)

A fork of [Themaister's parallel-rsp](https://github.com/libretro/parallel-rsp), a low-level (LLE) emulation of the N64's
Reality Signal Processor that recompiles RSP microcode with a JIT ([GNU Lightning](https://www.gnu.org/software/lightning/))
instead of interpreting it. The RSP core, the Lightning integration and the original design are Themaister's; see `CREDITS.txt` for the
emulators the RSP code builds on (CEN64, Ares, CXD4).

This fork exists to run parallel-rsp on **aarch64 (ARM) retro handhelds** as a **Mupen64Plus RSP plugin**
(`mupen64plus-rsp-parallel.so`), either as the RSP plugin itself or as the `RspFallback` of
[`mupen64plus-rsp-hle`](https://github.com/mupen64plus/mupen64plus-rsp-hle), which runs audio lists in HLE and hands graphics tasks to
the fallback. It is normally paired with an LLE RDP plugin such as `mupen64plus-rdp-foir` or angrylion-rdp-plus. The branch
`m64p-aarch64` carries the changes below; `master` tracks upstream.

## What this fork changes

Everything is bit-identical to the code it replaces (checked by the tools in `tests/`, see below). Measured results are in
[docs/results.md](docs/results.md).

- **Mupen64Plus plugin target.** CMake option `PARALLEL_RSP_M64P_PLUGIN` builds `mupen64plus-rsp-parallel.so` from `parallel.cpp` and the new
  `m64p_plugin_entry.cpp` (the plugin entry points), with a private static copy of Lightning, hidden symbols and `PARALLEL_INTEGRATION`
  (real SP DMA and RDP list handling). `build_m64p_plugin.sh` and `toolchain-aarch64-linux-gnu.cmake` wrap it for native and cross builds.
- **aarch64 VU and load/store JIT fast paths** (`rsp_jit.cpp`, `rsp/vu_inline.cpp`, `rsp/vu_neon.h`, `rsp/neon_enc.h`, `rsp/ls.cpp`). The
  original JIT called a C helper for every vector instruction and vector load/store; in Super Mario 64 that was about 46 % of the emulation
  thread. Now the VU op bodies are position-independent NEON functions with a register calling convention
  (`vs` in `v0`, shuffled `vt` in `v1`, result in `v0`, CPUState in `x19`) that the JIT validates at start-up and copies into its
  code (inline for small bodies, a shared stub otherwise). Because they only touch scratch registers, the scalar register cache is no
  longer flushed around VU ops. Multiply-accumulate and carry/clamp sequences were shortened (e.g. VMADN 30 to 24 instructions).
  Vector loads/stores (LBV ... SUV) are a few inline instructions with an out-of-line slow path through new `RSP_xxx_A` helper entries;
  MTC2/MFC2, VMOV, VSAR, VNOP, MFC0 and the DMA MTC0 writes are inline too. Block dispatch happens in the enter thunk. Lightning gets a
  `jit_raw()` macro for raw instruction words (aarch64 backend only). Defined only for aarch64 with NEON; `-DPARALLEL_RSP_NO_VU_FAST` turns it off.
- **Code layout and SP DMA** (`jit_allocator.cpp`, `rsp/cp0.cpp`; all Linux-like targets, not only aarch64). JIT regions are packed back to
  back, 64-byte aligned, instead of one page each (L1I refills -35 %, I-TLB refills -50 % in `perf stat`). SP DMA copies whole rows with `memcpy`.
- **Gauntlet HALT fix** (`parallel.cpp`, `parallelRSPDoRspCycles`). A ucode that polls `SP_STATUS` too long makes `RSP_MFC0` force
  HALT; upstream's early return for "HALT without BREAK" treated that like a ucode halting itself, so a task waiting for the CPU was
  never resumed (Gauntlet Legends showed a black screen with 10 RDP lists). A polling time-out now clears HALT again, like cxd4's
  `WAIT_FOR_CPU_HOST`; a real self-halt still returns with HALT set. SM64, Banjo-Kazooie and Ocarina of Time are unchanged.
- **Skipping IMEM invalidation on unchanged DMA** (`rsp/cp0.cpp` `copy_to_imem`, `state.hpp`). The RSP only modifies its IMEM through SP DMA,
  and the boot ucode copies identical text for every task, yet every DMA dirtied every covered code block and forced a re-analysis.
  The DMA now compares per 256-byte block and dirties a block (and its predecessor, which a compiled region can extend into) only when the
  bytes change. 0.08-0.09 ms/frame in Banjo-Kazooie and SM64 on the RK3566; nothing measurable where overlays really change. `PRSP_NO_IMEM_SKIP=1` disables it.
- **Inlined VRCP / VRCPL / VRSQ / VRSQL and CFC2** (`rsp_jit.cpp` `jit_emit_vu_divide`). The reciprocal ops become a branch-free
  30-35 instruction sequence (the ROM table is reached from `CPUState`); CFC2 is a NEON movemask into the cached MIPS register. No
  fallback path is needed. 0.04-0.14 ms/frame; together with the IMEM filter, JIT time over nine games is -2.8 %.
  `PRSP_NO_DIV_INLINE=1` disables it.
- **Debug and verification aids.**
  - `PRSP_SHADOW=1 ./build_m64p_plugin.sh` (`shadow.cpp`): a build that replays every `DoRspCycles` call on a second JIT with the previous
    code paths (`PRSP_SHADOW_MODE=prev|old`) and compares registers, memories, RDP command lists and SP DMA writes to RDRAM, and checks that no
    stale compiled region survives an IMEM change. Summary goes to `PRSP_SHADOW_OUT`. Slower than real time, for verification only.
  - `PRSP_PROFILE=1 ./build_m64p_plugin.sh`: a build with a dynamic instruction-mix profile, VU/LS run lengths and slow-path histograms
    (`PRSP_PROF_OUT=file`).
  - `PRSP_DEBUG_VU=1`: print the start-up validation result of every VU body.
  - `PRSP_DUMP_JIT=file`: write an image of the code region (for `objdump -D -b binary -m aarch64`).
  - A/B switches: `PRSP_NO_VU_FAST=1` (helper-call code generation for VU ops, vector loads/stores and COP0/COP2 moves), `PRSP_NO_DIV_INLINE=1`,
    `PRSP_NO_IMEM_SKIP=1`, and `PRSP_INLINE_WORDS=n` (VU bodies up to n instructions are copied inline, default 8).
- **Test tools** (`tests/`), all comparing the new code against the code it replaced:
  - `jit_diff.cpp` (CMake `PARALLEL_RSP_JIT_DIFF=ON`, target `rsp-jit-diff`): random RSP programs run through the fast JIT and the helper-call JIT,
    full state compared; modes `single`, `vu`, `random`, `div` (exhaustive VRCP/VRSQ/CFC2 sweeps), `all`.
  - `dma_diff.cpp` / `cp0_reference.cpp` / `dma_diff.sh`: SP DMA and the IMEM dirty bits against a word-by-word reference.
  - `neon_enc_check.sh`: every raw encoder in `rsp/neon_enc.h` against `aarch64-linux-gnu-as`.

## Using it with Mupen64Plus

Pass the plugin as the RSP plugin:

```sh
mupen64plus --rsp /path/to/mupen64plus-rsp-parallel.so --gfx mupen64plus-rdp-foir.so ROM.z64
```

This runs graphics and audio tasks in LLE (audio ucode output was bit-identical to cxd4 in the 7 games checked). To keep audio in HLE, which is faster, use
`mupen64plus-rsp-hle` and set the parallel plugin as its fallback:

```ini
[Rsp-HLE]
DisplayListToGraphicsPlugin = False
RspFallback = "/path/to/mupen64plus-rsp-parallel.so"
```

`DisplayListToGraphicsPlugin = False` makes graphics tasks go to the fallback, which feeds the RDP plugin. Pair either setup with an LLE RDP
plugin (foir, angrylion-rdp-plus); the `Core[RandomizeInterrupt]` setting is only relevant for bit-exact comparisons (set it to `False`).
The foir plugin's README (section "RSP pairing") has the measured pairings and a full command line.

## Building

Needs CMake, a C++14 compiler and the Mupen64Plus API headers (`m64p_types.h`, `m64p_plugin.h`); the default location is
`../mupen64plus-core/src/api`, override with `M64P_API_INCLUDE_DIR=<dir>`.

```sh
./build_m64p_plugin.sh aarch64   # cross build with aarch64-linux-gnu-g++ -> build-m64p-aarch64/mupen64plus-rsp-parallel.so
./build_m64p_plugin.sh           # native build                           -> build-m64p/mupen64plus-rsp-parallel.so
PRSP_SHADOW=1  ./build_m64p_plugin.sh aarch64   # shadow-verification variant (build-m64p-aarch64-shadow/)
PRSP_PROFILE=1 ./build_m64p_plugin.sh aarch64   # profiling variant (build-m64p-aarch64-prof/)
```

Release builds use `-O3` with generic armv8-a; `-mcpu` / `-march` made no measurable difference.

Tests (run the aarch64 ones under `qemu-aarch64-static -L /usr/aarch64-linux-gnu` on an x86_64 host, or directly on a device):

```sh
cmake -S . -B build-test -DCMAKE_TOOLCHAIN_FILE=$PWD/toolchain-aarch64-linux-gnu.cmake -DCMAKE_BUILD_TYPE=Release \
      -DPARALLEL_RSP_JIT_DIFF=ON -DPARALLEL_RSP_DEBUG_JIT=OFF -DPARALLEL_RSP_TESTS=OFF
cmake --build build-test --target rsp-jit-diff
qemu-aarch64-static -L /usr/aarch64-linux-gnu build-test/rsp-jit-diff 2000 1 all   # [programs] [seed] [single|vu|random|div|all]; expect 0 mismatches
tests/dma_diff.sh 30000 1       # native; [iterations] [seed]
tests/neon_enc_check.sh         # needs aarch64-linux-gnu-as / objdump
```

## License

Upstream parallel-rsp is dual licensed under the **MIT license** (`LICENSE.MIT`, Copyright (c) 2020 Hans-Kristian Arntzen) or the
**GNU LGPL v3** (`LICENSE.LESSER`); `LICENSE` states this and is kept unchanged. The additions and changes in this fork (branch `m64p-aarch64`) are licensed under the same terms as upstream: MIT or LGPLv3, at your option.
Third-party code bundled in the tree:

- `lightning/` is GNU Lightning (Free Software Foundation), dual-licensed GPLv3 / LGPLv3 (`lightning/COPYING`, `lightning/COPYING.LESSER`;
  the source files carry the LGPL-3.0-or-later header). The Mupen64Plus plugin links it statically, so a distributed plugin binary falls under the
  LGPLv3 option. parallel-rsp can alternatively be built against a dynamically linked Lightning or without it.
- `rsp/vfunctions.cpp`, `rsp/reciprocal.cpp` and most of `arch/simd/rsp/*.h` descend from CEN64 (Tyler J. Stachecki) and Ares/CXD4 code and keep their
  headers, which refer to the repository's `LICENSE`; `arch/simd/rsp/SSE2NEON.h` is MIT-licensed (see its header).

Keep the license files and the notices in `CREDITS.txt` and the source headers when redistributing.
