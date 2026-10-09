#!/bin/bash
# Builds mupen64plus-rsp-parallel.so, the Mupen64Plus RSP plugin (standalone RSP, or RspFallback of rsp-hle).
#   ./build_m64p_plugin.sh            native build        -> build-m64p/mupen64plus-rsp-parallel.so
#   ./build_m64p_plugin.sh aarch64    cross build for aarch64 Linux (aarch64-linux-gnu-g++)
#                                     -> build-m64p-aarch64/mupen64plus-rsp-parallel.so
# M64P_API_INCLUDE_DIR may point at the directory holding m64p_types.h (default ../mupen64plus-core/src/api).
set -e
cd "$(dirname "$0")"
DIR=build-m64p
ARGS=()
if [ "$1" = "aarch64" ]; then
	DIR=build-m64p-aarch64
	ARGS+=(-DCMAKE_TOOLCHAIN_FILE="$PWD/toolchain-aarch64-linux-gnu.cmake")
fi
# PRSP_PROFILE=1 builds the dynamic instruction-mix profiling variant into a separate build dir.
if [ -n "$PRSP_PROFILE" ]; then DIR=$DIR-prof; ARGS+=(-DCMAKE_CXX_FLAGS=-DPRSP_PROFILE); fi
[ -n "$M64P_API_INCLUDE_DIR" ] && ARGS+=(-DM64P_API_INCLUDE_DIR="$M64P_API_INCLUDE_DIR")
mkdir -p "$DIR"
cd "$DIR"
cmake .. -DCMAKE_BUILD_TYPE=Release -DPARALLEL_RSP_M64P_PLUGIN=ON -DPARALLEL_RSP_DEBUG_JIT=OFF \
	-DPARALLEL_RSP_TESTS=OFF "${ARGS[@]}"
cmake --build . --target mupen64plus-rsp-parallel --parallel
echo "built: $PWD/mupen64plus-rsp-parallel.so"
