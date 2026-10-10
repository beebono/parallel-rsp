#!/bin/bash
# Builds and runs the SP DMA differential test natively.
set -e
cd "$(dirname "$0")/.."
API=${M64P_API_INCLUDE_DIR:-../mupen64plus-core/src/api}
T=$(mktemp -d)
g++ -O2 -DPARALLEL_INTEGRATION -DM64P_PLUGIN_API -I. -Iarch/simd/rsp -I"$API" -c rsp/cp0.cpp -o $T/new.o
g++ -O2 -DPARALLEL_INTEGRATION -DM64P_PLUGIN_API -I. -Iarch/simd/rsp -I"$API" -DRSP_MTC0=RSP_MTC0_ref -DRSP_MFC0=RSP_MFC0_ref -c tests/cp0_reference.cpp -o $T/ref.o
g++ -O2 -DPARALLEL_INTEGRATION -DM64P_PLUGIN_API -I. -Iarch/simd/rsp -I"$API" tests/dma_diff.cpp $T/new.o $T/ref.o rsp/reciprocal.cpp -o $T/dma_diff
$T/dma_diff "$@"
rm -rf $T
