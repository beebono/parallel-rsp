cmake_minimum_required(VERSION 3.5)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_C_STANDARD 99)
project(parallel-rsp LANGUAGES CXX C)

if (CMAKE_COMPILER_IS_GNUCXX OR (${CMAKE_CXX_COMPILER_ID} MATCHES "Clang"))
    set(PARALLEL_RSP_CXX_FLAGS -Wall -Wextra -Wno-missing-field-initializers -Wno-empty-body -ffast-math -Wno-unused-parameter)
elseif (MSVC)
    set(PARALLEL_RSP_CXX_FLAGS /D_CRT_SECURE_NO_WARNINGS /wd4267 /wd4244 /wd4309 /wd4005 /MP /DNOMINMAX)
endif()

set(PARALLEL_RSP_SOURCES
		rsp/vfunctions.cpp
		rsp_jit.cpp rsp_jit.hpp
		jit_allocator.cpp jit_allocator.hpp
		rsp_disasm.cpp rsp_disasm.hpp
		rsp/ls.cpp rsp/pipeline.h
		rsp/reciprocal.cpp rsp/reciprocal.h
		rsp_1.1.h
		rsp/cp0.cpp rsp/cp2.cpp
		rsp/vu_inline.cpp rsp/vu_inline.h
		arch/simd/rsp/rsp_core.cpp
		arch/simd/rsp/clamp.h
		arch/simd/rsp/rsp_common.h
		arch/simd/rsp/SSE2NEON.h
		arch/simd/rsp/rsp_impl.h
		arch/simd/rsp/vcr.h
		arch/simd/rsp/vabs.h
		arch/simd/rsp/vadd.h
		arch/simd/rsp/vaddc.h
		arch/simd/rsp/vand.h
		arch/simd/rsp/vch.h
		arch/simd/rsp/vcl.h
		arch/simd/rsp/vcr.h
		arch/simd/rsp/vcmp.h
		arch/simd/rsp/vdivh.h
		arch/simd/rsp/vmac.h
		arch/simd/rsp/vmrg.h
		arch/simd/rsp/vmudh.h
		arch/simd/rsp/vmul.h
		arch/simd/rsp/vmull.h
		arch/simd/rsp/vmulh.h
		arch/simd/rsp/vmuln.h
		arch/simd/rsp/vor.h
		arch/simd/rsp/vrcpsq.h
		arch/simd/rsp/vrsq.h
		arch/simd/rsp/vsub.h
		arch/simd/rsp/vsubc.h
		arch/simd/rsp/vxor.h
		arch/simd/rsp/vmulm.h)
# rsp/vu_inline.cpp holds the register-convention VU op bodies the aarch64 JIT copies into its code (x19 = CPUState). The copied
# code must not touch any integer register the JIT keeps live, so every one of them is fixed for that file (x0-x8, x16, x17 stay free).
if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
	set_source_files_properties(rsp/vu_inline.cpp PROPERTIES COMPILE_FLAGS
		"-ffixed-x9 -ffixed-x10 -ffixed-x11 -ffixed-x12 -ffixed-x13 -ffixed-x14 -ffixed-x15 -ffixed-x18 -ffixed-x20 -ffixed-x21 -ffixed-x22 -ffixed-x23 -ffixed-x24 -ffixed-x25 -ffixed-x26 -ffixed-x27 -ffixed-x28")
endif()
add_library(parallel-rsp STATIC ${PARALLEL_RSP_SOURCES})

option(PARALLEL_RSP_DEBUG_JIT "Enable the debug JIT." ON)

if (PARALLEL_RSP_DEBUG_JIT)
	if (ANDROID)
		message("Android does not support debug JIT. Disabling it.")
	elseif (NOT WIN32)
		target_sources(parallel-rsp PRIVATE debug_rsp.cpp debug_rsp.hpp debug_jit.cpp debug_jit.hpp)
		target_compile_definitions(parallel-rsp PUBLIC PARALLEL_RSP_DEBUG_JIT)
	endif()
endif()

target_include_directories(parallel-rsp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(parallel-rsp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/arch/simd/rsp)
target_compile_options(parallel-rsp PRIVATE ${PARALLEL_RSP_CXX_FLAGS})

option(PARALLEL_RSP_BAKED_LIGHTNING "Use built-in Lightning." ON)

target_compile_definitions(parallel-rsp PUBLIC PARALLEL_RSP_LIGHTNING)
if (PARALLEL_RSP_BAKED_LIGHTNING)
	add_library(lightning STATIC
			lightning/lib/jit_disasm.c
			lightning/lib/jit_memory.c
			lightning/lib/jit_names.c
			lightning/lib/jit_note.c
			lightning/lib/jit_print.c
			lightning/lib/jit_size.c
			lightning/lib/lightning.c)
	target_include_directories(lightning PUBLIC
			${CMAKE_CURRENT_SOURCE_DIR}/lightning/include)
	if (WIN32)
		target_sources(lightning PRIVATE win32/mman/sys/mman.c)
		target_include_directories(lightning PRIVATE win32/mman)
	endif()
	target_compile_definitions(lightning PUBLIC HAVE_MMAP=1)
endif()
target_link_libraries(parallel-rsp PUBLIC lightning)
target_compile_definitions(parallel-rsp PUBLIC DEBUG_JIT)

if (NOT WIN32)
	target_link_libraries(parallel-rsp PRIVATE dl)
endif()

add_executable(rsp-runner main.cpp)
target_link_libraries(rsp-runner PRIVATE parallel-rsp)
target_compile_options(rsp-runner PRIVATE ${PARALLEL_RSP_CXX_FLAGS})

if (PARALLEL_RSP_DEBUG_JIT AND NOT WIN32)
	set_target_properties(rsp-runner PROPERTIES LINK_FLAGS "-rdynamic")
endif()

option(PARALLEL_RSP_TESTS "Enable unit tests for Lightning CPU." ON)

# Differential test: optimized JIT code generation vs the reference helper-call code generation (tests/jit_diff.cpp).
option(PARALLEL_RSP_JIT_DIFF "Build the JIT differential test." OFF)
if (PARALLEL_RSP_JIT_DIFF)
	add_executable(rsp-jit-diff tests/jit_diff.cpp)
	target_link_libraries(rsp-jit-diff PRIVATE parallel-rsp)
	target_compile_definitions(parallel-rsp PUBLIC PRSP_LS_COUNT)
	target_compile_options(rsp-jit-diff PRIVATE ${PARALLEL_RSP_CXX_FLAGS})
endif()

add_executable(rsp-vu-fuzzer rsp_vu_fuzzer.cpp)
target_link_libraries(rsp-vu-fuzzer PRIVATE parallel-rsp)
target_compile_options(rsp-vu-fuzzer PRIVATE ${PARALLEL_RSP_CXX_FLAGS})

if (PARALLEL_RSP_TESTS)
	enable_testing()

	if (ANDROID)
		add_test(NAME rsp-vu-fuzz
				COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/android_fuzz_runner.sh
				${CMAKE_CURRENT_BINARY_DIR}/rsp-vu-fuzzer)
	else()
		add_test(NAME rsp-vu-fuzz COMMAND $<TARGET_FILE:rsp-vu-fuzzer>)
	endif()
	add_custom_target(rsp-tests ALL)

	add_custom_target(rsp-test-crt
			COMMAND ${CMAKE_MAKE_PROGRAM} crt-obj -j1
			WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/debug-toolchain)

	function(rsp_add_test NAME)
		add_custom_target(rsp-test-binary-${NAME}
				COMMAND ${CMAKE_MAKE_PROGRAM} RSP_TEST=${NAME} -j1
				WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/debug-toolchain
				DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/debug-toolchain/${NAME}.s)
		add_dependencies(rsp-test-binary-${NAME} rsp-test-crt)
		add_dependencies(rsp-tests rsp-test-binary-${NAME})

		if (ANDROID)
			add_test(NAME rsp-test-${NAME}
					COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/android_test_runner.sh
					${CMAKE_CURRENT_BINARY_DIR}/rsp-runner
					${CMAKE_CURRENT_SOURCE_DIR}/debug-toolchain ${NAME})
		else()
			add_test(NAME rsp-test-${NAME}
					COMMAND $<TARGET_FILE:rsp-runner>
					${CMAKE_CURRENT_SOURCE_DIR}/debug-toolchain/${NAME}.global.bin
					${CMAKE_CURRENT_SOURCE_DIR}/debug-toolchain/${NAME}.bin)
		endif()
	endfunction()

	rsp_add_test(sll)
	rsp_add_test(srl)
	rsp_add_test(sra)
	rsp_add_test(sllv)
	rsp_add_test(srlv)
	rsp_add_test(srav)
	rsp_add_test(add)
	rsp_add_test(sub)
	rsp_add_test(or)
	rsp_add_test(and)
	rsp_add_test(xor)
	rsp_add_test(nor)
	rsp_add_test(slt)
	rsp_add_test(sltu)
	rsp_add_test(addi)
	rsp_add_test(slti)
	rsp_add_test(sltiu)
	rsp_add_test(andi)
	rsp_add_test(ori)
	rsp_add_test(xori)
	rsp_add_test(lui)
	rsp_add_test(lb)
	rsp_add_test(lbu)
	rsp_add_test(lh)
	rsp_add_test(lhu)
	rsp_add_test(lh-unaligned)
	rsp_add_test(lhu-unaligned)
	rsp_add_test(lw)
	rsp_add_test(lw-unaligned)
	rsp_add_test(sb)
	rsp_add_test(sh)
	rsp_add_test(sw)
	rsp_add_test(sh-unaligned)
	rsp_add_test(sw-unaligned)
	rsp_add_test(beq-impossible-delay-slot-second-taken)
	rsp_add_test(beq-impossible-delay-slot-first-taken)
	#rsp_add_test(beq-impossible-delay-slot-both-taken) Hangs the reference implementation :D
	rsp_add_test(bne)
	rsp_add_test(blez)
	rsp_add_test(bltz)
	rsp_add_test(bltzal)
	rsp_add_test(bgtz)
	rsp_add_test(bgezal)
	rsp_add_test(j)
	rsp_add_test(jal)
	rsp_add_test(jr)
	rsp_add_test(jalr)
	rsp_add_test(lw-unaligned-in-branch-delay)
	rsp_add_test(delay-slot-before-break)
	rsp_add_test(delay-slot-before-new-block)
	rsp_add_test(delay-slot-before-new-block-illegal)
	rsp_add_test(delay-slot-before-new-block-not-taken)
	rsp_add_test(unconditional-delay-slot-before-break)
	rsp_add_test(cop0)
	rsp_add_test(cop2-basic)
	rsp_add_test(cop2-ls)
	rsp_add_test(cop2-vector)
	rsp_add_test(bug-shl-into-branch)
endif()

# Mupen64Plus RSP plugin (mupen64plus-rsp-parallel.so), usable standalone as the RSP plugin or as the
# RspFallback of mupen64plus-rsp-hle. This needs PARALLEL_INTEGRATION (real SP DMA / RDP list handling in
# cp0.cpp instead of the rsp-runner test harness behaviour) and the release JIT (no DEBUG_JIT), so it
# gets its own copy of the library rather than sharing the target used by rsp-runner / the fuzzer.
#
#   cmake .. -DPARALLEL_RSP_M64P_PLUGIN=ON -DPARALLEL_RSP_DEBUG_JIT=OFF -DPARALLEL_RSP_TESTS=OFF \
#            -DM64P_API_INCLUDE_DIR=<mupen64plus-core>/src/api -DCMAKE_BUILD_TYPE=Release
option(PARALLEL_RSP_M64P_PLUGIN "Build the Mupen64Plus RSP plugin." OFF)
set(M64P_API_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../mupen64plus-core/src/api" CACHE PATH
    "Directory with m64p_types.h / m64p_plugin.h")

if (PARALLEL_RSP_M64P_PLUGIN AND NOT WIN32)
	add_library(lightning-m64p STATIC
			lightning/lib/jit_disasm.c
			lightning/lib/jit_memory.c
			lightning/lib/jit_names.c
			lightning/lib/jit_note.c
			lightning/lib/jit_print.c
			lightning/lib/jit_size.c
			lightning/lib/lightning.c)
	target_include_directories(lightning-m64p PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/lightning/include)
	target_compile_definitions(lightning-m64p PUBLIC HAVE_MMAP=1)
	set_target_properties(lightning-m64p PROPERTIES POSITION_INDEPENDENT_CODE ON C_VISIBILITY_PRESET hidden)

	add_library(mupen64plus-rsp-parallel MODULE ${PARALLEL_RSP_SOURCES} parallel.cpp m64p_plugin_entry.cpp)
	set_target_properties(mupen64plus-rsp-parallel PROPERTIES
			PREFIX ""
			POSITION_INDEPENDENT_CODE ON
			CXX_VISIBILITY_PRESET hidden
			VISIBILITY_INLINES_HIDDEN ON)
	target_include_directories(mupen64plus-rsp-parallel PRIVATE
			${CMAKE_CURRENT_SOURCE_DIR}
			${CMAKE_CURRENT_SOURCE_DIR}/arch/simd/rsp
			${M64P_API_INCLUDE_DIR})
	target_compile_definitions(mupen64plus-rsp-parallel PRIVATE
			PARALLEL_INTEGRATION M64P_PLUGIN_API PARALLEL_RSP_LIGHTNING)
	target_compile_options(mupen64plus-rsp-parallel PRIVATE ${PARALLEL_RSP_CXX_FLAGS})
	target_link_libraries(mupen64plus-rsp-parallel PRIVATE lightning-m64p dl)
	# Keep every non-API symbol (Lightning's _emit_*, etc.) private to the module.
	target_link_options(mupen64plus-rsp-parallel PRIVATE -Wl,-Bsymbolic -Wl,--exclude-libs,ALL)
endif()
