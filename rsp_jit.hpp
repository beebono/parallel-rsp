#ifndef RSP_JIT_HPP__
#define RSP_JIT_HPP__

#include <memory>
#include <stdint.h>
#include <string.h>
#include <string>
#include <unordered_map>
#include <vector>

#include "rsp_op.hpp"
#include "state.hpp"
#include "jit_allocator.hpp"
#ifdef PRSP_NATIVE
#include "prsp_native.h"
#endif

#if defined(__aarch64__) && defined(__ARM_NEON) && !defined(PARALLEL_RSP_NO_VU_FAST)
// aarch64: VU ops are emitted as register-convention bodies (rsp/vu_inline.cpp) instead of RSP_Vxxx helper calls.
#define PRSP_VU_FAST 1
#include "rsp/vu_inline.h"
#endif

extern "C"
{
#include <lightning.h>
}

namespace RSP
{
namespace JIT
{
using Func = jit_pointer_t;

class RegisterCache
{
public:
	enum { COND_BRANCH_TAKEN = 32, SCRATCH_REGISTER0 = 33, SCRATCH_REGISTER1 = 34 };

	// Ensures a mips register is present in a JIT register, lock it,
	// and ensure register extend state is correct for opcodes which care.
	unsigned load_mips_register_noext(jit_state_t *_jit, unsigned mips_reg);
	unsigned load_mips_register_sext(jit_state_t *_jit, unsigned mips_reg);
	unsigned load_mips_register_zext(jit_state_t *_jit, unsigned mips_reg);

	// Prepare to modify the register and lock it.
	unsigned modify_mips_register(jit_state_t *_jit, unsigned mips_reg);

	// modify_mips_register + jit_movi helper.
	unsigned immediate_mips_register(jit_state_t *_jit, unsigned mips_reg, jit_word_t value);

	// Once we no longer require the register to be in register file, free it.
	void unlock_mips_register(unsigned mips_reg);

	// Before leaving a JIT block, or before branching, we must flush out all live registers.
	void flush_register_window(jit_state_t *_jit);

	// Similar, but we only need to invalidate caller save registers (JIT_R).
	void flush_caller_save_registers(jit_state_t *_jit);

	// Flush a single MIPS register.
	void flush_mips_register(jit_state_t *_jit, unsigned mips_reg);

	// Out-of-line slow paths that call C helpers: they must keep the caller-save registers that are live at the branch point.
	// snapshot_caller_save() records them, emit_spill() / emit_reload() write the dirty ones to CPUState::sr before the call
	// and restore all of them afterwards, without touching the cache bookkeeping of the main code path.
	struct Saved
	{
		unsigned jit_reg;
		unsigned mips_reg;
		bool modified;
		bool zext;
	};
	enum { MaxSaved = 8 };
	unsigned snapshot_caller_save(Saved *out) const;
	static void emit_spill(jit_state_t *_jit, const Saved *saved, unsigned count);
	static void emit_reload(jit_state_t *_jit, const Saved *saved, unsigned count);

	void reset();

private:
	enum { NumFreeRegisters = JIT_R_NUM + (JIT_V_NUM - 3) };

	enum SignState { SExt, ZExt, Unknown };

	struct CacheEntry
	{
		unsigned mips_register;
		unsigned timestamp;
		unsigned num_locks;
		SignState sign;
		bool is_live;
		bool modified;
	};
	CacheEntry entries[NumFreeRegisters] = {};
	unsigned timestamp = 0;

	CacheEntry *find_live_mips_register(unsigned mips_reg);
	CacheEntry *find_free_register();
	CacheEntry *find_oldest_unlocked_register();
	CacheEntry &find_register(unsigned mips_reg);
	void writeback_register(jit_state_t *_jit, CacheEntry &entry);

	static unsigned jit_register_to_index(unsigned jit_reg);
	unsigned entry_to_jit_register(const CacheEntry &entry);
};

class alignas(64) CPU
{
public:
	CPU();

	~CPU();

	CPU(CPU &&) = delete;

	void operator=(CPU &&) = delete;

	void set_dmem(uint32_t *dmem)
	{
		state.dmem = dmem;
	}

	void set_imem(uint32_t *imem)
	{
		state.imem = imem;
	}

	void set_rdram(uint32_t *rdram)
	{
		state.rdram = rdram;
	}

	void invalidate_imem();

#ifdef PRSP_VU_FAST
	// Test hooks: choose the helper-call VU path (false) or the register-convention path before generating code.
	void set_vu_fast(bool enable)
	{
		vu_fast_enabled = enable;
	}

	// Test hook: the inlined VRCP / VRCPL / VRSQ / VRSQL / CFC2 (default on; only used while the register-convention path is on).
	void set_div_inline(bool enable)
	{
		div_inline_enabled = enable;
	}
#endif

	// Test hook: the IMEM DMA invalidation filter (see CPUState).
	void set_imem_skip(bool enable)
	{
		state.imem_skip_identical = enable ? 1 : 0;
	}

	CPUState &get_state()
	{
		return state;
	}

	ReturnMode run();

	void print_registers();

	Func get_jit_block(uint32_t pc);

#ifdef PRSP_NATIVE
	// Native-routine hook (prsp_native.h). A stub is the block that sits in blocks[] at the entry pc of a matched routine.
	struct NativeStub
	{
		const void *code = nullptr; // the stub's code, entered like any block
		unsigned pc_word = 0;
		unsigned cookie = 0;
		unsigned dep_lo = 0, dep_hi = 0; // byte range, hi <= lo: none
		Func jit = nullptr; // the JIT's own region for the entry pc (compiled on demand, dropped with the stub)
		bool installed = false;
	};
	void native_install(const PrspNativeHost *host);
	void native_set_pcs(const uint16_t *entries, unsigned n_entries, const uint16_t *splits, unsigned n_splits);
	Func native_enter(NativeStub *stub); // from the stub's thunk: the block to continue in, or null to leave the run loop with native_mode
	int native_oracle(unsigned entry_pc, unsigned stop_pc);
	void native_resync()
	{
		invalidate_imem();
		invalidate_code();
	}
#endif

private:
	CPUState state;
	Func blocks[IMEM_WORDS] = {};

	void invalidate_code();

#ifdef PRSP_NATIVE
	const PrspNativeHost *native_host = nullptr;
	bool native_busy = false; // inside the oracle: stubs decline
	bool native_entry[IMEM_WORDS] = {};
	bool native_split[IMEM_WORDS] = {};
	bool native_have_splits = false;
	std::unordered_map<unsigned, NativeStub> native_stubs;
	std::unordered_map<unsigned, const void *> native_stops;
	std::vector<unsigned> native_installed; // word pcs whose stub is in blocks[]
	std::vector<unsigned> native_entry_words;
	Func native_lookup(unsigned word_pc); // get_jit_block, on a miss: a stub if a routine applies
	unsigned native_cap_end(unsigned word_pc, unsigned end) const; // no region extends across a split pc
	NativeStub *native_stub(unsigned word_pc);
	Func native_region(unsigned word_pc); // the compiled region for word_pc, never a stub
	const void *native_stop_stub(unsigned word_pc);
	void native_invalidate(uint32_t dirty);
	void native_drop_all();
#endif

#ifdef PRSP_SHADOW
public:
	// Shadow-verification build: every compiled region remembers the IMEM range and hash it was compiled from. A region that is
	// still reachable through blocks[] must still match IMEM (checked after invalidate_code() in run()), and a region that no
	// longer matches IMEM after an SP DMA must be covered by a dirty bit (shadow_check_blocks(first, last)).
	struct BlockRec
	{
		unsigned pc_word, count;
		uint64_t hash;
	};
	std::unordered_map<const void *, BlockRec> sh_block_rec;
	uint64_t sh_checks = 0, sh_stale_after_invalidate = 0, sh_stale_not_dirty = 0;
	void shadow_check_blocks(bool after_invalidate, unsigned first_word, unsigned last_word);

private:
#endif

	uint64_t hash_imem(unsigned pc, unsigned count) const;

	alignas(64) uint32_t cached_imem[IMEM_WORDS] = {};

	std::unordered_map<uint64_t, Func> cached_blocks[IMEM_WORDS];

	Func jit_region(uint64_t hash, unsigned pc_word, unsigned instruction_count);

	int enter(uint32_t pc);

	void init_jit_thunks();

	struct
	{
		int (*enter_frame)(void *state) = nullptr;

		Func enter_thunk = nullptr;
		Func return_thunk = nullptr;
#ifdef PRSP_NATIVE
		Func native_thunk = nullptr; // entered from a stub with JIT_R2 = NativeStub *
#endif
	} thunks;

	unsigned analyze_static_end(unsigned pc, unsigned end);

	struct InstructionInfo
	{
		uint32_t branch_target;
		bool indirect;
		bool branch;
		bool conditional;
		bool handles_delay_slot;
	};
	void jit_instruction(jit_state_t *_jit, uint32_t pc, uint32_t instr, InstructionInfo &info, const InstructionInfo &last_info,
	                     bool first_instruction, bool next_instruction_is_target);
	void jit_exit(jit_state_t *_jit, uint32_t pc, const InstructionInfo &last_info, ReturnMode mode, bool first_instruction);
	void jit_exit_dynamic(jit_state_t *_jit, uint32_t pc, const InstructionInfo &last_info, bool first_instruction);
	void jit_end_of_block(jit_state_t *_jit, uint32_t pc, const InstructionInfo &last_info);

	void jit_handle_delay_slot(jit_state_t *_jit, const InstructionInfo &last_info, uint32_t base_pc, uint32_t end_pc);
	void jit_handle_impossible_delay_slot(jit_state_t *_jit, const InstructionInfo &info, const InstructionInfo &last_info,
	                                      uint32_t base_pc, uint32_t end_pc);
	void jit_handle_latent_delay_slot(jit_state_t *_jit, const InstructionInfo &last_info);
	void jit_mark_block_entries(uint32_t pc, uint32_t end, bool *block_entries);
	void jit_emit_load_operation(jit_state_t *_jit, uint32_t pc, uint32_t instr,
	                             void (*jit_emitter)(jit_state_t *_jit, unsigned, unsigned, unsigned), const char *asmop,
	                             jit_pointer_t rsp_unaligned_op,
	                             uint32_t endian_flip,
	                             const InstructionInfo &last_info);
	void jit_emit_store_operation(jit_state_t *_jit, uint32_t pc, uint32_t instr,
	                              void (*jit_emitter)(jit_state_t *_jit, unsigned, unsigned, unsigned), const char *asmop,
	                              jit_pointer_t rsp_unaligned_op,
	                              uint32_t endian_flip,
	                              const InstructionInfo &last_info);

	static void jit_begin_call(jit_state_t *_jit);
	static void jit_end_call(jit_state_t *_jit, jit_pointer_t ptr);
	void jit_save_illegal_cond_branch_taken(jit_state_t *_jit);
	static void jit_restore_illegal_cond_branch_taken(jit_state_t *_jit, unsigned reg);
	static void jit_clear_illegal_cond_branch_taken(jit_state_t *_jit, unsigned tmp_reg);
	void jit_save_indirect_register(jit_state_t *_jit, unsigned mips_register);
	static void jit_load_indirect_register(jit_state_t *_jit, unsigned jit_reg);
	static void jit_save_illegal_indirect_register(jit_state_t *_jit);
	static void jit_load_illegal_indirect_register(jit_state_t *_jit, unsigned jit_reg);

	std::string mips_disasm;
	struct Link
	{
		jit_node_t *node;
		unsigned local_index;
	};
	std::vector<Link> local_branches;

	RegisterCache regs;
	Allocator allocator;

#ifdef PRSP_VU_FAST
	struct VUStub
	{
		const uint32_t *src = nullptr; // body in the plugin image (without the final ret)
		unsigned words = 0;
		const void *addr = nullptr; // copy (with ret) in the JIT code region, reachable with BL
		bool ok = false;
	};
	VUStub vu_stubs[RSP_VU_STUB_COUNT];
	bool vu_fast_enabled = true;
	bool div_inline_enabled = true;
	unsigned vu_inline_words = 8; // VU bodies up to this many words are copied inline instead of called
	int v0_holds = -1; // vector register whose value v0 currently holds (-1 = unknown)
	void init_vu_stubs();
	bool jit_emit_vu_fast(jit_state_t *_jit, uint32_t instr);
	void jit_emit_vu_divide(jit_state_t *_jit, uint32_t instr);

	// Vector load / store fast paths with out-of-line slow paths (see jit_emit_ls_fast).
	struct SlowLS
	{
		jit_node_t *guard;
		jit_node_t *cont;
		void (*helper)(CPUState *, unsigned, unsigned, unsigned);
		unsigned rt, e, sum_reg;
		RegisterCache::Saved saved[RegisterCache::MaxSaved];
		unsigned num_saved;
		int reload_v0; // vector register to reload into v0 after the helper call, or -1
		unsigned prof_id;
		bool inline_lqv; // handled by NEON byte shuffling instead of a helper call (LQV, e == 0, unaligned address)
		unsigned vreg_offset; // byte offset of the destination vector register in CPUState (inline_lqv)
		unsigned addr_reg;
	};
	std::vector<SlowLS> slow_ls;
	bool jit_emit_ls_fast(jit_state_t *_jit, uint32_t instr);
	void jit_emit_slow_paths(jit_state_t *_jit);
#endif
};
} // namespace JIT
} // namespace RSP

#endif