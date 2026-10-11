// Observation hooks for tools that embed parallel-rsp (profilers, code maps). Everything here expands to nothing unless
// PRSP_HOOKS is defined, so the normal build is unchanged (same object code). With PRSP_HOOKS defined a host installs a
// PrspHooks table with prsp_hooks_install(); every member may be null. The hooks only observe: they must not change RSP,
// JIT or memory state.
#ifndef PRSP_HOOKS_H__
#define PRSP_HOOKS_H__

#ifdef PRSP_HOOKS
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <string>
#include <vector>

// One compiled region (JIT block) as emitted. `offs` has `count + 1` entries: byte offsets from `code` of the code emitted
// for IMEM word pc_word + i (offs[i] up to offs[i + 1]); offs[count] is where the end-of-block / latent delay slot / slow
// path code starts (up to `size`). Code emitted for a branch's delay slot is part of the instruction that precedes it in
// the stream, as emitted (lightning has no finer notion).
struct PrspRegionInfo
{
	const void *code;
	size_t size;
	unsigned pc_word;  // first IMEM word
	unsigned count;    // instructions
	uint64_t hash;     // the region key (hash_imem)
	const unsigned *offs;
	const uint32_t *instrs; // the `count` instruction words (host order, as in IMEM)
};

struct PrspHooks
{
	uint64_t (*now)();                                   // timestamp for the lookup/compile timings below (any unit)
	void (*region)(const PrspRegionInfo *r);             // after a region was emitted
	// 'S' VU stub, 'T' enter/return thunks: `addr` is where the code runs, `bytes` a readable copy of it (the code pages are execute-only
	// once committed)
	void (*misc)(char kind, const void *addr, const void *bytes, size_t size, const char *name);
	// get_jit_block miss finished (blocks[] was empty): `ticks` from entry to return, `compile_ticks` inside jit_region (0 if
	// the region was cached), `hashed` words hashed
	void (*lookup)(uint64_t ticks, uint64_t compile_ticks, unsigned hashed);
	void (*imem_dma)(unsigned blocks, unsigned changed); // an SP DMA row into IMEM: 256-byte blocks covered / whose bytes changed
	unsigned (*site)(unsigned pc, uint32_t instr);       // JIT time, PRSP_PROFILE: the counter index for an instruction
	void (*hit)(unsigned site);                          // run time, PRSP_PROFILE: before each instruction executes
};

// SP DMA operations and IMEM invalidations are counted inline (a hook call per DMA operation cost more than the counting): a host reads and diffs them
struct PrspCounters
{
	uint64_t dma_rd, dma_rd_bytes, dma_rd_imem, dma_wr, dma_wr_bytes; // SP DMA operations (all rows of one), bytes; dma_rd_imem: reads into IMEM
	uint64_t inval, inval_blocks;                                      // invalidate_code with something dirty, 256-byte blocks cleared
};
namespace PrspHookState
{
inline PrspCounters &counters()
{
	static PrspCounters c = {};
	return c;
}
inline PrspHooks &table()
{
	static PrspHooks h = {};
	return h;
}
struct Misc
{
	char kind;
	const void *addr;
	std::vector<uint8_t> bytes;
	std::string name;
};
inline std::vector<Misc> &early()
{
	static std::vector<Misc> v; // thunks and VU stubs are created by the CPU's constructor, before a host can install hooks
	return v;
}
inline bool &installed()
{
	static bool b = false;
	return b;
}
} // namespace PrspHookState

inline void prsp_hooks_install(const PrspHooks &h)
{
	PrspHookState::table() = h;
	PrspHookState::installed() = true;
	if (h.misc)
		for (auto &m : PrspHookState::early())
			h.misc(m.kind, m.addr, m.bytes.data(), m.bytes.size(), m.name.c_str());
}

#define PRSP_HOOKS_ON() (PrspHookState::installed())
#define PRSP_HOOK_ONLY(...) __VA_ARGS__
#define PRSP_HOOK_CALL(member, ...) \
	do { if (PrspHookState::table().member) PrspHookState::table().member(__VA_ARGS__); } while (0)

// jit_region: a marker node before each instruction (and one after the last) gives the code offset of each instruction
#define PRSP_HOOK_REGION_DECL() std::vector<jit_node_t *> prsp_marks
#define PRSP_HOOK_REGION_MARK() \
	do { if (PrspHookState::table().region) prsp_marks.push_back(jit_note(NULL, 0)); } while (0)
// the offsets are read right after jit_emit() (the nodes are gone after jit_clear_state), the callback comes after the size is final
#define PRSP_HOOK_REGION_OFFS(code_) \
	std::vector<unsigned> prsp_offs; \
	if (PrspHookState::table().region && prsp_marks.size()) \
		for (auto *n : prsp_marks) \
			prsp_offs.push_back(unsigned(static_cast<uint8_t *>(jit_address(n)) - static_cast<uint8_t *>(code_)))
#define PRSP_HOOK_REGION_CALL(code_, bytes_, pc_word_, count_, hash_) \
	do { \
		if (PrspHookState::table().region && prsp_offs.size() == size_t(count_) + 1) \
		{ \
			PrspRegionInfo prsp_ri = { code_, size_t(bytes_), pc_word_, count_, hash_, prsp_offs.data(), state.imem + (pc_word_) }; \
			PrspHookState::table().region(&prsp_ri); \
		} \
	} while (0)

#define PRSP_HOOK_MISC(kind_, code_, bytes_, name_) \
	do { \
		PrspHookState::early().push_back({ kind_, code_, std::vector<uint8_t>(reinterpret_cast<const uint8_t *>(code_), \
		                                                                       reinterpret_cast<const uint8_t *>(code_) + (bytes_)), name_ }); \
		PRSP_HOOK_CALL(misc, kind_, code_, code_, size_t(bytes_), name_); \
	} while (0)

#define PRSP_HOOK_NOW() (PrspHookState::table().now ? PrspHookState::table().now() : uint64_t(0))
#define PRSP_HOOK_LOOKUP(t0, compile_ticks, hashed) \
	PRSP_HOOK_CALL(lookup, PRSP_HOOK_NOW() - (t0), compile_ticks, hashed)
#define PRSP_HOOK_INVALIDATE(n_) \
	do { auto &c_ = PrspHookState::counters(); c_.inval++; c_.inval_blocks += (n_); } while (0)
#define PRSP_HOOK_IMEM_DMA(blocks, changed) PRSP_HOOK_CALL(imem_dma, blocks, changed)
#define PRSP_HOOK_DMA(write_, bytes_, to_imem_) \
	do \
	{ \
		auto &c_ = PrspHookState::counters(); \
		if (write_) { c_.dma_wr++; c_.dma_wr_bytes += (bytes_); } \
		else { c_.dma_rd++; c_.dma_rd_bytes += (bytes_); c_.dma_rd_imem += (to_imem_) ? 1 : 0; } \
	} while (0)

// PRSP_PROFILE: prsp_prof_hit(id, site) gets the counter index of the instruction as a second argument
#define PRSP_HOOK_HIT_PARAM , unsigned prsp_site
#define PRSP_HOOK_HIT(...) PRSP_HOOK_CALL(hit, prsp_site)
#define PRSP_HOOK_HIT_PUSH(pc, instr) \
	jit_pushargi(PrspHookState::table().site ? PrspHookState::table().site(pc, instr) : 0u)

#else // !PRSP_HOOKS

#define PRSP_HOOKS_ON() 0
#define PRSP_HOOK_ONLY(...)
#define PRSP_HOOK_REGION_DECL()
#define PRSP_HOOK_REGION_MARK()
#define PRSP_HOOK_REGION_OFFS(code_)
#define PRSP_HOOK_REGION_CALL(code_, bytes_, pc_word_, count_, hash_)
#define PRSP_HOOK_MISC(kind_, code_, bytes_, name_)
#define PRSP_HOOK_NOW() 0
#define PRSP_HOOK_LOOKUP(t0, compile_ticks, hashed)
#define PRSP_HOOK_INVALIDATE(n)
#define PRSP_HOOK_IMEM_DMA(blocks, changed)
#define PRSP_HOOK_DMA(write, bytes, to_imem)
#define PRSP_HOOK_HIT_PARAM
#define PRSP_HOOK_HIT(...)
#define PRSP_HOOK_HIT_PUSH(pc, instr)

#endif
#endif
