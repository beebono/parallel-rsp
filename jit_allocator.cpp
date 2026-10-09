#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <unistd.h>
#include <sys/mman.h>
#endif
#include <limits>
#include <algorithm>

#include "jit_allocator.hpp"

namespace RSP
{
namespace JIT
{
#ifdef IOS // iOS/tvOS is 64bit but does not allow an infinite amount of VA space
static constexpr bool huge_va = false;
#else
static constexpr bool huge_va = std::numeric_limits<size_t>::max() > 0x100000000ull;
#endif
// On 64-bit systems, we will never allocate more than one block, this is important since we must ensure that
// relative jumps are reachable in 32-bits.
// We won't actually allocate 1 GB on 64-bit, but just reserve VA space for it, which we have basically an infinite amount of.
static constexpr size_t block_size = huge_va ? (1024 * 1024 * 1024) : (2 * 1024 * 1024);
Allocator::~Allocator()
{
#ifdef _WIN32
	for (auto &block : blocks)
		VirtualFree(block.code, 0, MEM_RELEASE);
#else
	for (auto &block : blocks)
		munmap(block.code, block.size);
#endif
}

static size_t align_page(size_t offset)
{
#if defined(__APPLE__) && defined(__aarch64__)
	size_t pagesize = sysconf(_SC_PAGESIZE) - 1;
#else
	size_t pagesize = 4095;
#endif
	return (offset + pagesize) & ~size_t(pagesize);
}

static bool commit_read_write(void *ptr, size_t size)
{
#ifdef _WIN32
	return VirtualAlloc(ptr, size, MEM_COMMIT, PAGE_READWRITE) == ptr;
#else
	return mprotect(ptr, size, PROT_READ | PROT_WRITE) == 0;
#endif
}

static bool commit_execute(void *ptr, size_t size)
{
#ifdef _WIN32
	DWORD old_protect;
	return VirtualProtect(ptr, align_page(size), PAGE_EXECUTE, &old_protect) != 0;
#else
	return mprotect(ptr, size, PROT_EXEC) == 0;
#endif
}

// On Linux-like systems regions are packed back to back (64-byte aligned) instead of one page each: the page-aligned layout
// put every hot region entry on the same L1I sets and wasted about half of the code pages (more I-cache and I-TLB misses).
#if !defined(_WIN32) && !defined(IOS) && !(defined(__APPLE__) && defined(__aarch64__))
#define PRSP_PACK_CODE 1
#endif

bool Allocator::commit_code(void *code, size_t size)
{
#ifdef PRSP_PACK_CODE
	uintptr_t begin = reinterpret_cast<uintptr_t>(code) & ~uintptr_t(4095);
	uintptr_t end = (reinterpret_cast<uintptr_t>(code) + size + 4095) & ~uintptr_t(4095);
	return commit_execute(reinterpret_cast<void *>(begin), end - begin);
#else
	return commit_execute(code, size);
#endif
}

void *Allocator::allocate_code(size_t size)
{
#ifdef PRSP_PACK_CODE
	size = (size + 63) & ~size_t(63);
#else
	size = align_page(size);
#endif
	if (blocks.empty())
		blocks.push_back(reserve_block(std::max(size, block_size)));

	auto *block = &blocks.back();
	if (!block->code)
		return nullptr;

#ifdef PRSP_PACK_CODE
	block->offset = (block->offset + 63) & ~size_t(63);
#else
	block->offset = align_page(block->offset);
#endif
	if (block->offset + size > block->size)
		block = nullptr;

	if (!block)
	{
		if (huge_va)
			abort();
		blocks.push_back(reserve_block(std::max(size, block_size)));
		block = &blocks.back();
	}

	if (!block || !block->code)
		return nullptr;

	void *ret = block->code + block->offset;
	block->offset += size;
	last_code = static_cast<uint8_t *>(ret);
	last_size = size;

#ifdef PRSP_PACK_CODE
	uintptr_t begin = reinterpret_cast<uintptr_t>(ret) & ~uintptr_t(4095);
	uintptr_t end = (reinterpret_cast<uintptr_t>(ret) + size + 4095) & ~uintptr_t(4095);
	if (!commit_read_write(reinterpret_cast<void *>(begin), end - begin))
		return nullptr;
#else
	if (!commit_read_write(ret, size))
		return nullptr;
#endif
	return ret;
}

void Allocator::shrink_last(void *code, size_t used)
{
#ifdef PRSP_PACK_CODE
	if (blocks.empty() || code != last_code)
		return;
	used = (used + 63) & ~size_t(63);
	if (used < last_size)
	{
		blocks.back().offset -= last_size - used;
		last_size = used;
	}
#else
	(void)code;
	(void)used;
#endif
}

Allocator::Block Allocator::reserve_block(size_t size)
{
	Block block;
#ifdef _WIN32
	block.code = static_cast<uint8_t *>(VirtualAlloc(nullptr, size, MEM_RESERVE, PAGE_READWRITE));
	block.size = size;
	return block;
#else
	block.code = static_cast<uint8_t *>(mmap(nullptr, size, PROT_NONE, MAP_ANONYMOUS | MAP_PRIVATE,
	                                         -1, 0));
	block.size = size;
	return block;
#endif
}
}
}
