// Native-routine hook for a host that embeds parallel-rsp (arMLEt). Everything here is compiled only with PRSP_NATIVE defined; without it
// the fork is unchanged (same object code). The C API at the bottom is what the host calls; the host fills a PrspNativeHost and installs it.
//
// The idea: the host knows microcode routines (a vertex transform, a triangle setup) that it can run as native code on the RSP state. The
// JIT hands control to the host when it is about to enter such a routine, the host runs it and says where the RSP continues, or declines
// and the JIT runs its own code for the routine as it always did.
//
//  * An *entry* is an IMEM byte pc (a routine's first instruction). A block lookup (get_jit_block) that misses on an entry pc asks the host's
//    `lookup` whether a routine applies to the IMEM content as it is now. If it does, blocks[] gets a small stub block instead of a compiled
//    region, so a block that has no routine costs nothing. A stub lives in blocks[] like any compiled region (IMEM DMA that changes the
//    code removes it; the next entry looks up again, so the match is keyed on the content, not on the pc), and is also removed when a DMA
//    changes a 256-byte block that overlaps [dep_lo, dep_hi) of the match (the extent the host's decision depended on).
//  * A *split* pc is an entry or an exit of a routine. Compiled regions never extend across a split pc (they end just before it and continue
//    through the enter thunk), so a jump or a fall-through to an entry reaches its stub, and the JIT can be stopped at an exit.
//  * `enter` runs when the JIT reaches a stub: at a block boundary, with no delay slot pending (otherwise the stub declines by itself).
//  * The *oracle* is the JIT's own code: prsp_native_oracle() runs it from an entry until an exit pc is reached, nested in the host's `enter`.
//    This is how a host verifies a routine: save, run native in a sandbox, restore, run the oracle, compare.
#ifndef PRSP_NATIVE_H__
#define PRSP_NATIVE_H__

#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

#define PRSP_NATIVE_DECLINE 0
#define PRSP_NATIVE_DONE 1

// ReturnMode values the host may hand back in *mode (state.hpp): continue, or leave the run loop as the instruction that caused it would.
#define PRSP_NATIVE_MODE_CONTINUE 1
#define PRSP_NATIVE_MODE_BREAK 2
#define PRSP_NATIVE_MODE_DMA_READ 3
#define PRSP_NATIVE_MODE_CHECK_FLAGS 4
#define PRSP_NATIVE_MODE_STOP 5 // the oracle reached its stop pc (returned by prsp_native_oracle as 0, listed for completeness)

typedef struct PrspNativeMatch
{
	unsigned cookie;           // the host's id of the routine; passed back to enter()
	unsigned dep_lo, dep_hi;   // IMEM byte range [lo, hi) the decision depended on, besides the region itself (hi <= lo: none)
	unsigned dep2_lo, dep2_hi; // a second range (hi <= lo: none): the stub is dropped when a changed block overlaps either, not the gap between them
} PrspNativeMatch;

// Side effects the SP registers can have, in program order. Recorded while prsp_native_record(1) is on, by RSP_MTC0 and the DMA code,
// whether the JIT or the host (through prsp_native_mtc0) caused them.
enum
{
	PRSP_EV_DMA_RD = 1,    // a=SP_MEM_ADDR b=SP_DRAM_ADDR c=SP_RD_LEN written
	PRSP_EV_DMA_WR = 2,    // a=SP_MEM_ADDR b=SP_DRAM_ADDR c=SP_WR_LEN written; hash=FNV-1a-64 of the bytes written to RDRAM (rows in order, with their address)
	PRSP_EV_STATUS = 3,    // a=value written to SP_STATUS
	PRSP_EV_SEMAPHORE = 4,
	PRSP_EV_CMD_START = 5, // a=value written to DPC_START
	PRSP_EV_CMD_END = 6,   // a=DPC_CURRENT b=DPC_END written c=DPC_STATUS; hash=FNV-1a-64 of the list bytes [a, b) (from DMEM when XBUS is set)
	PRSP_EV_CMD_STATUS = 7 // a=value written to DPC_STATUS
};

typedef struct PrspNativeEvent
{
	uint32_t kind, a, b, c;
	uint64_t hash;
} PrspNativeEvent;

typedef struct PrspNativeHost
{
	void *user;
	// A region is about to be compiled at entry pc `pc` (byte offset, a multiple of 4): does a routine apply to the current IMEM?
	int (*lookup)(void *user, unsigned pc, const uint32_t *imem, PrspNativeMatch *m);
	// The JIT reached the entry of a matched routine. The RSP state is the live one (prsp_native_view). Return PRSP_NATIVE_DECLINE with the
	// state untouched, or PRSP_NATIVE_DONE with *next_pc the pc to continue at (byte offset) and *mode PRSP_NATIVE_MODE_CONTINUE, or a
	// mode that leaves the run loop (the run loop then returns it with pc = *next_pc).
	int (*enter)(void *user, unsigned cookie, unsigned pc, unsigned *next_pc, int *mode);
	// A recorded side effect (see PRSP_EV_*); may be null.
	void (*event)(void *user, const PrspNativeEvent *ev);
} PrspNativeHost;

// The live RSP state. Valid for the life of the plugin (the CPU does not move).
typedef struct PrspNativeView
{
	uint32_t *sr;               // [33]: GPRs 1..31, [0] is not maintained, [32] is the JIT's scratch
	uint16_t (*vr)[8];          // [32][8]: vector registers, element i of register n at vr[n][i]
	uint16_t *acc;              // [24]: HI at 0, MD at 8, LO at 16 (8 lanes each)
	uint16_t (*flags)[16];      // [3][16]: VCO, VCC, VCE; lane value 0 or 0xffff; bit n (0..7) is lane 8+n, bit 8+n is lane n
	int16_t *div_out, *div_in;
	int8_t *dp_flag;
	uint32_t *dmem;             // 1024 host words holding the big-endian words of DMEM
	uint32_t *imem;             // the same for IMEM
	uint32_t *rdram;
	uint32_t *cr[16];           // the SP/DPC registers (cp0)
	uint32_t *irq;              // MI_INTR
	short *mfc0_count;          // [32]
	uint32_t *pc;               // the JIT's pc (byte offset)
	uint32_t *has_delay_slot;
	uint32_t *branch_target;
	uint32_t *dirty_blocks;     // IMEM 256-byte blocks changed by SP DMA since the JIT last looked (set by the DMA, cleared by the run loop)
} PrspNativeView;

// A copy of everything that is RSP state (not RDRAM). Fixed layout, so it can be written to a file.
#define PRSP_NATIVE_SNAP_MAGIC 0x50534e31u // "PSN1"
typedef struct PrspNativeSnap
{
	uint32_t magic, size;
	uint32_t pc, has_delay_slot, branch_target, irq, dirty_blocks;
	int32_t sp_status_timeout;
	int16_t mfc0_count[32];
	uint32_t cr[16];
	uint32_t sr[33];
	uint16_t vr[32][8];
	uint16_t flags[3][16];
	uint16_t acc[24];
	int16_t div_out, div_in;
	int8_t dp_flag, pad_[3];
	uint32_t dmem[1024];
	uint32_t imem[1024];
} PrspNativeSnap;

// The poll time-out policy at the start of a task (PRSP_POLL_IDLE=N, see parallel.cpp): resets the idle-poll tracker; the hosts call it before they snapshot the JIT.
void prsp_poll_task_begin(unsigned pc);
// The same policy for a host that runs part of a task (arMLEt, F3DEX2 front-end): limit N (0: off); whether the last SP_STATUS poll was an idle cut; the tracker state
// (save before a run that may be undone, load to undo it); and the state a cut leaves: sr[32] = the scalar registers at the cut, N idle polls, cut flagged.
typedef struct PrspPollState
{
	uint32_t sr[32];
	int idle, cut;
} PrspPollState;
int prsp_poll_idle_limit(void);
int prsp_poll_idle_cut(void);
void prsp_poll_save(PrspPollState *s);
void prsp_poll_load(const PrspPollState *s);
void prsp_poll_idle_mark(const uint32_t *sr);
void prsp_native_install(const PrspNativeHost *host);                 // null: remove (all stubs go)
void prsp_native_set_pcs(const uint16_t *entries, unsigned n_entries, const uint16_t *splits, unsigned n_splits); // byte pcs; replaces the sets
void prsp_native_view(PrspNativeView *v);
void prsp_native_save(PrspNativeSnap *s);
void prsp_native_load(const PrspNativeSnap *s);
// After prsp_native_load into a CPU whose compiled code may be for other IMEM contents (an offline harness): drops what no longer matches.
void prsp_native_resync(void);

// Runs the JIT's own code from the entry `entry_pc` (which must have a stub installed) until it is about to execute `stop_pc`; stubs are
// not taken meanwhile. Returns 0 when it stopped there (the RSP pc is stop_pc), otherwise the ReturnMode that ended the run (BREAK,
// CHECK_FLAGS, DMA_READ), with the RSP in the state the JIT left it. The caller must not return PRSP_NATIVE_DECLINE after this.
int prsp_native_oracle(unsigned entry_pc, unsigned stop_pc);

// Side-effect recording and the sandbox. record(1): every effectful SP register write (PRSP_EV_*) goes to host->event. sandbox(1): SP DMA
// writes to RDRAM are done but their old bytes are kept; the RDP is not run (no ProcessRdpList, no deferral flush); sandbox(0) puts the
// RDRAM bytes back. The registers themselves are written as usual (the host saves and restores them with prsp_native_save/load).
void prsp_native_record(int on);
void prsp_native_sandbox(int on);

// An MTC0 from the host: as the instruction does it (returns the ReturnMode of RSP_MTC0).
int prsp_native_mtc0(unsigned rd, uint32_t value);
// An MFC0 from the host, for the status-poll timeout bookkeeping (rt is the register the ucode would have read into).
uint32_t prsp_native_mfc0(unsigned rd, unsigned rt, int *mode);

#ifdef __cplusplus
}
#endif

#endif
