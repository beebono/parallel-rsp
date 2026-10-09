#ifndef RSP_VU_INLINE_H
#define RSP_VU_INLINE_H
// See vu_inline.cpp. Table of register-convention VU op bodies (aarch64 only).
#define RSP_VU_STUB_LIST(X) \
	X(VMULF) \
	X(VMULU) \
	X(VMUDL) \
	X(VMUDM) \
	X(VMUDN) \
	X(VMUDH) \
	X(VMACF) \
	X(VMACU) \
	X(VMADL) \
	X(VMADM) \
	X(VMADN) \
	X(VMADH) \
	X(VADD) \
	X(VSUB) \
	X(VABS) \
	X(VADDC) \
	X(VSUBC) \
	X(VLT) \
	X(VEQ) \
	X(VNE) \
	X(VGE) \
	X(VCL) \
	X(VCH) \
	X(VCR) \
	X(VMRG) \
	X(VAND) \
	X(VNAND) \
	X(VOR) \
	X(VNOR) \
	X(VXOR) \
	X(VNXOR) \
	X(RESERVED)

enum RSPVUStubId
{
#define X(name) RSP_VUS_ID_##name,
	RSP_VU_STUB_LIST(X)
#undef X
	RSP_VU_STUB_COUNT
};

extern "C"
{
	extern const void *const rsp_vu_stub_fn[RSP_VU_STUB_COUNT];
	extern const char *const rsp_vu_stub_name[RSP_VU_STUB_COUNT];
}
#endif
