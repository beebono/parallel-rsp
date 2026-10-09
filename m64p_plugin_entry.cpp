// Mupen64Plus RSP plugin entry points around the parallel-rsp Lightning JIT.
// Built by the PARALLEL_RSP_M64P_PLUGIN CMake option (defines PARALLEL_INTEGRATION).
#include "m64p_types.h"
#include "m64p_plugin.h"

extern "C"
{
	unsigned int parallelRSPDoRspCycles(unsigned int cycles);
	m64p_error parallelRSPPluginGetVersion(m64p_plugin_type *, int *, int *, const char **, int *);
	void parallelRSPRomClosed(void);
	void parallelRSPInitiateRSP(RSP_INFO info, unsigned int *cc);

	EXPORT m64p_error CALL PluginStartup(m64p_dynlib_handle, void *, void (*)(void *, int, const char *))
	{
		return M64ERR_SUCCESS;
	}

	EXPORT m64p_error CALL PluginShutdown(void)
	{
		return M64ERR_SUCCESS;
	}

	EXPORT m64p_error CALL PluginGetVersion(m64p_plugin_type *type, int *version, int *api, const char **name,
	                                        int *caps)
	{
		m64p_error e = parallelRSPPluginGetVersion(type, version, api, name, caps);
		if (name)
			*name = "parallel-rsp (Lightning JIT)";
		return e;
	}

	EXPORT unsigned int CALL DoRspCycles(unsigned int cycles)
	{
		return parallelRSPDoRspCycles(cycles);
	}

	EXPORT void CALL InitiateRSP(RSP_INFO info, unsigned int *cycle_count)
	{
		parallelRSPInitiateRSP(info, cycle_count);
	}

	EXPORT void CALL RomClosed(void)
	{
		parallelRSPRomClosed();
	}
}
