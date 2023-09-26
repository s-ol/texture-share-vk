#ifndef IPC_MEMORY_PROCESSOR_VK_H
#define IPC_MEMORY_PROCESSOR_VK_H

#include "texture_share_vk/ipc_memory/ipc_memory.h"
#include "texture_share_vk/platform/daemon_comm.h"
#include "texture_share_vk/texture_share_vk.h"

#include <map>
#include <set>


class IpcMemoryProcessorVk
        : public IpcMemory
{
	public:
		IpcMemoryProcessorVk(const std::string &ipc_cmd_memory_segment = IpcMemory::DEFAULT_IPC_CMD_MEMORY_NAME.data(),
		                     const std::string &ipc_map_memory_segment = IpcMemory::DEFAULT_IPC_MAP_MEMORY_NAME.data());
		~IpcMemoryProcessorVk();

		void InitializeVulkan();
		void CleanupVulkan();

		char ProcessCmd(uint64_t micro_sec_wait_time = DEFAULT_CMD_WAIT_TIME);
		void CleanupLocks();

		bool CheckConnectedProcs();

	private:
		TextureShareVk _vk_data;
		std::map<IpcMemory::IMAGE_NAME_T, SharedImageVk> _image_data;

		std::set<DaemonComm::PROC_T> _registered_pids;

		uint64_t _next_image_id = 1;

		char ProcessRegisterProcCmd(const ipc_commands::IpcCmdRegisterProc &ipc_cmd);
		char ProcessImageInitCmd(const ipc_commands::IpcCmdImageInit &ipc_cmd);
		char ProcessRenameCmd(const ipc_commands::IpcCmdRename &ipc_cmd);
		char ProcessHandleRequestCmd(const ipc_commands::IpcCmdRequestImageHandles &ipc_cmd);
};

#endif // IPC_MEMORY_PROCESSOR_VK_H
