#include "texture_share_vk_client.hpp"
#include "texture_share_vk/texture_share_vk_client.h"
#include <utility>

TextureShareVkClient::ClientImageDataGuard::ClientImageDataGuard(::ClientImageDataGuard *data)
	: _data(data)
{}

TextureShareVkClient::ClientImageDataGuard::~ClientImageDataGuard()
{
	vk_client_image_data_guard_destroy(this->_data);
	this->_data = nullptr;
}

TextureShareVkClient::ClientImageDataGuard::ClientImageDataGuard(ClientImageDataGuard &&other)
	: _data(std::move(other._data))
{
	other._data = nullptr;
}

TextureShareVkClient::ClientImageDataGuard &TextureShareVkClient::ClientImageDataGuard::operator=(
	ClientImageDataGuard &&other)
{
	this->_data = std::move(other._data);
	other._data = nullptr;

	return *this;
}

const ShmemDataInternal *TextureShareVkClient::ClientImageDataGuard::read() const
{
	return vk_client_image_data_guard_read(this->_data);
}

TextureShareVkClient::TextureShareVkClient() {}

TextureShareVkClient::TextureShareVkClient::TextureShareVkClient(TextureShareVkClient &&other)
	: _client(std::move(other._client))
{
	other._client = nullptr;
}

TextureShareVkClient &TextureShareVkClient::TextureShareVkClient::operator=(TextureShareVkClient &&other)
{
	this->_client = std::move(other._client);
	other._client = nullptr;

	return *this;
}

TextureShareVkClient::~TextureShareVkClient()
{
	this->destroy_client();
}

bool TextureShareVkClient::init(VkSetup *vk_setup, const char *socket_path, uint64_t timeout_in_millis)
{
	this->destroy_client();
	this->_client = vk_client_new(socket_path, vk_setup, timeout_in_millis);

	return this->_client != nullptr;
}

bool TextureShareVkClient::init_with_server_launch(VkSetup *vk_setup, const char *socket_path,
                                                   uint64_t client_timeout_in_millis, const char *server_program,
                                                   const char *server_lock_path, const char *server_socket_path,
                                                   const char *shmem_prefix,
                                                   uint64_t server_connection_timeout_in_millia,
                                                   uint64_t server_spawn_timeout_in_millis)
{
	this->destroy_client();
	this->_client = vk_client_new_with_server_launch(
		socket_path, vk_setup, client_timeout_in_millis, server_program, server_lock_path, server_socket_path,
		shmem_prefix, server_connection_timeout_in_millia, server_spawn_timeout_in_millis);

	return this->_client != nullptr;
}

void TextureShareVkClient::destroy_client()
{
	vk_client_destroy(this->_client);
	this->_client = nullptr;
}

int TextureShareVkClient::find_image(const char *image_name, bool force_update)
{
	if(!this->_client)
		return -1;

	return vk_client_find_image(this->_client, image_name, force_update);
}

TextureShareVkClient::ClientImageDataGuard TextureShareVkClient::find_image_data(const char *image_name,
                                                                                 bool force_update)
{
	if(!this->_client)
		return nullptr;

	return ClientImageDataGuard(vk_client_find_image_data(this->_client, image_name, force_update));
}

int TextureShareVkClient::send_image(const char *image_name, VkImage image, VkImageLayout layout, VkFence fence,
                                     VkOffset3D *extents)
{
	if(!this->_client)
		return -1;

	return vk_client_send_image(this->_client, image_name, image, layout, fence, extents);
}

int TextureShareVkClient::recv_image(const char *image_name, VkImage image, VkImageLayout layout, VkFence fence,
                                     VkOffset3D *extents)
{
	if(!this->_client)
		return -1;

	return vk_client_recv_image(this->_client, image_name, image, layout, fence, extents);
}
