use ash::vk;

use super::vk_cpu_buffer::VkCpuBuffer;
use super::vk_shared_image::VkSharedImage;
use crate::vk_setup::VkSetup;
use crate::vk_shared_image::{self, ImageBlit};

struct VkCpuSharedImage {
	image: VkSharedImage,
	cpu_buffer: VkCpuBuffer,
}

impl Drop for VkCpuSharedImage {
	fn drop(&mut self) {
		println!("Warning: VkCpuSharedImage should be manually destroyed, not dropped");
	}
}

impl VkCpuSharedImage {
	pub fn new(
		vk_setup: &VkSetup,
		width: u32,
		height: u32,
		format: vk::Format,
		id: u32,
	) -> Result<VkCpuSharedImage, vk::Result> {
		let vk_shared_image = VkSharedImage::new(vk_setup, width, height, format, id)?;
		Self::from_shared_image(vk_setup, vk_shared_image)
	}

	pub fn from_shared_image(
		vk_setup: &VkSetup,
		image: VkSharedImage,
	) -> Result<VkCpuSharedImage, vk::Result> {
		let cpu_buffer = VkCpuBuffer::new(vk_setup, image.data.allocation_size)?;
		Ok(VkCpuSharedImage { image, cpu_buffer })
	}

	pub fn destroy(self, vk_setup: &VkSetup) {
		self.cpu_buffer._destroy(vk_setup);
		self.image._destroy(vk_setup);

		std::mem::forget(self);
	}

	// pub fn to_shared_image(self, vk_setup: &VkSetup) -> VkSharedImage {
	// 	self.cpu_buffer._destroy(vk_setup);
	// 	std::mem::forget(self);

	// 	self.image
	// }
}

impl ImageBlit for VkCpuSharedImage {
	fn send_image_blit_with_extents(
		&self,
		vk_setup: &VkSetup,
		dst_image: &vk::Image,
		orig_dst_image_layout: vk::ImageLayout,
		target_dst_image_layout: vk::ImageLayout,
		dst_image_extent: &[vk::Offset3D; 2],
		fence: vk::Fence,
	) -> Result<(), vk::Result> {
		let src_image_extent = [
			vk::Offset3D { x: 0, y: 0, z: 0 },
			vk::Offset3D {
				x: self.image.data.width as i32,
				y: self.image.data.height as i32,
				z: 1,
			},
		];

		let send_image_cmd_fcn = |cmd_bud: vk::CommandBuffer| -> Result<(), vk::Result> {
			// Pipeline steps:
			// 1. HOST: CPU -> GPU buffer transfer for self.cpu_buffer
			// 2. TRANSFER: Copy buffer to image
			// 3. TRANSFER: Blit image

			// 1. HOST
			let src_buf_barrier = VkCpuBuffer::gen_buffer_memory_barrier(
				self.cpu_buffer.buffer.handle,
				vk::AccessFlags::NONE,
				vk::AccessFlags::HOST_READ,
				self.cpu_buffer.buffer_size,
			);
			unsafe {
				vk_setup.vk_device.cmd_pipeline_barrier(
					cmd_bud,
					vk::PipelineStageFlags::TOP_OF_PIPE,
					vk::PipelineStageFlags::HOST,
					vk::DependencyFlags::default(),
					&[],
					&[src_buf_barrier],
					&[],
				);
			}

			// 2. TRANSFER
			let src_buf_barrier = VkCpuBuffer::gen_buffer_memory_barrier(
				self.cpu_buffer.buffer.handle,
				vk::AccessFlags::HOST_READ,
				vk::AccessFlags::TRANSFER_READ,
				self.cpu_buffer.buffer_size,
			);
			let src_image_barrier = VkSharedImage::gen_img_mem_barrier(
				self.image.image,
				self.image.image_layout,
				vk::ImageLayout::TRANSFER_DST_OPTIMAL,
				vk::AccessFlags::NONE,
				vk::AccessFlags::TRANSFER_WRITE,
			);
			unsafe {
				vk_setup.vk_device.cmd_pipeline_barrier(
					cmd_bud,
					vk::PipelineStageFlags::HOST,
					vk::PipelineStageFlags::TRANSFER,
					vk::DependencyFlags::default(),
					&[],
					&[src_buf_barrier],
					&[src_image_barrier],
				);
			}

			// Copy buffer to image
			unsafe {
				let copy_region = vk::BufferImageCopy::builder()
					.buffer_row_length(0)
					.buffer_image_height(0)
					.image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
					.image_extent(vk::Extent3D {
						width: self.image.data.width,
						height: self.image.data.height,
						depth: 1,
					})
					.image_subresource(vk::ImageSubresourceLayers {
						aspect_mask: vk::ImageAspectFlags::COLOR,
						base_array_layer: 0,
						layer_count: 1,
						mip_level: 0,
						..Default::default()
					})
					.build();
				vk_setup.vk_device.cmd_copy_buffer_to_image(
					cmd_bud,
					self.cpu_buffer.buffer.handle,
					self.image.image,
					vk::ImageLayout::TRANSFER_DST_OPTIMAL,
					&[copy_region],
				)
			}

			// 3. TRANSFER
			let src_buf_barrier = VkCpuBuffer::gen_buffer_memory_barrier(
				self.cpu_buffer.buffer.handle,
				vk::AccessFlags::TRANSFER_READ,
				vk::AccessFlags::NONE,
				self.cpu_buffer.buffer_size,
			);
			let src_img_barrier = VkSharedImage::gen_img_mem_barrier(
				self.image.image,
				vk::ImageLayout::TRANSFER_DST_OPTIMAL,
				vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
				vk::AccessFlags::TRANSFER_WRITE,
				vk::AccessFlags::TRANSFER_READ,
			);
			let dst_img_barrier = VkSharedImage::gen_img_mem_barrier(
				*dst_image,
				orig_dst_image_layout,
				vk::ImageLayout::TRANSFER_DST_OPTIMAL,
				vk::AccessFlags::NONE,
				vk::AccessFlags::TRANSFER_WRITE,
			);
			unsafe {
				vk_setup.vk_device.cmd_pipeline_barrier(
					cmd_bud,
					vk::PipelineStageFlags::TRANSFER,
					vk::PipelineStageFlags::TRANSFER,
					vk::DependencyFlags::default(),
					&[],
					&[src_buf_barrier],
					&[src_img_barrier, dst_img_barrier],
				);
			}

			// Blit image
			unsafe {
				let image_subresource_layer = vk::ImageSubresourceLayers::builder()
					.aspect_mask(vk::ImageAspectFlags::COLOR)
					.base_array_layer(0)
					.layer_count(1)
					.mip_level(0)
					.build();
				vk_setup.vk_device.cmd_blit_image(
					cmd_bud,
					self.image.image,
					vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
					*dst_image,
					vk::ImageLayout::TRANSFER_DST_OPTIMAL,
					&[vk::ImageBlit {
						src_offsets: src_image_extent,
						src_subresource: image_subresource_layer,
						dst_offsets: *dst_image_extent,
						dst_subresource: image_subresource_layer,
					}],
					vk::Filter::NEAREST,
				)
			}

			let src_img_barrier = VkSharedImage::gen_img_mem_barrier(
				self.image.image,
				vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
				self.image.image_layout,
				vk::AccessFlags::TRANSFER_READ,
				vk::AccessFlags::NONE,
			);
			let dst_img_barrier = VkSharedImage::gen_img_mem_barrier(
				*dst_image,
				vk::ImageLayout::TRANSFER_DST_OPTIMAL,
				target_dst_image_layout,
				vk::AccessFlags::TRANSFER_WRITE,
				vk::AccessFlags::NONE,
			);
			unsafe {
				vk_setup.vk_device.cmd_pipeline_barrier(
					cmd_bud,
					vk::PipelineStageFlags::TRANSFER,
					vk::PipelineStageFlags::BOTTOM_OF_PIPE,
					vk::DependencyFlags::default(),
					&[],
					&[],
					&[src_img_barrier, dst_img_barrier],
				);
			}

			Ok(())
		};

		self.cpu_buffer.sync_memory_from_cpu(vk_setup)?;
		vk_setup.immediate_submit_with_fence(
			vk_setup.vk_command_buffer,
			send_image_cmd_fcn,
			&[],
			&[],
			fence,
		)?;

		Ok(())
	}

	fn send_image_blit(
		&self,
		vk_setup: &VkSetup,
		dst_image: &vk::Image,
		orig_dst_image_layout: vk::ImageLayout,
		target_dst_image_layout: vk::ImageLayout,
		fence: vk::Fence,
	) -> Result<(), vk::Result> {
		let dst_image_extent = [
			vk::Offset3D { x: 0, y: 0, z: 0 },
			vk::Offset3D {
				x: self.image.data.width as i32,
				y: self.image.data.height as i32,
				z: 1,
			},
		];

		self.send_image_blit_with_extents(
			vk_setup,
			dst_image,
			orig_dst_image_layout,
			target_dst_image_layout,
			&dst_image_extent,
			fence,
		)
	}

	fn recv_image_blit_with_extents(
		&self,
		vk_setup: &VkSetup,
		src_image: &vk::Image,
		orig_src_image_layout: vk::ImageLayout,
		target_src_image_layout: vk::ImageLayout,
		src_image_extent: &[vk::Offset3D; 2],
		fence: vk::Fence,
	) -> Result<(), vk::Result> {
		let dst_image_extent = [
			vk::Offset3D { x: 0, y: 0, z: 0 },
			vk::Offset3D {
				x: self.image.data.width as i32,
				y: self.image.data.height as i32,
				z: 1,
			},
		];

		let recv_image_cmd_fcn = |cmd_bud: vk::CommandBuffer| -> Result<(), vk::Result> {
			// Pipeline steps:
			// 1. TRANSFER: Blit image
			// 2. TRANSFER: Copy image to CPU buffer
			// 3. HOST: GPU -> CPU transfer for self.cpu_buffer

			let src_img_barrier = VkSharedImage::gen_img_mem_barrier(
				*src_image,
				orig_src_image_layout,
				vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
				vk::AccessFlags::NONE,
				vk::AccessFlags::TRANSFER_READ,
			);
			let dst_img_barrier = VkSharedImage::gen_img_mem_barrier(
				self.image.image,
				self.image.image_layout,
				vk::ImageLayout::TRANSFER_DST_OPTIMAL,
				vk::AccessFlags::NONE,
				vk::AccessFlags::TRANSFER_WRITE,
			);
			unsafe {
				vk_setup.vk_device.cmd_pipeline_barrier(
					cmd_bud,
					vk::PipelineStageFlags::TOP_OF_PIPE,
					vk::PipelineStageFlags::TRANSFER,
					vk::DependencyFlags::default(),
					&[],
					&[],
					&[src_img_barrier, dst_img_barrier],
				);
			}

			// Blit image
			unsafe {
				let image_subresource_layer = vk::ImageSubresourceLayers::builder()
					.aspect_mask(vk::ImageAspectFlags::COLOR)
					.base_array_layer(0)
					.layer_count(1)
					.mip_level(0)
					.build();
				vk_setup.vk_device.cmd_blit_image(
					cmd_bud,
					*src_image,
					vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
					self.image.image,
					vk::ImageLayout::TRANSFER_DST_OPTIMAL,
					&[vk::ImageBlit {
						src_offsets: *src_image_extent,
						src_subresource: image_subresource_layer,
						dst_offsets: dst_image_extent,
						dst_subresource: image_subresource_layer,
					}],
					vk::Filter::NEAREST,
				)
			}

			let src_img_barrier = VkSharedImage::gen_img_mem_barrier(
				*src_image,
				vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
				target_src_image_layout,
				vk::AccessFlags::TRANSFER_READ,
				vk::AccessFlags::NONE,
			);
			let dst_img_barrier = VkSharedImage::gen_img_mem_barrier(
				self.image.image,
				vk::ImageLayout::TRANSFER_DST_OPTIMAL,
				vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
				vk::AccessFlags::TRANSFER_WRITE,
				vk::AccessFlags::TRANSFER_READ,
			);
			let dst_buffer_barrier = VkCpuBuffer::gen_buffer_memory_barrier(
				self.cpu_buffer.buffer.handle,
				vk::AccessFlags::NONE,
				vk::AccessFlags::TRANSFER_WRITE,
				self.cpu_buffer.buffer_size,
			);
			unsafe {
				vk_setup.vk_device.cmd_pipeline_barrier(
					cmd_bud,
					vk::PipelineStageFlags::TRANSFER,
					vk::PipelineStageFlags::TRANSFER,
					vk::DependencyFlags::default(),
					&[],
					&[dst_buffer_barrier],
					&[src_img_barrier, dst_img_barrier],
				);
			}

			// Copy image to buffer
			unsafe {
				let copy_region = vk::BufferImageCopy::builder()
					.buffer_row_length(0)
					.buffer_image_height(0)
					.image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
					.image_extent(vk::Extent3D {
						width: self.image.data.width,
						height: self.image.data.height,
						depth: 1,
					})
					.image_subresource(vk::ImageSubresourceLayers {
						aspect_mask: vk::ImageAspectFlags::COLOR,
						base_array_layer: 0,
						layer_count: 1,
						mip_level: 0,
						..Default::default()
					})
					.build();
				vk_setup.vk_device.cmd_copy_image_to_buffer(
					cmd_bud,
					self.image.image,
					vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
					self.cpu_buffer.buffer.handle,
					&[copy_region],
				)
			}

			let dst_img_barrier = VkSharedImage::gen_img_mem_barrier(
				self.image.image,
				vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
				self.image.image_layout,
				vk::AccessFlags::TRANSFER_READ,
				vk::AccessFlags::NONE,
			);
			let dst_buffer_barrier = VkCpuBuffer::gen_buffer_memory_barrier(
				self.cpu_buffer.buffer.handle,
				vk::AccessFlags::TRANSFER_WRITE,
				vk::AccessFlags::HOST_WRITE,
				self.cpu_buffer.buffer_size,
			);
			unsafe {
				vk_setup.vk_device.cmd_pipeline_barrier(
					cmd_bud,
					vk::PipelineStageFlags::TRANSFER,
					vk::PipelineStageFlags::HOST,
					vk::DependencyFlags::default(),
					&[],
					&[dst_buffer_barrier],
					&[dst_img_barrier],
				);
			}

			let dst_buffer_barrier = VkCpuBuffer::gen_buffer_memory_barrier(
				self.cpu_buffer.buffer.handle,
				vk::AccessFlags::HOST_WRITE,
				vk::AccessFlags::NONE,
				self.cpu_buffer.buffer_size,
			);
			unsafe {
				vk_setup.vk_device.cmd_pipeline_barrier(
					cmd_bud,
					vk::PipelineStageFlags::HOST,
					vk::PipelineStageFlags::BOTTOM_OF_PIPE,
					vk::DependencyFlags::default(),
					&[],
					&[dst_buffer_barrier],
					&[],
				);
			}

			Ok(())
		};

		vk_setup.immediate_submit_with_fence(
			vk_setup.vk_command_buffer,
			recv_image_cmd_fcn,
			&[],
			&[],
			fence,
		)?;
		self.cpu_buffer.sync_memory_to_cpu(vk_setup)?;

		Ok(())
	}

	fn recv_image_blit(
		&self,
		vk_setup: &VkSetup,
		src_image: &vk::Image,
		orig_src_image_layout: vk::ImageLayout,
		target_src_image_layout: vk::ImageLayout,
		fence: vk::Fence,
	) -> Result<(), vk::Result> {
		let src_image_extent = [
			vk::Offset3D { x: 0, y: 0, z: 0 },
			vk::Offset3D {
				x: self.image.data.width as i32,
				y: self.image.data.height as i32,
				z: 1,
			},
		];

		self.recv_image_blit_with_extents(
			vk_setup,
			src_image,
			orig_src_image_layout,
			target_src_image_layout,
			&src_image_extent,
			fence,
		)
	}
}

#[cfg(test)]
mod tests {
	use std::ffi::CStr;
	use std::slice;

	use ash::vk;

	use crate::vk_setup::VkSetup;
	use crate::vk_shared_image::ImageBlit;

	use super::VkCpuSharedImage;

	fn _init_vk_setup() -> VkSetup {
		VkSetup::new(CStr::from_bytes_with_nul(b"VkSetup\0").unwrap(), None).unwrap()
	}

	#[test]
	fn vk_cpu_shared_image_new() {
		let vk_setup = _init_vk_setup();
		let vk_cpu_shared_image =
			VkCpuSharedImage::new(&vk_setup, 1, 1, vk::Format::R8G8B8A8_UNORM, 0)
				.expect("Unable to create VkCpuSharedImage");

		vk_cpu_shared_image.destroy(&vk_setup);
	}

	#[test]
	fn vk_cpu_shared_image_copy() {
		let vk_setup = _init_vk_setup();
		let vk_cpu_shared_image_in =
			VkCpuSharedImage::new(&vk_setup, 1, 1, vk::Format::R8G8B8A8_UNORM, 0)
				.expect("Unable to create vk_cpu_shared_image_in");

		let vk_cpu_shared_image_out =
			VkCpuSharedImage::new(&vk_setup, 1, 1, vk::Format::R8G8B8A8_UNORM, 0)
				.expect("Unable to create vk_cpu_shared_image_out");

		let ram_in = unsafe {
			slice::from_raw_parts_mut(
				vk_cpu_shared_image_in.cpu_buffer.ram_memory as *mut u8,
				vk_cpu_shared_image_in.cpu_buffer.buffer_size as usize,
			)
		};
		let ram_out = unsafe {
			slice::from_raw_parts_mut(
				vk_cpu_shared_image_out.cpu_buffer.ram_memory as *mut u8,
				vk_cpu_shared_image_out.cpu_buffer.buffer_size as usize,
			)
		};

		let fence = vk_setup.create_fence(None).unwrap();

		let test_val = 5;
		let fake_val = 31;

		// Test recv_image_blit
		ram_in[0] = test_val;
		ram_out[0] = fake_val;

		vk_cpu_shared_image_in
			.cpu_buffer
			.write_image_from_cpu(
				&vk_setup,
				vk_cpu_shared_image_in.image.image,
				vk_cpu_shared_image_in.image.image_layout,
				vk_cpu_shared_image_in.image.data.width,
				vk_cpu_shared_image_in.image.data.height,
			)
			.unwrap();

		vk_cpu_shared_image_out
			.recv_image_blit(
				&vk_setup,
				&vk_cpu_shared_image_in.image.image,
				vk_cpu_shared_image_in.image.image_layout,
				vk_cpu_shared_image_in.image.image_layout,
				fence.handle,
			)
			.expect("Unable to recv_image_blit");

		assert_eq!(ram_in[0], test_val);
		assert_eq!(ram_out[0], test_val);

		// Test send_image_blit
		ram_out[0] = fake_val;

		vk_cpu_shared_image_in
			.send_image_blit(
				&vk_setup,
				&vk_cpu_shared_image_out.image.image,
				vk_cpu_shared_image_out.image.image_layout,
				vk_cpu_shared_image_out.image.image_layout,
				fence.handle,
			)
			.expect("Unable to send_image_blit");

		vk_cpu_shared_image_out
			.cpu_buffer
			.read_image_to_cpu(
				&vk_setup,
				vk_cpu_shared_image_out.image.image,
				vk_cpu_shared_image_out.image.image_layout,
				vk_cpu_shared_image_out.image.data.width,
				vk_cpu_shared_image_out.image.data.height,
			)
			.unwrap();

		assert_eq!(ram_in[0], test_val);
		assert_eq!(ram_out[0], test_val);

		vk_setup.destroy_fence(fence);

		vk_cpu_shared_image_out.destroy(&vk_setup);
		vk_cpu_shared_image_in.destroy(&vk_setup);
	}
}
