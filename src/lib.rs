use std::{
    alloc::Layout,
    cell::RefCell,
    collections::{BinaryHeap, HashMap, HashSet},
    ffi::CString,
    marker::PhantomData,
    sync::{Arc, Mutex, Weak},
};

use ash::{
    Entry,
    prelude::VkResult,
    vk::{
        self, BufferCreateInfo, BufferDeviceAddressInfo, BufferUsageFlags, DeviceCreateInfo,
        DeviceQueueCreateInfo, PhysicalDeviceFeatures2, PhysicalDeviceType, QueueFlags,
        SharingMode,
    },
};
use gpu_allocator::{
    MemoryLocation,
    vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator, AllocatorCreateDesc},
};
use raw_window_handle::{DisplayHandle, WindowHandle};
use thread_local::ThreadLocal;

pub struct GPU {
    entry: ash::Entry,
    device: ash::Device,
    instance: ash::Instance,
    physical_device: vk::PhysicalDevice,
    shader_object_device: ash::ext::shader_object::Device,
    swapchain_device: ash::khr::swapchain::Device,
    surface_loader: ash::khr::surface::Instance,
    queues: [vk::Queue; 3],
    queue_families: [u32; 3],
    command_buffer_semaphores: [vk::Semaphore; 3],
    command_buffer_counters: Mutex<[u64; 3]>,
    allocator: Option<Mutex<Allocator>>,
    thread_local: ThreadLocal<RefCell<GPUThreadLocalStorage>>,
    common_pipeline_graphics_layout: vk::PipelineLayout,
    common_pipeline_compute_layout: vk::PipelineLayout,
    descriptor_layouts: Vec<vk::DescriptorSetLayout>,
    settings: GPUSettings,
    weak: Weak<GPU>,
}
impl GPU {
    pub fn init(settings: GPUSettings) -> VkResult<Arc<GPU>> {
        unsafe {
            let entry = Entry::linked();
            let app_info = vk::ApplicationInfo {
                api_version: vk::API_VERSION_1_3,
                ..Default::default()
            };
            let mut enabled_layer_names = Vec::new();
            if settings.validate {
                enabled_layer_names.push("VK_LAYER_KHRONOS_validation\0".as_ptr() as *const i8);
            }
            let create_info = vk::InstanceCreateInfo {
                p_application_info: &app_info,
                ..Default::default()
            }
            .enabled_extension_names(&settings.instance_extensions)
            .enabled_layer_names(&enabled_layer_names);
            let instance = entry.create_instance(&create_info, None)?;

            let physical_device = instance
                .enumerate_physical_devices()
                .unwrap()
                .into_iter()
                .max_by_key(|physical_device| {
                    let device_type = instance
                        .get_physical_device_properties(*physical_device)
                        .device_type;
                    //let memory_properties = instance.get_physical_device_memory_properties(*physical_device);
                    //todo: better scoring
                    if device_type == PhysicalDeviceType::DISCRETE_GPU {
                        4
                    } else if device_type == PhysicalDeviceType::VIRTUAL_GPU {
                        3
                    } else if device_type == PhysicalDeviceType::INTEGRATED_GPU {
                        2
                    } else if device_type == PhysicalDeviceType::CPU {
                        1
                    } else {
                        0
                    }
                })
                .ok_or(vk::Result::ERROR_UNKNOWN)?;

            let queue_families =
                instance.get_physical_device_queue_family_properties(physical_device);

            let queue_families: [u32; 3] = std::array::from_fn(|queue| {
                let queue = match queue {
                    0 => QueueKind::Main,
                    1 => QueueKind::Compute,
                    2 => QueueKind::Transfer,
                    _ => unreachable!(),
                };
                queue_families
                    .iter()
                    .enumerate()
                    .filter_map(|(family, family_props)| {
                        if family_props.queue_count == 0 {
                            return None;
                        }
                        let prop_graphics = family_props.queue_flags.contains(QueueFlags::GRAPHICS);
                        let prop_compute = family_props.queue_flags.contains(QueueFlags::COMPUTE);
                        let prop_transfer = family_props.queue_flags.contains(QueueFlags::TRANSFER);
                        let mut unfitness = 0u32;
                        match (queue.is_graphics(), prop_graphics) {
                            (true, false) => return None,
                            (false, true) => unfitness += 1,
                            _ => {}
                        }
                        match (queue.is_compute(), prop_compute) {
                            (true, false) => return None,
                            (false, true) => unfitness += 1,
                            _ => {}
                        }
                        match (queue.is_transfer(), prop_transfer) {
                            (true, false) => return None,
                            (false, true) => unfitness += 1,
                            _ => {}
                        }
                        Some((family, unfitness))
                    })
                    .min_by_key(|(_, unfitness)| *unfitness)
                    .unwrap()
                    .0 as u32
            });
            let mut queue_creation_infos = Vec::new();
            let mut visited_families = HashSet::new();
            for queue_family in queue_families {
                if !visited_families.insert(queue_family) {
                    continue;
                }
                let mut device_queue_create_info = DeviceQueueCreateInfo::default();
                device_queue_create_info.queue_count = 1;
                device_queue_create_info.queue_family_index = queue_family as u32;
                device_queue_create_info.p_queue_priorities = &[1.0f32] as *const f32;
                queue_creation_infos.push(device_queue_create_info);
            }

            let mut enabled_extension_device_names = vec![];
            enabled_extension_device_names.push(ash::khr::swapchain::NAME.as_ptr());
            enabled_extension_device_names.push("VK_EXT_shader_object\0".as_ptr() as *const i8);
            let mut features1 = vk::PhysicalDeviceVulkan12Features::default()
                .runtime_descriptor_array(true)
                .shader_sampled_image_array_non_uniform_indexing(true)
                .shader_storage_image_array_non_uniform_indexing(true)
                .descriptor_binding_sampled_image_update_after_bind(true)
                .descriptor_binding_storage_image_update_after_bind(true)
                .descriptor_binding_partially_bound(true)
                .timeline_semaphore(true)
                .buffer_device_address(true)
                .draw_indirect_count(true)
                .scalar_block_layout(true)
                .shader_int8(true)
                .storage_buffer8_bit_access(true);
            let mut features2 =
                vk::PhysicalDeviceVulkan11Features::default().shader_draw_parameters(true);
            let mut features3 = vk::PhysicalDeviceVulkan13Features::default()
                .dynamic_rendering(true)
                .synchronization2(true)
                .maintenance4(true);
            let mut features4 =
                vk::PhysicalDeviceShaderObjectFeaturesEXT::default().shader_object(true);
            let mut features5 =
                vk::PhysicalDeviceDepthClipEnableFeaturesEXT::default().depth_clip_enable(true);
            let mut features6 = PhysicalDeviceFeatures2::default().features(
                vk::PhysicalDeviceFeatures::default()
                    .shader_int64(true)
                    .shader_int16(true)
                    .vertex_pipeline_stores_and_atomics(true)
                    .sampler_anisotropy(true)
                    .image_cube_array(true),
            );
            let device_create_info = DeviceCreateInfo::default()
                .queue_create_infos(&queue_creation_infos)
                .enabled_extension_names(&enabled_extension_device_names)
                .push_next(&mut features1)
                .push_next(&mut features2)
                .push_next(&mut features3)
                .push_next(&mut features4)
                .push_next(&mut features5)
                .push_next(&mut features6);

            let device = instance.create_device(physical_device, &device_create_info, None)?;
            let queues = queue_families.map(|family| device.get_device_queue(family as u32, 0));

            let shader_object_device = ash::ext::shader_object::Device::new(&instance, &device);
            let swapchain_device = ash::khr::swapchain::Device::new(&instance, &device);
            let surface_loader = ash::khr::surface::Instance::new(&entry, &instance);

            let allocator = Allocator::new(&AllocatorCreateDesc {
                instance: instance.clone(),
                device: device.clone(),
                physical_device,
                debug_settings: Default::default(),
                buffer_device_address: true,
                allocation_sizes: Default::default(),
            })
            .unwrap();

            let mut descriptor_layouts = Vec::new();
            {
                let bind_flags = vk::DescriptorBindingFlags::UPDATE_AFTER_BIND
                    | vk::DescriptorBindingFlags::PARTIALLY_BOUND;
                let mut flags_info = vk::DescriptorSetLayoutBindingFlagsCreateInfo::default()
                    .binding_flags(std::slice::from_ref(&bind_flags));
                {
                    let binding = vk::DescriptorSetLayoutBinding::default()
                        .binding(0)
                        .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                        .descriptor_count(settings.desc_heap_max_textures as u32)
                        .stage_flags(
                            vk::ShaderStageFlags::VERTEX
                                | vk::ShaderStageFlags::FRAGMENT
                                | vk::ShaderStageFlags::COMPUTE,
                        );
                    let create_info = vk::DescriptorSetLayoutCreateInfo::default()
                        .push_next(&mut flags_info)
                        .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
                        .bindings(std::slice::from_ref(&binding));
                    descriptor_layouts.push(
                        device
                            .create_descriptor_set_layout(&create_info, None)
                            .unwrap(),
                    );
                }
                {
                    let binding = vk::DescriptorSetLayoutBinding::default()
                        .binding(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .descriptor_count(settings.desc_heap_max_textures as u32)
                        .stage_flags(
                            vk::ShaderStageFlags::VERTEX
                                | vk::ShaderStageFlags::FRAGMENT
                                | vk::ShaderStageFlags::COMPUTE,
                        );
                    let create_info = vk::DescriptorSetLayoutCreateInfo::default()
                        .push_next(&mut flags_info)
                        .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
                        .bindings(std::slice::from_ref(&binding));
                    descriptor_layouts.push(
                        device
                            .create_descriptor_set_layout(&create_info, None)
                            .unwrap(),
                    );
                }
                {
                    let binding = vk::DescriptorSetLayoutBinding::default()
                        .binding(0)
                        .descriptor_type(vk::DescriptorType::SAMPLER)
                        .descriptor_count(settings.desc_heap_max_samplers as u32)
                        .stage_flags(
                            vk::ShaderStageFlags::VERTEX
                                | vk::ShaderStageFlags::FRAGMENT
                                | vk::ShaderStageFlags::COMPUTE,
                        );
                    let create_info = vk::DescriptorSetLayoutCreateInfo::default()
                        .push_next(&mut flags_info)
                        .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
                        .bindings(std::slice::from_ref(&binding));
                    descriptor_layouts.push(
                        device
                            .create_descriptor_set_layout(&create_info, None)
                            .unwrap(),
                    );
                }
            }

            let common_pipeline_graphics_layout = device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default()
                        .set_layouts(&descriptor_layouts)
                        .push_constant_ranges(&[vk::PushConstantRange {
                            offset: 0,
                            size: settings.max_push_constant_size as u32,
                            stage_flags: vk::ShaderStageFlags::VERTEX
                                | vk::ShaderStageFlags::FRAGMENT,
                        }]),
                    None,
                )
                .unwrap();
            let common_pipeline_compute_layout = device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default()
                        .set_layouts(&descriptor_layouts)
                        .push_constant_ranges(&[vk::PushConstantRange {
                            offset: 0,
                            size: settings.max_push_constant_size as u32,
                            stage_flags: vk::ShaderStageFlags::COMPUTE,
                        }]),
                    None,
                )
                .unwrap();

            Ok(Arc::new_cyclic(move |weak| GPU {
                shader_object_device,
                swapchain_device,
                surface_loader,
                queues,
                physical_device,
                queue_families,
                allocator: Some(Mutex::new(allocator)),
                thread_local: ThreadLocal::new(),
                command_buffer_counters: Mutex::new([0; 3]),
                command_buffer_semaphores: std::array::from_fn(|_| {
                    let mut semaphore_create_type_info = vk::SemaphoreTypeCreateInfo::default()
                        .initial_value(0)
                        .semaphore_type(vk::SemaphoreType::TIMELINE);
                    device
                        .create_semaphore(
                            &vk::SemaphoreCreateInfo::default()
                                .push_next(&mut semaphore_create_type_info),
                            None,
                        )
                        .unwrap()
                }),
                common_pipeline_graphics_layout,
                common_pipeline_compute_layout,
                descriptor_layouts,
                weak: weak.clone(),
                device,
                instance,
                entry,
                settings,
            }))
        }
    }
    pub fn queue_wait_idle(&self, queue: QueueKind) {
        let _lock = self.command_buffer_counters.lock().unwrap();
        unsafe {
            self.device
                .queue_wait_idle(self.queues[queue as usize])
                .unwrap()
        };
    }
    pub fn allocate_buffer(&self, layout: Layout, memory: MemoryKind) -> GPUBuffer {
        let location = match memory {
            MemoryKind::Default => MemoryLocation::CpuToGpu,
            MemoryKind::GPU => MemoryLocation::GpuOnly,
            MemoryKind::Readback => MemoryLocation::GpuToCpu,
        };
        let usage = BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | BufferUsageFlags::STORAGE_BUFFER
            | BufferUsageFlags::INDEX_BUFFER
            | BufferUsageFlags::TRANSFER_SRC
            | BufferUsageFlags::TRANSFER_DST
            | BufferUsageFlags::INDIRECT_BUFFER;
        let buffer = unsafe {
            let buffer_create_info = BufferCreateInfo::default()
                .size(layout.size() as u64)
                .usage(usage)
                .sharing_mode(SharingMode::EXCLUSIVE);
            self.device
                .create_buffer(&buffer_create_info, None)
                .unwrap()
        };
        let mut requirements = unsafe { self.device.get_buffer_memory_requirements(buffer) };
        requirements.alignment = requirements.alignment.max(layout.align() as u64);

        let allocation = self
            .allocator
            .as_ref()
            .unwrap()
            .lock()
            .unwrap()
            .allocate(&AllocationCreateDesc {
                name: "",
                requirements,
                location,
                linear: true,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })
            .unwrap();

        unsafe {
            self.device
                .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
                .unwrap()
        };

        let gpu_pointer = unsafe {
            self.device
                .get_buffer_device_address(&BufferDeviceAddressInfo::default().buffer(buffer))
        };
        GPUBuffer {
            allocation: Some(allocation),
            gpu_pointer,
            buffer,
            gpu: self.weak.upgrade().unwrap(),
        }
    }
    pub fn allocate_buffer_with<T: bytemuck::Pod>(&self, data: &[T]) -> GPUBuffer {
        let mut buffer =
            self.allocate_buffer(Layout::array::<T>(data.len()).unwrap(), MemoryKind::Default);
        buffer.upload(0, data);
        buffer
    }
    pub fn create_semaphore(&self, value: u64) -> GPUSemaphore {
        GPUSemaphore::create_raw(&self.device, self.weak.upgrade().unwrap(), value)
    }
    pub fn create_shaders_graphic_spirv(
        &self,
        code: &[u8],
        entry_point_vertex: &str,
        entry_point_fragment: &str,
        specializations: &HashMap<u32, u32>,
    ) -> VkResult<GraphicsShaders> {
        Ok(GraphicsShaders {
            vertex: self.create_shader_spirv(
                code,
                entry_point_vertex,
                ShaderKind::Vertex,
                specializations,
            )?,
            fragment: self.create_shader_spirv(
                code,
                entry_point_fragment,
                ShaderKind::Fragment,
                specializations,
            )?,
        })
    }
    pub fn create_shader_spirv(
        &self,
        code: &[u8],
        entry_point: &str,
        kind: ShaderKind,
        specializations: &HashMap<u32, u32>,
    ) -> VkResult<GPUShader> {
        let mut aligned_code = vec![0u32; code.len().div_ceil(4)];
        let aligned_code =
            &mut bytemuck::cast_slice_mut::<_, u8>(aligned_code.as_mut_slice())[..code.len()];
        aligned_code.copy_from_slice(code);
        let (stage, next_stage) = match kind {
            ShaderKind::Vertex => (vk::ShaderStageFlags::VERTEX, vk::ShaderStageFlags::FRAGMENT),
            ShaderKind::Fragment => (
                vk::ShaderStageFlags::FRAGMENT,
                vk::ShaderStageFlags::empty(),
            ),
            ShaderKind::Compute => (vk::ShaderStageFlags::COMPUTE, vk::ShaderStageFlags::empty()),
        };
        let mut push_constants = Vec::new();
        push_constants.push(match kind {
            ShaderKind::Vertex | ShaderKind::Fragment => vk::PushConstantRange {
                offset: 0,
                size: self.settings.max_push_constant_size as u32,
                stage_flags: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
            },
            ShaderKind::Compute { .. } => vk::PushConstantRange {
                offset: 0,
                size: self.settings.max_push_constant_size as u32,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
            },
        });
        //todo: compute specializations
        let mut specialization_data = Vec::new();
        let mut specialization_map = Vec::new();
        for (constant_id, data) in specializations {
            specialization_map.push(vk::SpecializationMapEntry {
                constant_id: *constant_id,
                offset: (specialization_data.len() * 4) as u32,
                size: 4,
            });
            specialization_data.push(*data);
        }
        let specialization_info = vk::SpecializationInfo::default()
            .map_entries(&specialization_map)
            .data(bytemuck::cast_slice(&specialization_data));
        let cstr_name = CString::new(entry_point).unwrap();
        let shader_create_info = vk::ShaderCreateInfoEXT::default()
            .name(&cstr_name)
            .code(aligned_code)
            .code_type(vk::ShaderCodeTypeEXT::SPIRV)
            .stage(stage)
            .next_stage(next_stage)
            .push_constant_ranges(&push_constants)
            .specialization_info(&specialization_info)
            .set_layouts(&self.descriptor_layouts);
        let shader = unsafe {
            self.shader_object_device
                .create_shaders(&[shader_create_info], None)
                .unwrap()[0] //todo: error handling
        };
        Ok(GPUShader {
            shader,
            kind,
            gpu: self.weak.upgrade().unwrap(),
        })
    }
    pub fn begin_commands(&self, queue: QueueKind) -> GPUCommandBuffer {
        let mut local = self
            .thread_local
            .get_or(|| {
                let vk_pools = std::array::from_fn(|i| {
                    let info = vk::CommandPoolCreateInfo::default()
                        .flags(
                            vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER
                                | vk::CommandPoolCreateFlags::TRANSIENT,
                        )
                        .queue_family_index(self.queue_families[i]);
                    unsafe { self.device.create_command_pool(&info, None).unwrap() }
                });
                RefCell::new(GPUThreadLocalStorage {
                    pools: Default::default(),
                    vk_pools,
                })
            })
            .borrow_mut();
        let command_begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        let current_semaphore_value = unsafe {
            self.device
                .get_semaphore_counter_value(self.command_buffer_semaphores[queue as usize])
                .unwrap()
        };
        if let Some(pooled) = local.pools[queue as usize].peek() {
            if pooled.timestamp <= current_semaphore_value {
                let pooled = local.pools[queue as usize].pop().unwrap();
                unsafe {
                    self.device
                        .begin_command_buffer(pooled.command_buffer, &command_begin_info)
                        .unwrap();
                }
                return GPUCommandBuffer {
                    command_buffer: pooled.command_buffer,
                    queue,
                    gpu: self.weak.upgrade().unwrap(),
                    signalling: Vec::new(),
                    waiting: Vec::new(),
                    _marker: PhantomData,
                    has_compute_shader: false,
                };
            }
        }
        let command_buffer = unsafe {
            self.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(local.vk_pools[queue as usize])
                    .command_buffer_count(1)
                    .level(vk::CommandBufferLevel::PRIMARY),
            )
        }
        .unwrap()[0];
        unsafe {
            self.device
                .begin_command_buffer(command_buffer, &command_begin_info)
                .unwrap();
        }
        GPUCommandBuffer {
            command_buffer,
            queue,
            gpu: self.weak.upgrade().unwrap(),
            signalling: Vec::new(),
            waiting: Vec::new(),
            _marker: PhantomData,
            has_compute_shader: false,
        }
    }
    pub fn create_texture(
        &self,
        desc: TextureDesc,
        queue: QueueKind,
        semaphore: &GPUSemaphore,
        wait_value: u64,
    ) -> GPUTexture {
        let mut usage = vk::ImageUsageFlags::TRANSFER_DST;
        if desc.usage.0 & TextureUsage::TRANSFER_SRC.0 != 0 {
            usage |= vk::ImageUsageFlags::TRANSFER_SRC;
        }
        if desc.usage.0 & TextureUsage::SAMPLED.0 != 0 {
            usage |= vk::ImageUsageFlags::SAMPLED;
        }
        if desc.usage.0 & TextureUsage::STORAGE.0 != 0 {
            usage |= vk::ImageUsageFlags::STORAGE;
        }
        if desc.usage.0 & TextureUsage::COLOR_ATTACHMENT.0 != 0 {
            usage |= vk::ImageUsageFlags::COLOR_ATTACHMENT;
        }
        if desc.usage.0 & TextureUsage::DEPTH_STENCIL_ATTACHMENT.0 != 0 {
            usage |= vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT;
        }
        let image = unsafe {
            self.device
                .create_image(
                    &&vk::ImageCreateInfo::default()
                        .array_layers(desc.layer_count)
                        .initial_layout(vk::ImageLayout::UNDEFINED)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE)
                        .samples(to_vk_sample_count(desc.sample_count))
                        .format(desc.format.to_vk())
                        .image_type(match desc.texture_type {
                            TextureType::D1 | TextureType::D1Array => vk::ImageType::TYPE_1D,
                            TextureType::D2
                            | TextureType::D2Array
                            | TextureType::Cube
                            | TextureType::CubeArray => vk::ImageType::TYPE_2D,
                            TextureType::D3 => vk::ImageType::TYPE_3D,
                        })
                        .usage(usage)
                        .extent(vk::Extent3D {
                            width: desc.dimensions[0],
                            height: desc.dimensions[1],
                            depth: desc.dimensions[2],
                        })
                        .mip_levels(desc.mip_count)
                        .flags(match desc.texture_type {
                            TextureType::Cube | TextureType::CubeArray => {
                                vk::ImageCreateFlags::CUBE_COMPATIBLE
                            }
                            _ => vk::ImageCreateFlags::empty(),
                        }),
                    None,
                )
                .unwrap()
        };
        let image_requirements = unsafe { self.device.get_image_memory_requirements(image) };
        let allocation = self
            .allocator
            .as_ref()
            .unwrap()
            .lock()
            .unwrap()
            .allocate(&AllocationCreateDesc {
                name: "",
                requirements: image_requirements,
                location: MemoryLocation::GpuOnly,
                linear: false,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })
            .unwrap();
        unsafe {
            self.device
                .bind_image_memory(image, allocation.memory(), allocation.offset())
        }
        .unwrap();
        let mut command_buffer = self.begin_commands(queue);
        let image_barrier = vk::ImageMemoryBarrier2::default()
            .image(image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: TextureFormat::aspect_flags(desc.format.to_vk()),
                layer_count: desc.layer_count,
                level_count: desc.mip_count,
                base_array_layer: 0,
                base_mip_level: 0,
            })
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE);
        unsafe {
            self.device.cmd_pipeline_barrier2(
                command_buffer.command_buffer,
                &vk::DependencyInfo::default().image_memory_barriers(&[image_barrier]),
            )
        };
        command_buffer.add_signal(semaphore, wait_value);
        command_buffer.submit();
        GPUTexture {
            texture_type: desc.texture_type,
            dimensions: desc.dimensions,
            format: desc.format.to_vk(),
            mip_count: desc.mip_count,
            layer_count: desc.layer_count,
            sample_count: desc.sample_count,
            image,
            allocation: Some(allocation),
            view_cache: Default::default(),
            gpu: self.weak.upgrade().unwrap(),
            is_swapchain: false,
        }
    }
    pub fn create_swapchain(
        &self,
        display_handle: DisplayHandle,
        window_handle: WindowHandle,
    ) -> GPUSwapchain {
        unsafe {
            let surface = ash_window::create_surface(
                &self.entry,
                &self.instance,
                display_handle.as_raw(),
                window_handle.as_raw(),
                None,
            )
            .unwrap();
            let mut textures = Vec::new();
            let swapchain = GPUSwapchain::make_swapchain(self, surface, &mut textures);
            GPUSwapchain {
                present_semaphores: std::iter::repeat_with(|| {
                    GPUSemaphore::create_binary(&self.device, self.weak.upgrade().unwrap())
                })
                .take(textures.len())
                .collect(),
                swapchain,
                textures,
                next_swap_image: 0,
                gpu: self.weak.upgrade().unwrap(),
                surface,
            }
        }
    }
    pub fn allocate_arena(&self, block_size: usize) -> GPUArena {
        let memory = MemoryKind::Default;
        GPUArena {
            gpu: self.weak.upgrade().unwrap(),
            memory,
            block_size,
            blocks: vec![
                self.allocate_buffer(Layout::from_size_align(block_size, 16).unwrap(), memory),
            ],
            block_offset: 0,
            block_index: 0,
        }
    }
    pub fn create_descriptor_heap(&self) -> GPUDescriptorHeap {
        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::SAMPLED_IMAGE,
                descriptor_count: self.settings.desc_heap_max_textures as u32,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: self.settings.desc_heap_max_textures as u32,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::SAMPLER,
                descriptor_count: self.settings.desc_heap_max_samplers as u32,
            },
        ];
        let pool_create_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(3)
            .flags(
                vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET
                    | vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND,
            );
        let pool = unsafe {
            self.device
                .create_descriptor_pool(&pool_create_info, None)
                .unwrap()
        };
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(&self.descriptor_layouts);
        let desc_sets = unsafe { self.device.allocate_descriptor_sets(&alloc_info).unwrap() };
        GPUDescriptorHeap {
            gpu: self.weak.upgrade().unwrap(),
            pool,
            textures: desc_sets[0],
            textures_rw: desc_sets[1],
            samplers: desc_sets[2],
            sampler_cache: HashMap::new(),
        }
    }
}
pub struct TextureDesc {
    pub texture_type: TextureType,
    pub dimensions: [u32; 3],
    pub format: TextureFormat,
    pub mip_count: u32,
    pub layer_count: u32,
    pub sample_count: u32,
    pub usage: TextureUsage,
}
impl Default for TextureDesc {
    fn default() -> Self {
        TextureDesc {
            texture_type: TextureType::D2,
            dimensions: [1; 3],
            format: TextureFormat::RGBA8Unorm,
            mip_count: 1,
            layer_count: 1,
            sample_count: 1,
            usage: TextureUsage::EMPTY,
        }
    }
}
impl Drop for GPU {
    fn drop(&mut self) {
        self.allocator = None;
        for thread_local in self.thread_local.iter_mut() {
            let thread_local = thread_local.get_mut();
            for pool in thread_local.vk_pools {
                unsafe { self.device.destroy_command_pool(pool, None) };
            }
        }
        for semaphore in self.command_buffer_semaphores {
            unsafe {
                self.device.destroy_semaphore(semaphore, None);
            }
        }
        for descriptor_set in self.descriptor_layouts.drain(..) {
            unsafe {
                self.device
                    .destroy_descriptor_set_layout(descriptor_set, None)
            };
        }
        unsafe {
            self.device
                .destroy_pipeline_layout(self.common_pipeline_graphics_layout, None);
            self.device
                .destroy_pipeline_layout(self.common_pipeline_compute_layout, None);
        }
        unsafe {
            self.device.destroy_device(None);
        }
    }
}
pub struct PooledCommandBuffer {
    command_buffer: vk::CommandBuffer,
    timestamp: u64,
}
impl PartialOrd for PooledCommandBuffer {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(other.timestamp.cmp(&self.timestamp))
    }
}
impl Ord for PooledCommandBuffer {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}
impl PartialEq for PooledCommandBuffer {
    fn eq(&self, _: &Self) -> bool {
        false
    }
}
impl Eq for PooledCommandBuffer {}
#[derive(Default)]
pub struct GPUThreadLocalStorage {
    pools: [BinaryHeap<PooledCommandBuffer>; 3],
    vk_pools: [vk::CommandPool; 3],
}
pub struct GPUCommandBuffer {
    command_buffer: vk::CommandBuffer,
    queue: QueueKind,
    gpu: Arc<GPU>,
    waiting: Vec<(vk::Semaphore, u64)>,
    signalling: Vec<(vk::Semaphore, u64)>,
    _marker: PhantomData<*mut ()>, //not send
    has_compute_shader: bool,
}
impl GPUCommandBuffer {
    pub fn add_wait(&mut self, semaphore: &GPUSemaphore, value: u64) {
        self.waiting.push((semaphore.semaphore, value));
    }
    pub fn add_signal(&mut self, semaphore: &GPUSemaphore, value: u64) {
        self.signalling.push((semaphore.semaphore, value));
    }
    pub fn barrier(&mut self, before: GPUStage, after: GPUStage, hazards: &[GPUHazard]) {
        let mut src_access = vk::AccessFlags::empty();
        let mut dst_access = vk::AccessFlags::empty();
        if hazards.is_empty() {
            src_access |= vk::AccessFlags::MEMORY_WRITE;
            dst_access |= vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE;
        }
        for hazard in hazards {
            match hazard {
                GPUHazard::DrawArguments => {
                    src_access |= vk::AccessFlags::SHADER_WRITE;
                    dst_access |= vk::AccessFlags::INDIRECT_COMMAND_READ;
                }
                GPUHazard::Descriptors => {
                    src_access |= vk::AccessFlags::SHADER_WRITE;
                    dst_access |= vk::AccessFlags::SHADER_READ;
                }
                GPUHazard::DepthStencil => {
                    src_access |= vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE;
                    dst_access |= vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE
                        | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ;
                }
            }
            let barrier_info = vk::MemoryBarrier::default()
                .src_access_mask(src_access)
                .dst_access_mask(dst_access);
            unsafe {
                self.gpu.device.cmd_pipeline_barrier(
                    self.command_buffer,
                    before.to_vk(),
                    after.to_vk(),
                    vk::DependencyFlags::empty(),
                    &[barrier_info],
                    &[],
                    &[],
                );
            }
        }
    }
    pub fn begin_render<'a>(&'a mut self, mut desc: RenderPassDesc) -> GPURenderPass<'a> {
        let mut sample_count = 0;
        for color_attachment in desc.color_attachments {
            sample_count = sample_count.max(color_attachment.texture.sample_count);
        }
        if let Some(depth_attachment) = &desc.depth_attachment {
            sample_count = sample_count.max(depth_attachment.texture.sample_count);
        }
        if let Some(attachment) = desc.color_attachments.get(0) {
            if desc.render_area_size[0] == 0 {
                desc.render_area_size[0] = attachment.texture.dimensions[0];
            }
            if desc.render_area_size[1] == 0 {
                desc.render_area_size[1] = attachment.texture.dimensions[1];
            }
        }
        fn create_vk_attachment<'a, C: IntoClearValue>(
            attachment: &RenderPassAttachment<'a, C>,
        ) -> vk::RenderingAttachmentInfo<'static> {
            let mut info = vk::RenderingAttachmentInfo::default()
                .image_layout(vk::ImageLayout::GENERAL)
                .load_op(match attachment.load {
                    RenderPassLoadOp::Load => vk::AttachmentLoadOp::LOAD,
                    RenderPassLoadOp::Clear(_) => vk::AttachmentLoadOp::CLEAR,
                })
                .store_op(match attachment.store {
                    RenderPassStoreOp::Store => vk::AttachmentStoreOp::STORE,
                    RenderPassStoreOp::Discard => vk::AttachmentStoreOp::NONE,
                })
                .image_view(attachment.texture.get_view(&attachment.view));
            if let RenderPassLoadOp::Clear(color) = &attachment.load {
                info = info.clear_value(color.into_clear_value(attachment.texture.format));
            }
            info
        }
        unsafe {
            let color_attachments = desc
                .color_attachments
                .iter()
                .map(|desc| create_vk_attachment(desc))
                .collect::<Vec<_>>();
            let mut info = vk::RenderingInfo::default()
                .color_attachments(&color_attachments)
                .layer_count(desc.layer_count)
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D {
                        x: desc.render_area_offset[0],
                        y: desc.render_area_offset[1],
                    },
                    extent: vk::Extent2D {
                        width: desc.render_area_size[0],
                        height: desc.render_area_size[1],
                    },
                })
                .view_mask(desc.view_mask);
            let depth_attachment = desc
                .depth_attachment
                .map(|depth_attachment| create_vk_attachment(&depth_attachment));
            if let Some(depth_attachment) = &depth_attachment {
                info = info.depth_attachment(depth_attachment);
            }
            let stencil_attachment = desc
                .stencil_attachment
                .map(|stencil_attachment| create_vk_attachment(&stencil_attachment));
            if let Some(stencil_attachment) = &stencil_attachment {
                info = info.stencil_attachment(stencil_attachment);
            }
            self.gpu
                .device
                .cmd_begin_rendering(self.command_buffer, &info);
            self.gpu
                .device
                .cmd_set_stencil_test_enable(self.command_buffer, false);
            self.gpu.shader_object_device.cmd_set_color_blend_enable(
                self.command_buffer,
                0,
                std::iter::repeat_n(0, desc.color_attachments.len())
                    .collect::<Vec<_>>()
                    .as_slice(),
            );
            self.gpu.shader_object_device.cmd_set_color_write_mask(
                self.command_buffer,
                0,
                std::iter::repeat_n(vk::ColorComponentFlags::RGBA, desc.color_attachments.len())
                    .collect::<Vec<_>>()
                    .as_slice(),
            );
            self.gpu.shader_object_device.cmd_set_rasterization_samples(
                self.command_buffer,
                to_vk_sample_count(sample_count),
            );
            self.gpu.shader_object_device.cmd_set_primitive_topology(
                self.command_buffer,
                vk::PrimitiveTopology::TRIANGLE_LIST,
            );
            self.gpu
                .shader_object_device
                .cmd_set_polygon_mode(self.command_buffer, vk::PolygonMode::FILL);
            self.gpu
                .shader_object_device
                .cmd_set_cull_mode(self.command_buffer, vk::CullModeFlags::NONE);
            self.gpu
                .shader_object_device
                .cmd_set_depth_compare_op(self.command_buffer, vk::CompareOp::ALWAYS); //todo
            self.gpu
                .shader_object_device
                .cmd_set_depth_bias_enable(self.command_buffer, false);
            self.gpu
                .shader_object_device
                .cmd_set_depth_write_enable(self.command_buffer, false);
            self.gpu
                .shader_object_device
                .cmd_set_depth_test_enable(self.command_buffer, false);

            self.gpu.shader_object_device.cmd_set_viewport_with_count(
                self.command_buffer,
                &[vk::Viewport {
                    x: 0.,
                    y: 0.,
                    width: desc.render_area_size[0] as f32,
                    height: desc.render_area_size[1] as f32,
                    min_depth: 0.,
                    max_depth: 1.,
                }],
            );
            self.gpu.shader_object_device.cmd_set_scissor_with_count(
                self.command_buffer,
                &[vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: vk::Extent2D {
                        width: desc.render_area_size[0],
                        height: desc.render_area_size[1],
                    },
                }],
            );
            self.gpu
                .shader_object_device
                .cmd_set_rasterizer_discard_enable(self.command_buffer, false);
            self.gpu.shader_object_device.cmd_set_color_blend_equation(
                self.command_buffer,
                0,
                &[vk::ColorBlendEquationEXT {
                    src_color_blend_factor: vk::BlendFactor::SRC_ALPHA,
                    dst_alpha_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
                    color_blend_op: vk::BlendOp::ADD,
                    src_alpha_blend_factor: vk::BlendFactor::ONE,
                    dst_color_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
                    alpha_blend_op: vk::BlendOp::ADD,
                }],
            );
            self.gpu.shader_object_device.cmd_set_sample_mask(
                self.command_buffer,
                to_vk_sample_count(sample_count),
                &[255],
            );
            self.gpu
                .shader_object_device
                .cmd_set_alpha_to_coverage_enable(self.command_buffer, false);

            self.gpu
                .shader_object_device
                .cmd_set_depth_clip_enable(self.command_buffer, true);
            self.gpu
                .shader_object_device
                .cmd_set_vertex_input(self.command_buffer, &[], &[]);
            self.gpu
                .shader_object_device
                .cmd_set_primitive_restart_enable(self.command_buffer, false);
        }
        GPURenderPass {
            command_buffer: self,
            has_shaders_bound: false,
        }
    }
    pub fn set_compute_shader(&mut self, shader: &GPUShader) {
        match shader.kind {
            ShaderKind::Compute => {}
            _ => panic!("wrong type of shader bound"),
        }
        unsafe {
            self.gpu.shader_object_device.cmd_bind_shaders(
                self.command_buffer,
                &[vk::ShaderStageFlags::COMPUTE],
                &[shader.shader],
            );
        }
        self.has_compute_shader = true;
    }
    pub fn dispatch<D: bytemuck::Pod>(&mut self, data: D, groups: [u32; 3]) {
        assert!(
            std::mem::size_of::<D>() <= self.gpu.settings.max_push_constant_size,
            "compute data size greater than set size"
        );
        assert!(self.has_compute_shader);
        unsafe {
            self.gpu.device.cmd_push_constants(
                self.command_buffer,
                self.gpu.common_pipeline_compute_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(&data),
            );
            self.gpu
                .device
                .cmd_dispatch(self.command_buffer, groups[0], groups[1], groups[2]);
        }
    }
    pub fn copy(&mut self, src: GPUPointer, dst: GPUPointer) {
        assert_eq!(src.size, dst.size);
        let copy = vk::BufferCopy::default()
            .src_offset(src.offset)
            .dst_offset(dst.offset)
            .size(src.size);
        unsafe {
            self.gpu
                .device
                .cmd_copy_buffer(self.command_buffer, src.buffer, dst.buffer, &[copy]);
        }
    }
    pub fn copy_to_texture(
        &mut self,
        texture: &GPUTexture,
        texture_region: TextureRegion,
        src: GPUPointer,
    ) {
        let aspect_mask = TextureFormat::aspect_flags(texture.format);
        let is_compressed = match texture.format {
            vk::Format::BC1_RGBA_UNORM_BLOCK
            | vk::Format::BC3_SRGB_BLOCK
            | vk::Format::BC7_SRGB_BLOCK
            | vk::Format::ASTC_4X4_SRGB_BLOCK
            | vk::Format::ETC2_R8G8B8_UNORM_BLOCK
            | vk::Format::ETC2_R8G8B8A8_UNORM_BLOCK
            | vk::Format::EAC_R11_UNORM_BLOCK
            | vk::Format::EAC_R11G11_UNORM_BLOCK => true,
            _ => false,
        };
        let mip_width = 1.max(texture.dimensions[0] >> texture_region.mip_level);
        let mip_height = 1.max(texture.dimensions[1] >> texture_region.mip_level);
        let mip_depth = 1.max(texture.dimensions[2] >> texture_region.mip_level);
        let copy_extent: [u32; 3] = std::array::from_fn(|i| {
            if texture_region.size[i] != 0 {
                texture_region.size[i]
            } else {
                [mip_width, mip_height, mip_depth][i]
            }
        });
        let copy_info = vk::BufferImageCopy::default()
            .buffer_offset(src.offset)
            .buffer_row_length(if is_compressed { 0 } else { mip_width })
            .buffer_image_height(if is_compressed { 0 } else { mip_height })
            .image_subresource(vk::ImageSubresourceLayers {
                aspect_mask,
                base_array_layer: texture_region.base_layer,
                layer_count: texture_region.layer_count,
                mip_level: texture_region.mip_level,
            })
            .image_offset(vk::Offset3D {
                x: texture_region.offset[0],
                y: texture_region.offset[1],
                z: texture_region.offset[2],
            })
            .image_extent(vk::Extent3D {
                width: copy_extent[0],
                height: copy_extent[1],
                depth: copy_extent[2],
            });
        unsafe {
            self.gpu.device.cmd_copy_buffer_to_image(
                self.command_buffer,
                src.buffer,
                texture.image,
                vk::ImageLayout::GENERAL,
                std::slice::from_ref(&copy_info),
            )
        };
    }
    pub fn submit(self) {}
    pub fn bind_descriptor_heap(&self, heap: &GPUDescriptorHeap) {
        let descriptor_sets = [heap.textures, heap.textures_rw, heap.samplers];
        unsafe {
            self.gpu.device.cmd_bind_descriptor_sets(
                self.command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.gpu.common_pipeline_graphics_layout,
                0,
                &descriptor_sets,
                &[],
            );
            self.gpu.device.cmd_bind_descriptor_sets(
                self.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.gpu.common_pipeline_compute_layout,
                0,
                &descriptor_sets,
                &[],
            );
        };
    }
}
impl Drop for GPUCommandBuffer {
    fn drop(&mut self) {
        unsafe {
            self.gpu
                .device
                .end_command_buffer(self.command_buffer)
                .unwrap();
        }
        let mut counters = self.gpu.command_buffer_counters.lock().unwrap();
        let cnt = counters[self.queue as usize] + 1;
        counters[self.queue as usize] = cnt;

        let mut signal_semaphores = vec![self.gpu.command_buffer_semaphores[self.queue as usize]];
        let mut signal_counters = vec![cnt];

        let mut wait_semaphores = Vec::new();
        let mut wait_stages = Vec::new();
        let mut wait_counters = Vec::new();

        for (sem, cnt) in self.signalling.drain(..) {
            signal_semaphores.push(sem);
            signal_counters.push(cnt);
        }
        for (sem, cnt) in self.waiting.drain(..) {
            wait_semaphores.push(sem);
            wait_stages.push(vk::PipelineStageFlags::ALL_COMMANDS);
            wait_counters.push(cnt);
        }

        let mut timeline_submit_info = vk::TimelineSemaphoreSubmitInfo::default()
            .signal_semaphore_values(&signal_counters)
            .wait_semaphore_values(&wait_counters);
        let submit_info = vk::SubmitInfo::default()
            .command_buffers(std::slice::from_ref(&self.command_buffer))
            .signal_semaphores(&signal_semaphores)
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&wait_stages)
            .push_next(&mut timeline_submit_info);

        unsafe {
            self.gpu
                .device
                .queue_submit(
                    self.gpu.queues[self.queue as usize],
                    &[submit_info],
                    vk::Fence::null(),
                )
                .unwrap();
        }

        self.gpu.thread_local.get().unwrap().borrow_mut().pools[self.queue as usize].push(
            PooledCommandBuffer {
                command_buffer: self.command_buffer,
                timestamp: cnt,
            },
        );
    }
}
pub struct GPURenderPass<'a> {
    command_buffer: &'a mut GPUCommandBuffer,
    has_shaders_bound: bool,
}
impl GPURenderPass<'_> {
    pub fn set_shaders_individual(&mut self, vertex: &GPUShader, fragment: &GPUShader) {
        match (vertex.kind, fragment.kind) {
            (ShaderKind::Vertex, ShaderKind::Fragment) => {}
            _ => panic!("wrong type of shader bound"),
        }
        self.has_shaders_bound = true;
        unsafe {
            self.command_buffer
                .gpu
                .shader_object_device
                .cmd_bind_shaders(
                    self.command_buffer.command_buffer,
                    &[vk::ShaderStageFlags::VERTEX, vk::ShaderStageFlags::FRAGMENT],
                    &[vertex.shader, fragment.shader],
                )
        };
    }
    pub fn set_shaders(&mut self, shaders: &GraphicsShaders) {
        self.set_shaders_individual(&shaders.vertex, &shaders.fragment);
    }
    pub fn set_culling(&mut self, cull_mode: GPUCullMode) {
        unsafe {
            self.command_buffer
                .gpu
                .shader_object_device
                .cmd_set_cull_mode(
                    self.command_buffer.command_buffer,
                    match cull_mode {
                        GPUCullMode::KeepAll => vk::CullModeFlags::NONE,
                        GPUCullMode::KeepCW => vk::CullModeFlags::FRONT,
                        GPUCullMode::KeepCCW => vk::CullModeFlags::BACK,
                    },
                )
        };
    }
    pub fn set_topology(&mut self, topology: GPUTopology) {
        unsafe {
            self.command_buffer
                .gpu
                .shader_object_device
                .cmd_set_primitive_topology(
                    self.command_buffer.command_buffer,
                    match topology {
                        GPUTopology::TriangleList => vk::PrimitiveTopology::TRIANGLE_LIST,
                        GPUTopology::TriangleStrip => vk::PrimitiveTopology::TRIANGLE_STRIP,
                        GPUTopology::TriangleFan => vk::PrimitiveTopology::TRIANGLE_FAN,
                    },
                )
        };
    }
    pub fn set_depth_test(&mut self, depth_config: Option<GPUDepthTestConfig>) {
        unsafe {
            self.command_buffer.gpu.device.cmd_set_depth_test_enable(
                self.command_buffer.command_buffer,
                depth_config.is_some(),
            );
            if let Some(depth_config) = depth_config {
                self.command_buffer.gpu.device.cmd_set_depth_write_enable(
                    self.command_buffer.command_buffer,
                    depth_config.write,
                );
                self.command_buffer.gpu.device.cmd_set_depth_compare_op(
                    self.command_buffer.command_buffer,
                    match depth_config.compare {
                        GPUCompareOp::Always => vk::CompareOp::ALWAYS,
                        GPUCompareOp::Never => vk::CompareOp::NEVER,
                        GPUCompareOp::Less => vk::CompareOp::LESS,
                        GPUCompareOp::LessEqual => vk::CompareOp::LESS_OR_EQUAL,
                        GPUCompareOp::Greater => vk::CompareOp::GREATER,
                        GPUCompareOp::GreaterEqual => vk::CompareOp::GREATER_OR_EQUAL,
                        GPUCompareOp::Equal => vk::CompareOp::EQUAL,
                        GPUCompareOp::NotEqual => vk::CompareOp::NOT_EQUAL,
                    },
                );
            }
        }
    }
    pub fn draw<D: bytemuck::Pod>(
        &mut self,
        draw_data: D,
        count: u32,
        instance_count: u32,
        index_buffer: Option<(GPUPointer, IndexBufferFormat)>,
    ) {
        let gpu = &self.command_buffer.gpu;
        assert!(
            std::mem::size_of::<D>() <= gpu.settings.max_push_constant_size,
            "draw data size greater than set size"
        );
        assert!(self.has_shaders_bound);
        unsafe {
            gpu.device.cmd_push_constants(
                self.command_buffer.command_buffer,
                gpu.common_pipeline_graphics_layout,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                0,
                bytemuck::bytes_of(&draw_data),
            );
            match index_buffer {
                Some((index_buffer, index_format)) => {
                    gpu.device.cmd_bind_index_buffer(
                        self.command_buffer.command_buffer,
                        index_buffer.buffer,
                        index_buffer.offset,
                        match index_format {
                            IndexBufferFormat::U16 => vk::IndexType::UINT16,
                            IndexBufferFormat::U32 => vk::IndexType::UINT32,
                        },
                    );
                    gpu.device.cmd_draw_indexed(
                        self.command_buffer.command_buffer,
                        count,
                        instance_count,
                        0,
                        0,
                        0,
                    );
                }
                None => {
                    gpu.device.cmd_draw(
                        self.command_buffer.command_buffer,
                        count,
                        instance_count,
                        0,
                        0,
                    );
                }
            }
        }
    }
}
impl Drop for GPURenderPass<'_> {
    fn drop(&mut self) {
        unsafe {
            self.command_buffer
                .gpu
                .device
                .cmd_end_rendering(self.command_buffer.command_buffer);
        }
    }
}
pub struct RenderPassDesc<'a> {
    pub render_area_offset: [i32; 2],
    pub render_area_size: [u32; 2],
    pub layer_count: u32,
    pub view_mask: u32,
    pub color_attachments: &'a [RenderPassAttachment<'a, [f32; 4]>],
    pub depth_attachment: Option<RenderPassAttachment<'a, f32>>,
    pub stencil_attachment: Option<RenderPassAttachment<'a, u32>>,
}
impl Default for RenderPassDesc<'_> {
    fn default() -> Self {
        RenderPassDesc {
            render_area_offset: [0, 0],
            render_area_size: [0, 0],
            layer_count: 1,
            view_mask: 0,
            color_attachments: &[],
            depth_attachment: None,
            stencil_attachment: None,
        }
    }
}
pub struct RenderPassAttachment<'a, C> {
    pub texture: &'a GPUTexture,
    pub view: TextureViewDesc,
    pub load: RenderPassLoadOp<C>,
    pub store: RenderPassStoreOp,
    pub resolve: Option<(&'a GPUTexture, TextureViewDesc)>,
}
trait IntoClearValue: Copy {
    fn into_clear_value(self, format: vk::Format) -> vk::ClearValue;
}
impl IntoClearValue for [f32; 4] {
    fn into_clear_value(self, _format: vk::Format) -> vk::ClearValue {
        vk::ClearValue {
            color: vk::ClearColorValue { float32: self },
        }
    }
}
impl IntoClearValue for f32 {
    fn into_clear_value(self, _format: vk::Format) -> vk::ClearValue {
        vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue {
                depth: self,
                stencil: 0,
            },
        }
    }
}
impl IntoClearValue for u32 {
    fn into_clear_value(self, _format: vk::Format) -> vk::ClearValue {
        vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue {
                depth: 0.,
                stencil: self,
            },
        }
    }
}
pub enum RenderPassLoadOp<C> {
    Load,
    Clear(C),
}
pub enum RenderPassStoreOp {
    Store,
    Discard,
}
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct TextureViewDesc {
    pub base_mip: u32,
    pub mip_count: u8,
    pub base_layer: u16,
    pub layer_count: u16,
}
impl Default for TextureViewDesc {
    fn default() -> Self {
        TextureViewDesc {
            base_mip: 0,
            mip_count: 1,
            base_layer: 0,
            layer_count: 1,
        }
    }
}
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub enum TextureType {
    D1,
    D2,
    D3,
    D1Array,
    D2Array,
    Cube,
    CubeArray,
}
impl TextureType {
    fn to_view_vk(self) -> vk::ImageViewType {
        match self {
            TextureType::D1 => vk::ImageViewType::TYPE_1D,
            TextureType::D2 => vk::ImageViewType::TYPE_2D,
            TextureType::D3 => vk::ImageViewType::TYPE_3D,
            TextureType::Cube => vk::ImageViewType::CUBE,
            TextureType::D2Array => vk::ImageViewType::TYPE_2D_ARRAY,
            TextureType::D1Array => vk::ImageViewType::TYPE_1D_ARRAY,
            TextureType::CubeArray => vk::ImageViewType::CUBE_ARRAY,
        }
    }
}
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub enum TextureFormat {
    R8Unorm,
    RG8Unorm,
    RGBA8Unorm,
    ABGR8Unorm,
    BGRA8Unorm,
    R8SRGB,
    RG8SRGB,
    RGBA8SRGB,
    ABGR8SRGB,
    BGRA8SRGB,
    R16Unorm,
    RG16Unorm,
    RGBA16Unorm,
    D16Unorm,
    D16UnormS8Uint,
    D24UnormPack32,
    D24UnormS8Uint,
    D32Float,
    R16Float,
    RG16Float,
    RGBA16Float,
    R32Float,
    RG32Float,
    RGBA32Float,
    BC1RGBAUnorm,
    BC3RGBAUnorm,
    BC4RUnorm,
    BC5RGUnorm,
    BC6HRGBFloat,
    BC7RGBAUnorm,
    BC7RGBASRGB,
    ASTC4x4RGBAUnorm,
    ETC2RGB8Unorm,
    ETC2RGBA8Unorm,
    EACR11Unorm,
    EACRG11Unorm,
}
impl TextureFormat {
    fn aspect_flags(texture_format: vk::Format) -> vk::ImageAspectFlags {
        match texture_format {
            vk::Format::D16_UNORM
            | vk::Format::D16_UNORM_S8_UINT
            | vk::Format::X8_D24_UNORM_PACK32
            | vk::Format::D24_UNORM_S8_UINT
            | vk::Format::D32_SFLOAT => vk::ImageAspectFlags::DEPTH,
            _ => vk::ImageAspectFlags::COLOR,
        }
    }
    fn to_vk(self) -> vk::Format {
        match self {
            TextureFormat::R8Unorm => vk::Format::R8_UNORM,
            TextureFormat::RG8Unorm => vk::Format::R8G8_UNORM,
            TextureFormat::RGBA8Unorm => vk::Format::R8G8B8A8_UNORM,
            TextureFormat::ABGR8Unorm => vk::Format::A8B8G8R8_UNORM_PACK32,
            TextureFormat::BGRA8Unorm => vk::Format::B8G8R8A8_UNORM,
            TextureFormat::R8SRGB => vk::Format::R8_SRGB,
            TextureFormat::RG8SRGB => vk::Format::R8G8_SRGB,
            TextureFormat::RGBA8SRGB => vk::Format::R8G8B8A8_SRGB,
            TextureFormat::ABGR8SRGB => vk::Format::A8B8G8R8_SRGB_PACK32,
            TextureFormat::BGRA8SRGB => vk::Format::B8G8R8A8_SRGB,
            TextureFormat::R16Unorm => vk::Format::R16_UNORM,
            TextureFormat::RG16Unorm => vk::Format::R16G16_UNORM,
            TextureFormat::RGBA16Unorm => vk::Format::R16G16B16A16_UNORM,
            TextureFormat::D16Unorm => vk::Format::D16_UNORM,
            TextureFormat::D16UnormS8Uint => vk::Format::D16_UNORM_S8_UINT,
            TextureFormat::D24UnormPack32 => vk::Format::X8_D24_UNORM_PACK32,
            TextureFormat::D24UnormS8Uint => vk::Format::D24_UNORM_S8_UINT,
            TextureFormat::D32Float => vk::Format::D32_SFLOAT,
            TextureFormat::R16Float => vk::Format::R16_SFLOAT,
            TextureFormat::RG16Float => vk::Format::R16G16_SFLOAT,
            TextureFormat::RGBA16Float => vk::Format::R16G16B16A16_SFLOAT,
            TextureFormat::R32Float => vk::Format::R32_SFLOAT,
            TextureFormat::RG32Float => vk::Format::R32G32_SFLOAT,
            TextureFormat::RGBA32Float => vk::Format::R32G32B32A32_SFLOAT,
            TextureFormat::BC1RGBAUnorm => vk::Format::BC1_RGBA_UNORM_BLOCK,
            TextureFormat::BC3RGBAUnorm => vk::Format::BC3_UNORM_BLOCK,
            TextureFormat::BC4RUnorm => vk::Format::BC4_UNORM_BLOCK,
            TextureFormat::BC5RGUnorm => vk::Format::BC5_UNORM_BLOCK,
            TextureFormat::BC6HRGBFloat => vk::Format::BC6H_SFLOAT_BLOCK,
            TextureFormat::BC7RGBAUnorm => vk::Format::BC7_UNORM_BLOCK,
            TextureFormat::BC7RGBASRGB => vk::Format::BC7_SRGB_BLOCK,
            TextureFormat::ASTC4x4RGBAUnorm => vk::Format::ASTC_4X4_UNORM_BLOCK,
            TextureFormat::ETC2RGB8Unorm => vk::Format::ETC2_R8G8B8_UNORM_BLOCK,
            TextureFormat::ETC2RGBA8Unorm => vk::Format::ETC2_R8G8B8A8_UNORM_BLOCK,
            TextureFormat::EACR11Unorm => vk::Format::EAC_R11_UNORM_BLOCK,
            TextureFormat::EACRG11Unorm => vk::Format::EAC_R11G11_UNORM_BLOCK,
        }
    }
}
pub struct GPUTexture {
    texture_type: TextureType,
    dimensions: [u32; 3],
    format: vk::Format,
    mip_count: u32,
    layer_count: u32,
    sample_count: u32,
    image: vk::Image,
    view_cache: Mutex<HashMap<TextureViewDesc, vk::ImageView>>,
    allocation: Option<Allocation>,
    is_swapchain: bool,
    gpu: Arc<GPU>,
}
impl GPUTexture {
    fn get_view(&self, view_desc: &TextureViewDesc) -> vk::ImageView {
        self.view_cache
            .lock()
            .unwrap()
            .entry(view_desc.clone())
            .or_insert_with(|| {
                assert!(
                    view_desc.base_layer as u32 + view_desc.layer_count as u32 <= self.layer_count
                );
                assert!(view_desc.base_mip as u32 + view_desc.mip_count as u32 <= self.mip_count);
                unsafe {
                    self.gpu
                        .device
                        .create_image_view(
                            &vk::ImageViewCreateInfo::default()
                                .image(self.image.clone())
                                .view_type(self.texture_type.to_view_vk())
                                .format(self.format)
                                .subresource_range(vk::ImageSubresourceRange {
                                    aspect_mask: TextureFormat::aspect_flags(self.format),
                                    base_array_layer: view_desc.base_layer as u32,
                                    base_mip_level: view_desc.base_mip,
                                    layer_count: view_desc.layer_count as u32,
                                    level_count: view_desc.mip_count as u32,
                                }),
                            None,
                        )
                        .unwrap()
                }
            })
            .clone()
    }
}
impl Drop for GPUTexture {
    fn drop(&mut self) {
        if let Some(allocation) = self.allocation.take() {
            self.gpu
                .allocator
                .as_ref()
                .unwrap()
                .lock()
                .unwrap()
                .free(allocation)
                .unwrap();
        }
        for (_, view) in self.view_cache.get_mut().unwrap() {
            unsafe {
                self.gpu.device.destroy_image_view(*view, None);
            }
        }
        if !self.is_swapchain {
            unsafe {
                self.gpu.device.destroy_image(self.image, None);
            }
        }
    }
}
pub struct GPUBuffer {
    allocation: Option<Allocation>,
    buffer: vk::Buffer,
    gpu_pointer: vk::DeviceAddress,
    gpu: Arc<GPU>,
}
impl GPUBuffer {
    pub fn gpu(&self) -> vk::DeviceAddress {
        self.gpu_pointer
    }
    pub fn upload<T: bytemuck::Pod>(&mut self, offset: usize, slice: &[T]) {
        let bytes = bytemuck::cast_slice::<T, u8>(slice);
        let allocation = self.allocation.as_mut().unwrap();
        if bytes.len() > allocation.size() as usize {
            panic!("too big, buffer size is {}", allocation.size());
        }
        let cpu_data = &mut allocation
            .mapped_slice_mut()
            .expect("buffer not mapped on cpu")[offset..(offset + bytes.len())];
        cpu_data.copy_from_slice(bytes);
    }
}
impl Drop for GPUBuffer {
    fn drop(&mut self) {
        self.gpu
            .allocator
            .as_ref()
            .unwrap()
            .lock()
            .unwrap()
            .free(self.allocation.take().unwrap())
            .unwrap();
        unsafe {
            self.gpu.device.destroy_buffer(self.buffer, None);
        }
    }
}
pub struct GPUSemaphore {
    semaphore: vk::Semaphore,
    gpu: Arc<GPU>,
}
impl GPUSemaphore {
    fn create_raw(device: &ash::Device, gpu: Arc<GPU>, value: u64) -> GPUSemaphore {
        let mut semaphore_create_type_info = vk::SemaphoreTypeCreateInfo::default()
            .initial_value(value)
            .semaphore_type(vk::SemaphoreType::TIMELINE);
        let semaphore = unsafe {
            device
                .create_semaphore(
                    &vk::SemaphoreCreateInfo::default().push_next(&mut semaphore_create_type_info),
                    None,
                )
                .unwrap()
        };
        GPUSemaphore { semaphore, gpu }
    }
    fn create_binary(device: &ash::Device, gpu: Arc<GPU>) -> GPUSemaphore {
        let mut semaphore_create_type_info =
            vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::BINARY);
        let semaphore = unsafe {
            device
                .create_semaphore(
                    &vk::SemaphoreCreateInfo::default().push_next(&mut semaphore_create_type_info),
                    None,
                )
                .unwrap()
        };
        GPUSemaphore { semaphore, gpu }
    }
    pub fn get_value(&self) -> u64 {
        unsafe {
            self.gpu
                .device
                .get_semaphore_counter_value(self.semaphore.clone())
                .unwrap()
        }
    }
    pub fn wait(&self, value: u64) {
        unsafe {
            self.gpu
                .device
                .wait_semaphores(
                    &vk::SemaphoreWaitInfo::default()
                        .semaphores(&[self.semaphore])
                        .values(&[value]),
                    u64::MAX,
                )
                .unwrap()
        }
    }
}
impl Drop for GPUSemaphore {
    fn drop(&mut self) {
        unsafe {
            self.gpu.device.destroy_semaphore(self.semaphore, None);
        }
    }
}
pub struct GPUShader {
    shader: vk::ShaderEXT,
    kind: ShaderKind,
    gpu: Arc<GPU>,
}
#[derive(Clone, Copy)]
pub enum ShaderKind {
    Vertex,
    Fragment,
    Compute,
}
impl Drop for GPUShader {
    fn drop(&mut self) {
        unsafe {
            self.gpu
                .shader_object_device
                .destroy_shader(self.shader, None);
        }
    }
}
pub struct GPUSwapchain {
    textures: Vec<Arc<GPUTexture>>,
    gpu: Arc<GPU>,
    present_semaphores: Vec<GPUSemaphore>,
    swapchain: vk::SwapchainKHR,
    next_swap_image: u32,
    surface: vk::SurfaceKHR,
}
impl GPUSwapchain {
    fn make_swapchain(
        gpu: &GPU,
        surface: vk::SurfaceKHR,
        textures: &mut Vec<Arc<GPUTexture>>,
    ) -> vk::SwapchainKHR {
        let surface_format: vk::SurfaceFormatKHR = unsafe {
            gpu.surface_loader
                .get_physical_device_surface_formats(gpu.physical_device, surface)
                .unwrap()
        }[0];
        let surface_capabilities = unsafe {
            gpu.surface_loader
                .get_physical_device_surface_capabilities(gpu.physical_device, surface)
                .unwrap()
        };
        let mut desired_image_count = surface_capabilities.min_image_count + 1;
        if surface_capabilities.max_image_count > 0
            && desired_image_count > surface_capabilities.max_image_count
        {
            desired_image_count = surface_capabilities.max_image_count;
        }
        let surface_resolution = match surface_capabilities.current_extent.width {
            u32::MAX => vk::Extent2D {
                width: 256,
                height: 256,
            },
            _ => surface_capabilities.current_extent,
        };
        let pre_transform = if surface_capabilities
            .supported_transforms
            .contains(vk::SurfaceTransformFlagsKHR::IDENTITY)
        {
            vk::SurfaceTransformFlagsKHR::IDENTITY
        } else {
            surface_capabilities.current_transform
        };
        let present_modes = unsafe {
            gpu.surface_loader
                .get_physical_device_surface_present_modes(gpu.physical_device, surface)
                .unwrap()
        };
        let present_mode = present_modes
            .iter()
            .cloned()
            .find(|&mode| mode == vk::PresentModeKHR::MAILBOX && !gpu.settings.vsync)
            .unwrap_or(vk::PresentModeKHR::FIFO);
        let info = vk::SwapchainCreateInfoKHR::default()
            .min_image_count(desired_image_count)
            .image_color_space(surface_format.color_space)
            .image_format(surface_format.format)
            .image_extent(surface_resolution)
            .pre_transform(pre_transform)
            .surface(surface)
            .present_mode(present_mode)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .clipped(true)
            .image_array_layers(1)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE);
        let swapchain = unsafe { gpu.swapchain_device.create_swapchain(&info, None).unwrap() };
        let images = unsafe {
            gpu.swapchain_device
                .get_swapchain_images(swapchain)
                .unwrap()
        };
        textures.extend(images.into_iter().map(|image| {
            Arc::new(GPUTexture {
                image,
                dimensions: [surface_resolution.width, surface_resolution.height, 1],
                format: surface_format.format,
                layer_count: 1,
                sample_count: 1,
                mip_count: 1,
                allocation: None,
                view_cache: Default::default(),
                gpu: gpu.weak.upgrade().unwrap(),
                texture_type: TextureType::D2,
                is_swapchain: true,
            })
        }));
        swapchain
    }
    pub fn acquire_next(&mut self) -> (Arc<GPUTexture>, SwapchainResult) {
        unsafe {
            let fence = self
                .gpu
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .unwrap();
            let (index, suboptimal) = self
                .gpu
                .swapchain_device
                .acquire_next_image(self.swapchain, u64::MAX, vk::Semaphore::null(), fence)
                .unwrap();
            self.next_swap_image = index;
            self.gpu
                .device
                .wait_for_fences(&[fence], true, u64::MAX)
                .unwrap();
            self.gpu.device.destroy_fence(fence, None);
            let command_buffer = self.gpu.begin_commands(QueueKind::Main);
            let image_barrier = vk::ImageMemoryBarrier2::default()
                .image(self.textures[index as usize].image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    layer_count: 1,
                    level_count: 1,
                    base_array_layer: 0,
                    base_mip_level: 0,
                })
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::GENERAL)
                .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE);
            self.gpu.device.cmd_pipeline_barrier2(
                command_buffer.command_buffer,
                &vk::DependencyInfo::default().image_memory_barriers(&[image_barrier]),
            );
            command_buffer.submit();
            (
                self.textures[index as usize].clone(),
                if suboptimal {
                    SwapchainResult::Suboptimal
                } else {
                    SwapchainResult::Ok
                },
            )
        }
    }
    pub fn present(
        &self,
        queue: QueueKind,
        wait_semaphore: &GPUSemaphore,
        wait_value: u64,
    ) -> SwapchainResult {
        let mut command_buffer = self.gpu.begin_commands(queue);
        let image_barrier = vk::ImageMemoryBarrier2::default()
            .image(self.textures[self.next_swap_image as usize].image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                layer_count: 1,
                level_count: 1,
                base_array_layer: 0,
                base_mip_level: 0,
            })
            .old_layout(vk::ImageLayout::GENERAL)
            .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(vk::AccessFlags2::MEMORY_READ);
        unsafe {
            self.gpu.device.cmd_pipeline_barrier2(
                command_buffer.command_buffer,
                &vk::DependencyInfo::default().image_memory_barriers(&[image_barrier]),
            )
        };
        command_buffer.add_wait(wait_semaphore, wait_value);
        command_buffer.add_signal(&self.present_semaphores[self.next_swap_image as usize], 0);
        command_buffer.submit();
        let _queue_lock = self.gpu.command_buffer_counters.lock().unwrap();
        unsafe {
            let suboptimal = self
                .gpu
                .swapchain_device
                .queue_present(
                    self.gpu.queues[queue as usize],
                    &vk::PresentInfoKHR::default()
                        .image_indices(&[self.next_swap_image])
                        .swapchains(&[self.swapchain])
                        .wait_semaphores(&[
                            self.present_semaphores[self.next_swap_image as usize].semaphore
                        ]),
                )
                .unwrap();
            match suboptimal {
                true => SwapchainResult::Suboptimal,
                false => SwapchainResult::Ok,
            }
        }
    }
    pub fn recreate(&mut self) {
        self.gpu.queue_wait_idle(QueueKind::Main);
        self.textures.clear();
        //todo: use old_swapchain
        unsafe {
            self.gpu
                .swapchain_device
                .destroy_swapchain(self.swapchain, None);
        }
        self.swapchain = GPUSwapchain::make_swapchain(&self.gpu, self.surface, &mut self.textures);
        self.present_semaphores = std::iter::repeat_with(|| {
            GPUSemaphore::create_binary(&self.gpu.device, self.gpu.clone())
        })
        .take(self.textures.len())
        .collect();
    }
}
pub enum SwapchainResult {
    Ok,
    Suboptimal,
}
impl Drop for GPUSwapchain {
    fn drop(&mut self) {
        self.gpu.queue_wait_idle(QueueKind::Main);
        unsafe {
            self.gpu
                .swapchain_device
                .destroy_swapchain(self.swapchain, None);
        }
        unsafe {
            self.gpu.surface_loader.destroy_surface(self.surface, None);
        }
    }
}
#[derive(Clone, Copy)]
pub enum QueueKind {
    Main = 0,
    Compute = 1,
    Transfer = 2,
}
impl QueueKind {
    fn is_graphics(self) -> bool {
        match self {
            QueueKind::Main => true,
            _ => false,
        }
    }
    fn is_compute(self) -> bool {
        match self {
            QueueKind::Main | QueueKind::Compute => true,
            _ => false,
        }
    }
    fn is_transfer(self) -> bool {
        true
    }
}
#[derive(Clone, Copy)]
pub enum MemoryKind {
    Default,
    GPU,
    Readback,
}

fn to_vk_sample_count(sample_count: u32) -> vk::SampleCountFlags {
    match sample_count {
        1 => vk::SampleCountFlags::TYPE_1,
        2 => vk::SampleCountFlags::TYPE_2,
        4 => vk::SampleCountFlags::TYPE_4,
        8 => vk::SampleCountFlags::TYPE_8,
        16 => vk::SampleCountFlags::TYPE_16,
        32 => vk::SampleCountFlags::TYPE_32,
        64 => vk::SampleCountFlags::TYPE_64,
        _ => panic!(),
    }
}
pub struct GPUArena {
    gpu: Arc<GPU>,
    memory: MemoryKind,
    block_size: usize,
    blocks: Vec<GPUBuffer>,
    block_index: usize,
    block_offset: usize,
}
impl GPUArena {
    pub fn alloc<T: bytemuck::Pod>(&mut self, slice: &[T]) -> GPUPointer {
        let layout = Layout::new::<T>().repeat(slice.len()).unwrap().0;
        if layout.align() > 16 {
            panic!("align must be at most 16");
        }
        if layout.size() > self.block_size {
            panic!(
                "cannot allocate objects bigger than block_size({})",
                self.block_size
            );
        }
        let next_addr_align = (self.block_offset + layout.align() - 1) & !(layout.align() - 1);
        if next_addr_align + layout.size() > self.block_size {
            self.block_index += 1;
            if self.block_index >= self.blocks.len() {
                self.blocks.push(self.gpu.allocate_buffer(
                    Layout::from_size_align(self.block_size, 16).unwrap(),
                    self.memory,
                ));
            }
            let block = self.blocks.get_mut(self.block_index).unwrap();
            block.upload(0, slice);
            self.block_offset = layout.size();
            GPUPointer {
                buffer: block.buffer,
                offset: 0,
                pointer: block.gpu(),
                size: layout.size() as u64,
            }
        } else {
            let block = &mut self.blocks[self.block_index];
            block.upload(next_addr_align, slice);
            self.block_offset = next_addr_align + layout.size();
            GPUPointer {
                buffer: block.buffer,
                offset: next_addr_align as u64,
                pointer: block.gpu() + next_addr_align as u64,
                size: layout.size() as u64,
            }
        }
    }
    pub fn alloc_multi_split<T: bytemuck::Pod>(&mut self, slice: &[T]) -> Vec<(GPUPointer, usize)> {
        let block_max_fit = self.block_size / std::mem::size_of::<T>();
        let mut output = Vec::new();
        let full_count = slice.len() / block_max_fit;
        for i in 0..full_count {
            output.push((
                self.alloc(&slice[(i * block_max_fit)..((i + 1) * block_max_fit)]),
                block_max_fit,
            ));
        }
        let reminder = slice.len() % block_max_fit;
        if reminder > 0 {
            output.push((self.alloc(&slice[(full_count * block_max_fit)..]), reminder));
        }
        output
    }
    pub fn clear(&mut self) {
        self.block_index = 0;
        self.block_offset = 0;
    }
}
pub struct GraphicsShaders {
    pub vertex: GPUShader,
    pub fragment: GPUShader,
}
pub struct GPUDescriptorHeap {
    gpu: Arc<GPU>,
    pool: vk::DescriptorPool,
    textures: vk::DescriptorSet,
    textures_rw: vk::DescriptorSet,
    samplers: vk::DescriptorSet,
    sampler_cache: HashMap<TextureSampler, vk::Sampler>,
}
impl GPUDescriptorHeap {
    pub fn set_textures(&mut self, offset: usize, textures: &[(&GPUTexture, TextureViewDesc)]) {
        let mut infos = Vec::new();
        for (texture, view) in textures {
            infos.push(
                vk::DescriptorImageInfo::default()
                    .image_view(texture.get_view(view))
                    .image_layout(vk::ImageLayout::GENERAL),
            );
        }
        let write = vk::WriteDescriptorSet::default()
            .dst_set(self.textures)
            .dst_binding(0)
            .dst_array_element(offset as u32)
            .image_info(&infos)
            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE);
        unsafe { self.gpu.device.update_descriptor_sets(&[write], &[]) };
    }
    pub fn set_textures_rw(&mut self, offset: usize, textures: &[(&GPUTexture, TextureViewDesc)]) {
        let mut infos = Vec::new();
        for (texture, view) in textures {
            infos.push(
                vk::DescriptorImageInfo::default()
                    .image_view(texture.get_view(view))
                    .image_layout(vk::ImageLayout::GENERAL),
            );
        }
        let write = vk::WriteDescriptorSet::default()
            .dst_set(self.textures_rw)
            .dst_binding(0)
            .dst_array_element(offset as u32)
            .image_info(&infos)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE);
        unsafe { self.gpu.device.update_descriptor_sets(&[write], &[]) };
    }
    pub fn set_samplers(&mut self, offset: usize, samplers: &[TextureSampler]) {
        let mut infos = Vec::new();
        for sampler in samplers {
            infos.push(vk::DescriptorImageInfo::default().sampler(
                *self.sampler_cache.entry(*sampler).or_insert_with(|| {
                    let create_info = vk::SamplerCreateInfo::default()
                        .address_mode_u(sampler.u.to_vk())
                        .address_mode_v(sampler.v.to_vk())
                        .address_mode_w(sampler.w.to_vk())
                        .anisotropy_enable(sampler.max_anisotropy.is_some())
                        .max_anisotropy(sampler.max_anisotropy.unwrap_or(0) as f32)
                        .border_color(sampler.border.to_vk())
                        .min_filter(sampler.min.to_vk())
                        .mag_filter(sampler.mag.to_vk())
                        .mipmap_mode(sampler.mip.to_vk_mip());
                    unsafe { self.gpu.device.create_sampler(&create_info, None).unwrap() }
                }),
            ));
        }
        let write = vk::WriteDescriptorSet::default()
            .dst_set(self.samplers)
            .dst_binding(0)
            .dst_array_element(offset as u32)
            .image_info(&infos)
            .descriptor_type(vk::DescriptorType::SAMPLER);
        unsafe { self.gpu.device.update_descriptor_sets(&[write], &[]) };
    }
}
impl Drop for GPUDescriptorHeap {
    fn drop(&mut self) {
        unsafe {
            self.gpu
                .device
                .free_descriptor_sets(self.pool, &[self.textures, self.textures_rw, self.samplers])
                .unwrap();
            self.gpu.device.destroy_descriptor_pool(self.pool, None);
            for sampler in self.sampler_cache.values() {
                self.gpu.device.destroy_sampler(*sampler, None);
            }
        };
    }
}
#[derive(Hash, Clone, Copy, PartialEq, Eq)]
pub struct TextureSampler {
    pub min: SamplerFilter,
    pub mag: SamplerFilter,
    pub mip: SamplerFilter,
    pub u: SamplerAddressMode,
    pub v: SamplerAddressMode,
    pub w: SamplerAddressMode,
    pub max_anisotropy: Option<u8>, //should be float, but then not eq
    pub border: SamplerBorder,
}
impl Default for TextureSampler {
    fn default() -> Self {
        TextureSampler {
            min: SamplerFilter::Linear,
            mag: SamplerFilter::Linear,
            mip: SamplerFilter::Linear,
            u: SamplerAddressMode::Repeat,
            v: SamplerAddressMode::Repeat,
            w: SamplerAddressMode::Repeat,
            max_anisotropy: None,
            border: SamplerBorder::FloatTransparentBlack,
        }
    }
}
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum SamplerFilter {
    Nearest,
    Linear,
}
impl SamplerFilter {
    fn to_vk(self) -> vk::Filter {
        match self {
            SamplerFilter::Linear => vk::Filter::LINEAR,
            SamplerFilter::Nearest => vk::Filter::NEAREST,
        }
    }
    fn to_vk_mip(self) -> vk::SamplerMipmapMode {
        match self {
            SamplerFilter::Nearest => vk::SamplerMipmapMode::NEAREST,
            SamplerFilter::Linear => vk::SamplerMipmapMode::LINEAR,
        }
    }
}
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum SamplerAddressMode {
    Repeat,
    MirrorRepeat,
    Edge,
    Border,
}
impl SamplerAddressMode {
    fn to_vk(self) -> vk::SamplerAddressMode {
        match self {
            SamplerAddressMode::Repeat => vk::SamplerAddressMode::REPEAT,
            SamplerAddressMode::MirrorRepeat => vk::SamplerAddressMode::MIRRORED_REPEAT,
            SamplerAddressMode::Edge => vk::SamplerAddressMode::CLAMP_TO_EDGE,
            SamplerAddressMode::Border => vk::SamplerAddressMode::CLAMP_TO_BORDER,
        }
    }
}
//todo: more
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum SamplerBorder {
    FloatTransparentBlack,
    IntTransparentBlack,
}
impl SamplerBorder {
    fn to_vk(self) -> vk::BorderColor {
        match self {
            SamplerBorder::FloatTransparentBlack => vk::BorderColor::FLOAT_TRANSPARENT_BLACK,
            SamplerBorder::IntTransparentBlack => vk::BorderColor::INT_TRANSPARENT_BLACK,
        }
    }
}
pub struct GPUSettings {
    pub desc_heap_max_textures: usize,
    pub desc_heap_max_samplers: usize,
    pub max_push_constant_size: usize,
    pub validate: bool,
    pub vsync: bool,
    pub instance_extensions: Vec<*const i8>,
}
impl GPUSettings {
    pub fn get_required_extensions(display: DisplayHandle) -> Vec<*const i8> {
        ash_window::enumerate_required_extensions(display.as_raw())
            .unwrap()
            .to_vec()
    }
}
impl Default for GPUSettings {
    fn default() -> Self {
        GPUSettings {
            desc_heap_max_textures: 32768,
            desc_heap_max_samplers: 32,
            max_push_constant_size: 128,
            validate: true,
            vsync: false,
            instance_extensions: Vec::new(),
        }
    }
}
pub struct TextureRegion {
    pub offset: [i32; 3],
    pub size: [u32; 3],
    pub mip_level: u32,
    pub base_layer: u32,
    pub layer_count: u32,
}
impl Default for TextureRegion {
    fn default() -> Self {
        TextureRegion {
            offset: [0; 3],
            size: [0; 3],
            mip_level: 0,
            base_layer: 0,
            layer_count: 1,
        }
    }
}
#[derive(Clone, Copy)]
pub struct GPUPointer {
    buffer: vk::Buffer,
    offset: u64,
    pointer: u64,
    size: u64,
}
impl GPUPointer {
    pub fn gpu(self) -> u64 {
        self.pointer
    }
    pub fn slice(self, offset: u64, size: u64) -> GPUPointer {
        if offset + size > self.size {
            panic!("overflow")
        };
        GPUPointer {
            buffer: self.buffer,
            offset: self.offset + offset,
            pointer: self.pointer + offset,
            size,
        }
    }
}
impl Into<GPUPointer> for &GPUBuffer {
    fn into(self) -> GPUPointer {
        GPUPointer {
            buffer: self.buffer,
            offset: self.allocation.as_ref().unwrap().offset(),
            pointer: self.gpu_pointer,
            size: self.allocation.as_ref().unwrap().size(),
        }
    }
}
pub struct TextureUsage(u8);
impl TextureUsage {
    pub const EMPTY: TextureUsage = TextureUsage(0);
    pub const SAMPLED: TextureUsage = TextureUsage(1);
    pub const STORAGE: TextureUsage = TextureUsage(2);
    pub const TRANSFER_SRC: TextureUsage = TextureUsage(4);
    pub const COLOR_ATTACHMENT: TextureUsage = TextureUsage(8);
    pub const DEPTH_STENCIL_ATTACHMENT: TextureUsage = TextureUsage(16);
}
impl std::ops::BitOr for TextureUsage {
    type Output = TextureUsage;
    fn bitor(self, rhs: Self) -> Self::Output {
        TextureUsage(self.0 | rhs.0)
    }
}
#[derive(Clone, Copy)]
pub enum IndexBufferFormat {
    U16,
    U32,
}
impl IndexBufferFormat {
    pub fn size(self) -> usize {
        match self {
            IndexBufferFormat::U16 => 2,
            IndexBufferFormat::U32 => 4,
        }
    }
}
#[derive(Clone, Copy)]
pub enum GPUStage {
    Transfer,
    Compute,
    RasterColorOut,
    FragmentShader,
    VertexShader,
    All,
}
impl GPUStage {
    fn to_vk(self) -> vk::PipelineStageFlags {
        match self {
            GPUStage::Transfer => vk::PipelineStageFlags::TRANSFER,
            GPUStage::Compute => vk::PipelineStageFlags::COMPUTE_SHADER,
            GPUStage::RasterColorOut => vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            GPUStage::FragmentShader => vk::PipelineStageFlags::FRAGMENT_SHADER,
            GPUStage::VertexShader => vk::PipelineStageFlags::VERTEX_SHADER,
            GPUStage::All => vk::PipelineStageFlags::ALL_COMMANDS,
        }
    }
}
#[derive(Clone, Copy)]
pub enum GPUHazard {
    DrawArguments,
    Descriptors,
    DepthStencil,
}
#[derive(Clone, Copy)]
pub enum GPUCullMode {
    KeepAll,
    KeepCW,
    KeepCCW,
}
#[derive(Clone, Copy)]
pub enum GPUTopology {
    TriangleList,
    TriangleStrip,
    TriangleFan,
}
#[derive(Clone, Copy)]
pub struct GPUDepthTestConfig {
    write: bool,
    compare: GPUCompareOp,
}
impl Default for GPUDepthTestConfig {
    fn default() -> Self {
        Self {
            write: true,
            compare: GPUCompareOp::Less,
        }
    }
}
#[derive(Clone, Copy)]
pub enum GPUCompareOp {
    Always,
    Never,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    Equal,
    NotEqual,
}
