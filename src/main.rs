use std::{
    alloc::Layout,
    cell::RefCell,
    collections::{BinaryHeap, HashMap, HashSet},
    ffi::{CStr, CString},
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
use thread_local::ThreadLocal;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    raw_window_handle::{DisplayHandle, HasDisplayHandle, HasWindowHandle, WindowHandle},
    window::WindowId,
};

#[derive(Default)]
struct App {
    data: Option<AppData>,
}
struct AppData {
    window: winit::window::Window,
    gpu: Arc<GPU>,
    swapchain: GPUSwapchain,
    semaphore: GPUSemaphore,
    bump_value: u64,
    vertex_shader: GPUShader,
    fragment_shader: GPUShader,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = event_loop
            .create_window(winit::window::Window::default_attributes())
            .unwrap();
        let mut enabled_extensions =
            ash_window::enumerate_required_extensions(window.display_handle().unwrap().as_raw())
                .unwrap()
                .to_vec();
        println!(
            "here: {:?}",
            enabled_extensions
                .into_iter()
                .map(|ptr| unsafe { CStr::from_ptr(ptr).to_str().unwrap() })
                .collect::<Vec<_>>()
        );
        let gpu = GPU::init().unwrap();
        let semaphore = gpu.create_semaphore(0);
        let swapchain = gpu.create_swapchain(
            window.display_handle().unwrap(),
            window.window_handle().unwrap(),
        );
        let shader = include_bytes!("shader.spv");
        println!("{}", shader.len());
        let vertex_shader = gpu
            .create_shader_spirv(
                shader,
                "vertexMain",
                GPUShaderKind::Vertex,
                Default::default(),
            )
            .unwrap();
        let fragment_shader = gpu
            .create_shader_spirv(
                shader,
                "fragmentMain",
                GPUShaderKind::Fragment,
                Default::default(),
            )
            .unwrap();
        self.data = Some(AppData {
            window,
            gpu,
            swapchain,
            semaphore,
            bump_value: 1,
            vertex_shader,
            fragment_shader,
        });
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                println!("The close button was pressed; stopping");
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                let Some(data) = &mut self.data else { return };
                let mut commands = data.gpu.begin_commands(QueueKind::Main);
                let target = data.swapchain.acquire_next();
                if true {
                    let mut render_pass = commands.begin_render(RenderPassDesc {
                        color_attachments: &[RenderPassAttachment {
                            load: RenderPassLoadOp::Clear([1., 1., 0., 1.]),
                            store: RenderPassStoreOp::Store,
                            resolve: None,
                            texture: &target,
                            view: Default::default(),
                        }],
                        ..Default::default()
                    });
                    render_pass.set_shaders(&data.vertex_shader, &data.fragment_shader);
                    render_pass.draw(0, 0, 3, 1);
                }
                commands.add_signal(&data.semaphore, data.bump_value);
                commands.submit();

                data.swapchain
                    .present(QueueKind::Main, &data.semaphore, data.bump_value);

                data.bump_value += 1;
                //println!("frame rendered {}", self.bump_value);

                data.window.request_redraw();
            }
            _ => (),
        }
    }
}

fn main() -> VkResult<()> {
    let event_loop = EventLoop::new().unwrap();
    //event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::default();
    event_loop.run_app(&mut app);
    Ok(())
}

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
    command_buffer_semaphores: [GPUSemaphore; 3],
    command_buffer_counters: Mutex<[u64; 3]>,
    allocator: Mutex<Allocator>,
    thread_local: ThreadLocal<RefCell<GPUThreadLocalStorage>>,
    graphics_common_pipeline_layout: vk::PipelineLayout,
    weak: Weak<GPU>,
}
impl GPU {
    pub fn init() -> VkResult<Arc<GPU>> {
        unsafe {
            let entry = Entry::linked();
            let app_info = vk::ApplicationInfo {
                api_version: vk::API_VERSION_1_3,
                ..Default::default()
            };
            let mut enabled_extension_instance_names = vec![];
            enabled_extension_instance_names.push(ash::khr::surface::NAME.as_ptr());
            enabled_extension_instance_names.push(ash::khr::xlib_surface::NAME.as_ptr());
            let create_info = vk::InstanceCreateInfo {
                p_application_info: &app_info,
                ..Default::default()
            }
            .enabled_extension_names(&enabled_extension_instance_names);
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

            let graphics_common_pipeline_layout = unsafe {
                device
                    .create_pipeline_layout(
                        &vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&[
                            vk::PushConstantRange {
                                offset: 0,
                                size: std::mem::size_of::<GraphicsShaderPushConstants>() as u32,
                                stage_flags: vk::ShaderStageFlags::VERTEX
                                    | vk::ShaderStageFlags::FRAGMENT,
                            },
                        ]),
                        None,
                    )
                    .unwrap()
            };

            Ok(Arc::new_cyclic(move |weak| GPU {
                shader_object_device,
                swapchain_device,
                surface_loader,
                queues,
                physical_device,
                queue_families,
                allocator: Mutex::new(allocator),
                thread_local: ThreadLocal::new(),
                command_buffer_counters: Mutex::new([0; 3]),
                command_buffer_semaphores: std::array::from_fn(|_| {
                    GPUSemaphore::create_raw(&device, weak, 0)
                }),
                graphics_common_pipeline_layout,
                weak: weak.clone(),
                device,
                instance,
                entry,
            }))
        }
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
            gpu: self.weak.clone(),
        }
    }
    pub fn allocate_buffer_with<T: bytemuck::Pod>(&self, data: &[T]) -> GPUBuffer {
        let mut buffer =
            self.allocate_buffer(Layout::array::<T>(data.len()).unwrap(), MemoryKind::Default);
        buffer.upload(data);
        buffer
    }
    pub fn create_semaphore(&self, value: u64) -> GPUSemaphore {
        GPUSemaphore::create_raw(&self.device, &self.weak, value)
    }
    pub fn create_shader_spirv(
        &self,
        code: &[u8],
        entry_point: &str,
        kind: GPUShaderKind,
        specializations: HashMap<u32, u32>,
    ) -> VkResult<GPUShader> {
        let mut aligned_code = vec![0u32; code.len().div_ceil(4)];
        let aligned_code =
            &mut bytemuck::cast_slice_mut::<_, u8>(aligned_code.as_mut_slice())[..code.len()];
        aligned_code.copy_from_slice(code);
        let (stage, next_stage) = match kind {
            GPUShaderKind::Vertex => (vk::ShaderStageFlags::VERTEX, vk::ShaderStageFlags::FRAGMENT),
            GPUShaderKind::Fragment => (
                vk::ShaderStageFlags::FRAGMENT,
                vk::ShaderStageFlags::empty(),
            ),
            GPUShaderKind::Compute { workgroup_size } => {
                (vk::ShaderStageFlags::COMPUTE, vk::ShaderStageFlags::empty())
            }
        };
        let mut push_constants = Vec::new();
        push_constants.push(match kind {
            GPUShaderKind::Vertex | GPUShaderKind::Fragment => vk::PushConstantRange {
                offset: 0,
                size: std::mem::size_of::<GraphicsShaderPushConstants>() as u32,
                stage_flags: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
            },
            GPUShaderKind::Compute { .. } => vk::PushConstantRange {
                offset: 0,
                size: std::mem::size_of::<ComputeShaderPushConstants>() as u32,
                stage_flags: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
            },
        });
        //todo: compute specializations
        let mut specialization_data = Vec::new();
        let mut specialization_map = Vec::new();
        for (constant_id, data) in specializations {
            specialization_map.push(vk::SpecializationMapEntry {
                constant_id,
                offset: (specialization_data.len() * 4) as u32,
                size: 4,
            });
            specialization_data.push(data);
        }
        let mut specialization_info = vk::SpecializationInfo::default()
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
            .specialization_info(&specialization_info);
        let shader = unsafe {
            self.shader_object_device
                .create_shaders(&[shader_create_info], None)
                .unwrap()[0] //todo: error handling
        };
        Ok(GPUShader {
            shader,
            kind,
            gpu: self.weak.clone(),
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
        let current_semaphore_value = self.command_buffer_semaphores[queue as usize].get_value();
        if let Some(pooled) = local.pools[queue as usize].peek() {
            if pooled.timestamp <= current_semaphore_value {
                let pooled = local.pools[queue as usize].pop().unwrap();
                unsafe {
                    self.device
                        .begin_command_buffer(pooled.command_buffer, &command_begin_info);
                }
                return GPUCommandBuffer {
                    command_buffer: pooled.command_buffer,
                    queue,
                    gpu: self.weak.clone(),
                    signalling: Vec::new(),
                    waiting: Vec::new(),
                    submitted: false,
                    _marker: PhantomData,
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
                .begin_command_buffer(command_buffer, &command_begin_info);
        }
        GPUCommandBuffer {
            command_buffer,
            queue,
            gpu: self.weak.clone(),
            signalling: Vec::new(),
            waiting: Vec::new(),
            submitted: false,
            _marker: PhantomData,
        }
    }
    pub fn create_texture(&self, desc: TextureDesc) -> GPUTexture {
        let image = unsafe {
            self.device
                .create_image(
                    &&vk::ImageCreateInfo::default()
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
                        .usage(
                            //todo: proper usage
                            vk::ImageUsageFlags::TRANSFER_DST
                                | vk::ImageUsageFlags::TRANSFER_SRC
                                | vk::ImageUsageFlags::SAMPLED
                                | vk::ImageUsageFlags::STORAGE
                                | vk::ImageUsageFlags::COLOR_ATTACHMENT
                                | vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
                        )
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
        GPUTexture {
            texture_type: desc.texture_type,
            dimensions: desc.dimensions,
            format: desc.format,
            mip_count: desc.mip_count,
            layer_count: desc.layer_count,
            sample_count: desc.sample_count,
            image,
            view_cache: Default::default(),
            gpu: self.weak.clone(),
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
            let surface_format = self
                .surface_loader
                .get_physical_device_surface_formats(self.physical_device, surface)
                .unwrap()[0];
            let surface_capabilities = self
                .surface_loader
                .get_physical_device_surface_capabilities(self.physical_device, surface)
                .unwrap();
            let mut desired_image_count = surface_capabilities.min_image_count + 1;
            if surface_capabilities.max_image_count > 0
                && desired_image_count > surface_capabilities.max_image_count
            {
                desired_image_count = surface_capabilities.max_image_count;
            }
            let surface_resolution = match surface_capabilities.current_extent.width {
                u32::MAX => vk::Extent2D {
                    width: 400,
                    height: 400,
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
            let present_modes = self
                .surface_loader
                .get_physical_device_surface_present_modes(self.physical_device, surface)
                .unwrap();
            /*let present_mode = present_modes
            .iter()
            .cloned()
            .find(|&mode| mode == vk::PresentModeKHR::MAILBOX)
            .unwrap_or(vk::PresentModeKHR::FIFO);*/
            let present_mode = vk::PresentModeKHR::FIFO;
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
            let swapchain = unsafe { self.swapchain_device.create_swapchain(&info, None).unwrap() };
            let images = self
                .swapchain_device
                .get_swapchain_images(swapchain)
                .unwrap();
            let textures: Vec<Arc<GPUTexture>> = images
                .into_iter()
                .map(|image| {
                    Arc::new(GPUTexture {
                        image,
                        dimensions: [surface_resolution.width, surface_resolution.height, 1],
                        format: match surface_format.format {
                            vk::Format::B8G8R8A8_SRGB => TextureFormat::BGRA8SRGB,
                            format => unimplemented!("{:?}", format),
                        },
                        layer_count: 1,
                        sample_count: 1,
                        mip_count: 1,
                        view_cache: Default::default(),
                        gpu: self.weak.clone(),
                        texture_type: TextureType::D2,
                    })
                })
                .collect();
            GPUSwapchain {
                present_semaphores: std::iter::repeat_with(|| {
                    GPUSemaphore::create_binary(&self.device, &self.weak)
                })
                .take(textures.len())
                .collect(),
                swapchain,
                textures,
                next_swap_image: 0,
                gpu: self.weak.clone(),
            }
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
}
impl Default for TextureDesc {
    fn default() -> Self {
        TextureDesc {
            texture_type: TextureType::D2,
            dimensions: [0; 3],
            format: TextureFormat::RGBA8Unorm,
            mip_count: 1,
            layer_count: 1,
            sample_count: 1,
        }
    }
}
impl Drop for GPU {
    fn drop(&mut self) {
        /*unsafe {
            self.device.destroy_device(None);
        }*/
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
    gpu: Weak<GPU>,
    waiting: Vec<(vk::Semaphore, u64)>,
    signalling: Vec<(vk::Semaphore, u64)>,
    _marker: PhantomData<*mut ()>, //not send
    submitted: bool,
}
impl GPUCommandBuffer {
    pub fn add_wait(&mut self, semaphore: &GPUSemaphore, value: u64) {
        self.waiting.push((semaphore.semaphore, value));
    }
    pub fn add_signal(&mut self, semaphore: &GPUSemaphore, value: u64) {
        self.signalling.push((semaphore.semaphore, value));
    }
    pub fn begin_render<'a>(&'a mut self, mut desc: RenderPassDesc) -> GPURenderPass<'a> {
        let gpu = self.gpu.upgrade().unwrap();
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
            gpu.device.cmd_begin_rendering(self.command_buffer, &info);
            gpu.device
                .cmd_set_stencil_test_enable(self.command_buffer, false);
            gpu.shader_object_device.cmd_set_color_blend_enable(
                self.command_buffer,
                0,
                std::iter::repeat_n(0, desc.color_attachments.len())
                    .collect::<Vec<_>>()
                    .as_slice(),
            );
            gpu.shader_object_device.cmd_set_color_write_mask(
                self.command_buffer,
                0,
                std::iter::repeat_n(vk::ColorComponentFlags::RGBA, desc.color_attachments.len())
                    .collect::<Vec<_>>()
                    .as_slice(),
            );
            gpu.shader_object_device.cmd_set_rasterization_samples(
                self.command_buffer,
                to_vk_sample_count(sample_count),
            );
            gpu.shader_object_device.cmd_set_primitive_topology(
                self.command_buffer,
                vk::PrimitiveTopology::TRIANGLE_LIST,
            );
            gpu.shader_object_device
                .cmd_set_polygon_mode(self.command_buffer, vk::PolygonMode::FILL);
            gpu.shader_object_device
                .cmd_set_cull_mode(self.command_buffer, vk::CullModeFlags::NONE); //todo: configurable
            gpu.shader_object_device
                .cmd_set_depth_compare_op(self.command_buffer, vk::CompareOp::ALWAYS); //todo
            gpu.shader_object_device
                .cmd_set_depth_bias_enable(self.command_buffer, false);
            gpu.shader_object_device
                .cmd_set_depth_write_enable(self.command_buffer, false);
            gpu.shader_object_device
                .cmd_set_depth_test_enable(self.command_buffer, false);

            gpu.shader_object_device.cmd_set_viewport_with_count(
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
            gpu.shader_object_device.cmd_set_scissor_with_count(
                self.command_buffer,
                &[vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: vk::Extent2D {
                        width: desc.render_area_size[0],
                        height: desc.render_area_size[1],
                    },
                }],
            );
            gpu.shader_object_device
                .cmd_set_rasterizer_discard_enable(self.command_buffer, false);
            gpu.shader_object_device.cmd_set_color_blend_equation(
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
            gpu.shader_object_device.cmd_set_sample_mask(
                self.command_buffer,
                to_vk_sample_count(sample_count),
                &[255],
            );
            gpu.shader_object_device
                .cmd_set_alpha_to_coverage_enable(self.command_buffer, false);

            gpu.shader_object_device
                .cmd_set_depth_clip_enable(self.command_buffer, true);
            gpu.shader_object_device
                .cmd_set_vertex_input(self.command_buffer, &[], &[]);
            gpu.shader_object_device
                .cmd_set_primitive_restart_enable(self.command_buffer, false);
        }
        GPURenderPass {
            command_buffer: self,
        }
    }
    pub fn submit(mut self) {
        self.submitted = true;
        let gpu = self.gpu.upgrade().unwrap();
        unsafe {
            gpu.device.end_command_buffer(self.command_buffer).unwrap();
        }
        let mut counters = gpu.command_buffer_counters.lock().unwrap();
        let cnt = counters[self.queue as usize] + 1;
        counters[self.queue as usize] = cnt;

        let mut signal_semaphores =
            vec![gpu.command_buffer_semaphores[self.queue as usize].semaphore];
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
            gpu.device
                .queue_submit(
                    gpu.queues[self.queue as usize],
                    &[submit_info],
                    vk::Fence::null(),
                )
                .unwrap();
        }

        gpu.thread_local.get().unwrap().borrow_mut().pools[self.queue as usize].push(
            PooledCommandBuffer {
                command_buffer: self.command_buffer,
                timestamp: cnt,
            },
        );
    }
}
impl Drop for GPUCommandBuffer {
    fn drop(&mut self) {
        if !self.submitted {
            panic!("buffer not submitted")
        }
    }
}
pub struct GPURenderPass<'a> {
    command_buffer: &'a mut GPUCommandBuffer,
}
impl GPURenderPass<'_> {
    pub fn set_shaders(&mut self, vertex: &GPUShader, fragment: &GPUShader) {
        let gpu = self.command_buffer.gpu.upgrade().unwrap();
        unsafe {
            gpu.shader_object_device.cmd_bind_shaders(
                self.command_buffer.command_buffer,
                &[vk::ShaderStageFlags::VERTEX, vk::ShaderStageFlags::FRAGMENT],
                &[vertex.shader, fragment.shader],
            )
        };
    }
    pub fn draw(
        &mut self,
        vertex_data: vk::DeviceAddress,
        fragment_data: vk::DeviceAddress,
        vertex_count: u32,
        instance_count: u32,
    ) {
        let gpu = self.command_buffer.gpu.upgrade().unwrap();
        unsafe {
            gpu.device.cmd_push_constants(
                self.command_buffer.command_buffer,
                gpu.graphics_common_pipeline_layout,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                0,
                bytemuck::bytes_of(&GraphicsShaderPushConstants {
                    vertex_data,
                    fragment_data,
                    indirect_data: 0,
                }),
            );
            gpu.device.cmd_draw(
                self.command_buffer.command_buffer,
                vertex_count,
                instance_count,
                0,
                0,
            );
        }
    }
}
impl Drop for GPURenderPass<'_> {
    fn drop(&mut self) {
        let gpu = self.command_buffer.gpu.upgrade().unwrap();
        unsafe {
            gpu.device
                .cmd_end_rendering(self.command_buffer.command_buffer);
        }
    }
}
pub struct RenderPassDesc<'a> {
    render_area_offset: [i32; 2],
    render_area_size: [u32; 2],
    layer_count: u32,
    view_mask: u32,
    color_attachments: &'a [RenderPassAttachment<'a, [f32; 4]>],
    depth_attachment: Option<RenderPassAttachment<'a, f32>>,
    //stencil_attachment: Option<RenderPassAttachment<'a>>,
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
        }
    }
}
pub struct RenderPassAttachment<'a, C> {
    texture: &'a GPUTexture,
    view: TextureViewDesc,
    load: RenderPassLoadOp<C>,
    store: RenderPassStoreOp,
    resolve: Option<(&'a GPUTexture, TextureViewDesc)>,
}
trait IntoClearValue: Copy {
    fn into_clear_value(self, format: TextureFormat) -> vk::ClearValue;
}
impl IntoClearValue for [f32; 4] {
    fn into_clear_value(self, format: TextureFormat) -> vk::ClearValue {
        vk::ClearValue {
            color: vk::ClearColorValue { float32: self },
        }
    }
}
impl IntoClearValue for f32 {
    fn into_clear_value(self, format: TextureFormat) -> vk::ClearValue {
        vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue {
                depth: self,
                stencil: 0,
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
enum TextureFormat {
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
    fn to_vk(self) -> vk::Format {
        match self {
            TextureFormat::R8Unorm => vk::Format::R8_UNORM,
            TextureFormat::RG8Unorm => vk::Format::R8G8_UNORM,
            TextureFormat::RGBA8Unorm => vk::Format::R8G8B8A8_UNORM,
            TextureFormat::ABGR8Unorm => vk::Format::A8B8G8R8_UNORM_PACK32,
            TextureFormat::BGRA8Unorm => vk::Format::B8G8R8A8_UNORM,
            TextureFormat::R8SRGB => todo!(),
            TextureFormat::RG8SRGB => todo!(),
            TextureFormat::RGBA8SRGB => todo!(),
            TextureFormat::ABGR8SRGB => todo!(),
            TextureFormat::BGRA8SRGB => vk::Format::B8G8R8A8_SRGB,
            TextureFormat::R16Unorm => todo!(),
            TextureFormat::RG16Unorm => todo!(),
            TextureFormat::RGBA16Unorm => todo!(),
            TextureFormat::D16Unorm => todo!(),
            TextureFormat::D16UnormS8Uint => todo!(),
            TextureFormat::D24UnormPack32 => todo!(),
            TextureFormat::D24UnormS8Uint => todo!(),
            TextureFormat::D32Float => todo!(),
            TextureFormat::R16Float => todo!(),
            TextureFormat::RG16Float => todo!(),
            TextureFormat::RGBA16Float => todo!(),
            TextureFormat::R32Float => todo!(),
            TextureFormat::RG32Float => todo!(),
            TextureFormat::RGBA32Float => todo!(),
            TextureFormat::BC1RGBAUnorm => todo!(),
            TextureFormat::BC3RGBAUnorm => todo!(),
            TextureFormat::BC4RUnorm => todo!(),
            TextureFormat::BC5RGUnorm => todo!(),
            TextureFormat::BC6HRGBFloat => todo!(),
            TextureFormat::BC7RGBAUnorm => todo!(),
            TextureFormat::BC7RGBASRGB => todo!(),
            TextureFormat::ASTC4x4RGBAUnorm => todo!(),
            TextureFormat::ETC2RGB8Unorm => todo!(),
            TextureFormat::ETC2RGBA8Unorm => todo!(),
            TextureFormat::EACR11Unorm => todo!(),
            TextureFormat::EACRG11Unorm => todo!(),
        }
    }
}
struct GPUTexture {
    texture_type: TextureType,
    dimensions: [u32; 3],
    format: TextureFormat,
    mip_count: u32,
    layer_count: u32,
    sample_count: u32,
    image: vk::Image,
    view_cache: Mutex<HashMap<TextureViewDesc, vk::ImageView>>,
    gpu: Weak<GPU>,
}
impl GPUTexture {
    fn get_view(&self, view_desc: &TextureViewDesc) -> vk::ImageView {
        self.view_cache
            .lock()
            .unwrap()
            .entry(view_desc.clone())
            .or_insert_with(|| {
                let gpu = self.gpu.upgrade().unwrap();
                assert!(
                    view_desc.base_layer as u32 + view_desc.layer_count as u32 <= self.layer_count
                );
                assert!(view_desc.base_mip as u32 + view_desc.mip_count as u32 <= self.mip_count);
                unsafe {
                    gpu.device
                        .create_image_view(
                            &vk::ImageViewCreateInfo::default()
                                .image(self.image.clone())
                                .view_type(self.texture_type.to_view_vk())
                                .format(self.format.to_vk())
                                .subresource_range(vk::ImageSubresourceRange {
                                    aspect_mask: match self.format {
                                        TextureFormat::D16Unorm
                                        | TextureFormat::D16UnormS8Uint
                                        | TextureFormat::D24UnormPack32
                                        | TextureFormat::D24UnormS8Uint
                                        | TextureFormat::D32Float => vk::ImageAspectFlags::DEPTH,
                                        _ => vk::ImageAspectFlags::COLOR,
                                    },
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
        if let Some(gpu) = self.gpu.upgrade() {
            for (_, view) in self.view_cache.get_mut().unwrap() {
                unsafe {
                    gpu.device.destroy_image_view(*view, None);
                }
            }
            unsafe {
                gpu.device.destroy_image(self.image, None);
            }
        }
    }
}
pub struct GPUBuffer {
    allocation: Option<Allocation>,
    buffer: vk::Buffer,
    gpu_pointer: vk::DeviceAddress,
    gpu: Weak<GPU>,
}
impl GPUBuffer {
    pub fn gpu(&self) -> vk::DeviceAddress {
        self.gpu_pointer
    }
    pub fn upload<T: bytemuck::Pod>(&mut self, slice: &[T]) {
        let bytes = bytemuck::cast_slice::<T, u8>(slice);
        let allocation = self.allocation.as_mut().unwrap();
        if bytes.len() > allocation.size() as usize {
            panic!("too big, buffer size is {}", allocation.size());
        }
        let cpu_data = &mut allocation
            .mapped_slice_mut()
            .expect("buffer not mapped on cpu")[..bytes.len()];
        cpu_data.copy_from_slice(bytes);
    }
}
impl Drop for GPUBuffer {
    fn drop(&mut self) {
        if let Some(gpu) = self.gpu.upgrade() {
            gpu.allocator
                .lock()
                .unwrap()
                .free(self.allocation.take().unwrap())
                .unwrap();
            unsafe {
                gpu.device.destroy_buffer(self.buffer, None);
            }
        }
    }
}
pub struct GPUSemaphore {
    semaphore: vk::Semaphore,
    gpu: Weak<GPU>,
}
impl GPUSemaphore {
    fn create_raw(device: &ash::Device, gpu: &Weak<GPU>, value: u64) -> GPUSemaphore {
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
        GPUSemaphore {
            semaphore,
            gpu: gpu.clone(),
        }
    }
    fn create_binary(device: &ash::Device, gpu: &Weak<GPU>) -> GPUSemaphore {
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
        GPUSemaphore {
            semaphore,
            gpu: gpu.clone(),
        }
    }
    pub fn get_value(&self) -> u64 {
        let gpu = self.gpu.upgrade().unwrap();
        unsafe {
            gpu.device
                .get_semaphore_counter_value(self.semaphore.clone())
                .unwrap()
        }
    }
    pub fn wait(&self, value: u64) {
        let gpu = self.gpu.upgrade().unwrap();
        unsafe {
            gpu.device
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
        if let Some(gpu) = self.gpu.upgrade() {
            unsafe {
                gpu.device.destroy_semaphore(self.semaphore, None);
            }
        }
    }
}
pub struct GPUShader {
    shader: vk::ShaderEXT,
    kind: GPUShaderKind,
    gpu: Weak<GPU>,
}
#[derive(Clone, Copy)]
pub enum GPUShaderKind {
    Vertex,
    Fragment,
    Compute { workgroup_size: [f32; 3] },
}
impl Drop for GPUShader {
    fn drop(&mut self) {
        if let Some(gpu) = self.gpu.upgrade() {
            unsafe {
                gpu.shader_object_device.destroy_shader(self.shader, None);
            }
        }
    }
}
pub struct GPUSwapchain {
    swapchain: vk::SwapchainKHR,
    textures: Vec<Arc<GPUTexture>>,
    gpu: Weak<GPU>,
    present_semaphores: Vec<GPUSemaphore>,
    next_swap_image: u32,
}
impl GPUSwapchain {
    pub fn acquire_next(&mut self) -> Arc<GPUTexture> {
        unsafe {
            let gpu = self.gpu.upgrade().unwrap();
            let fence = gpu
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .unwrap();
            let (index, suboptimal) = gpu
                .swapchain_device
                .acquire_next_image(self.swapchain, u64::MAX, vk::Semaphore::null(), fence)
                .unwrap();
            self.next_swap_image = index;
            gpu.device.wait_for_fences(&[fence], true, u64::MAX);
            gpu.device.destroy_fence(fence, None);
            let mut command_buffer = gpu.begin_commands(QueueKind::Main);
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
            gpu.device.cmd_pipeline_barrier2(
                command_buffer.command_buffer,
                &vk::DependencyInfo::default().image_memory_barriers(&[image_barrier]),
            );
            command_buffer.submit();
            self.textures[index as usize].clone()
        }
    }
    pub fn present(
        &self,
        queue: QueueKind,
        wait_semaphore: &GPUSemaphore,
        wait_value: u64,
    ) -> SwapchainPresentResult {
        let gpu = self.gpu.upgrade().unwrap();
        let mut command_buffer = gpu.begin_commands(queue);
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
            gpu.device.cmd_pipeline_barrier2(
                command_buffer.command_buffer,
                &vk::DependencyInfo::default().image_memory_barriers(&[image_barrier]),
            )
        };
        command_buffer.add_wait(wait_semaphore, wait_value);
        command_buffer.add_signal(&self.present_semaphores[self.next_swap_image as usize], 0);
        command_buffer.submit();
        let queue_lock = gpu.command_buffer_counters.lock().unwrap();
        unsafe {
            let suboptimal = gpu
                .swapchain_device
                .queue_present(
                    gpu.queues[queue as usize],
                    &vk::PresentInfoKHR::default()
                        .image_indices(&[self.next_swap_image])
                        .swapchains(&[self.swapchain])
                        .wait_semaphores(&[
                            self.present_semaphores[self.next_swap_image as usize].semaphore
                        ]),
                )
                .unwrap();
            match suboptimal {
                true => SwapchainPresentResult::Suboptimal,
                false => SwapchainPresentResult::Ok,
            }
        }
    }
}
pub enum SwapchainPresentResult {
    Ok,
    Suboptimal,
}
impl Drop for GPUSwapchain {
    fn drop(&mut self) {
        if let Some(gpu) = self.gpu.upgrade() {
            unsafe {
                gpu.swapchain_device.destroy_swapchain(self.swapchain, None);
            }
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
pub enum MemoryKind {
    Default,
    GPU,
    Readback,
}
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct GraphicsShaderPushConstants {
    vertex_data: vk::DeviceAddress,
    fragment_data: vk::DeviceAddress,
    indirect_data: vk::DeviceAddress,
}
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct ComputeShaderPushConstants {
    compute_data: vk::DeviceAddress,
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
