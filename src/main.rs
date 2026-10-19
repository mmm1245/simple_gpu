use std::{
    alloc::Layout,
    cell::RefCell,
    collections::{BinaryHeap, HashMap, HashSet},
    ffi::{CStr, CString},
    marker::PhantomData,
    ops::{Add, Rem, Sub},
    sync::{Arc, Mutex, Weak},
    time::Instant,
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
    raw_window_handle::{
        DisplayHandle, HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle,
        WindowHandle,
    },
    window::WindowId,
};

#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct Vertex {
    pos: [f32; 3],
    color: [f32; 3],
}

#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct DrawConstants {
    vertex_data: u64,
    time: f32,
    _pad: u32,
}

#[derive(Default)]
struct App {
    data: Option<AppData>,
}
struct AppData {
    window: winit::window::Window,
    gpu: Arc<GPU>,
    swapchain: GPUSwapchain,
    semaphore: GPUSemaphore,
    next_frame: u64,
    shaders: GraphicsShaders,
    frame_arenas: [GPUArena; 3],
    start_time: Instant,
    desc_heap: GPUDescriptorHeap,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.data.is_some() {
            return;
        }
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
        let gpu = GPU::init(Default::default()).unwrap();
        let semaphore = gpu.create_semaphore(0);
        let swapchain = gpu.create_swapchain(
            window.display_handle().unwrap().as_raw(),
            window.window_handle().unwrap().as_raw(),
        );
        let shader = include_bytes!("shader.spv");
        let shaders = gpu
            .create_shaders_graphic_spirv(shader, "vertexMain", "fragmentMain", &Default::default())
            .unwrap();
        let frame_arenas = std::array::from_fn(|_| gpu.allocate_arena(1024 * 1024 * 8));
        let mut desc_heap = gpu.create_descriptor_heap();
        desc_heap.set_samplers(
            0,
            &[TextureSampler {
                ..Default::default()
            }],
        );
        let mut upload_semaphore = gpu.create_semaphore(0);
        let image = image::open("image.png").unwrap().into_rgba8();
        let mut image_buffer = image
            .pixels()
            .map(|rgba| rgba.0.into_iter())
            .flatten()
            .collect::<Vec<_>>();
        let mut texture = gpu.create_texture(
            TextureDesc {
                dimensions: [image.width(), image.height(), 1],
                usage: TextureUsage::SAMPLED,
                ..Default::default()
            },
            QueueKind::Main,
            &upload_semaphore,
            1,
        );
        let mut upload_arena = gpu.allocate_arena(4 * 1024 * 1024);
        let mut upload_buffer = gpu.begin_commands(QueueKind::Main);
        upload_buffer.add_wait(&upload_semaphore, 1);
        upload_buffer.add_signal(&upload_semaphore, 2);
        upload_buffer.copy_to_texture(
            &texture,
            TextureRegion::default(),
            upload_arena.alloc(image_buffer.as_slice()),
        );
        upload_buffer.submit();
        upload_semaphore.wait(2);
        desc_heap.set_textures(1, &[(&texture, TextureViewDesc::default())]);

        self.data = Some(AppData {
            frame_arenas,
            window,
            gpu,
            swapchain,
            semaphore,
            next_frame: 1,
            shaders,
            start_time: Instant::now(),
            desc_heap,
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
                if data.next_frame > 3 {
                    data.semaphore.wait(data.next_frame - 3);
                }

                let frame_arena = &mut data.frame_arenas[data.next_frame as usize % 3];
                frame_arena.clear();
                let time = data.start_time.elapsed().as_secs_f32();
                let mod_time = time.rem(2.).sub(1.).abs();
                let vertices = frame_arena.alloc(&[
                    Vertex {
                        pos: [0., 0., 0.],
                        color: [mod_time, 0., 0.],
                    },
                    Vertex {
                        pos: [1., 0., 0.],
                        color: [0., mod_time, 0.],
                    },
                    Vertex {
                        pos: [1., 1., 0.],
                        color: [0., 0., mod_time],
                    },
                ]);

                let mut commands = data.gpu.begin_commands(QueueKind::Main);
                commands.bind_descriptor_heap(&data.desc_heap);
                let target = data.swapchain.acquire_next();
                {
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
                    render_pass.set_shaders(&data.shaders);
                    render_pass.draw(
                        DrawConstants {
                            vertex_data: vertices.gpu(),
                            time,
                            _pad: 0,
                        },
                        3,
                        1,
                        Default::default(),
                    );
                }
                commands.add_signal(&data.semaphore, data.next_frame);
                commands.submit();

                match data
                    .swapchain
                    .present(QueueKind::Main, &data.semaphore, data.next_frame)
                {
                    SwapchainPresentResult::Ok => {}
                    SwapchainPresentResult::Suboptimal => {
                        data.swapchain.resize();
                    }
                }

                data.next_frame += 1;
                //println!("frame rendered {}", self.bump_value);

                data.window.request_redraw();
            }
            WindowEvent::Resized(new_size) => {}
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
            let mut enabled_extension_instance_names = vec![];
            enabled_extension_instance_names.push(ash::khr::surface::NAME.as_ptr());
            enabled_extension_instance_names.push(ash::khr::xlib_surface::NAME.as_ptr());
            let mut enabled_layer_names = Vec::new();
            enabled_layer_names.push("VK_LAYER_KHRONOS_validation\0".as_ptr() as *const i8);
            let create_info = vk::InstanceCreateInfo {
                p_application_info: &app_info,
                ..Default::default()
            }
            .enabled_extension_names(&enabled_extension_instance_names)
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

            let common_pipeline_graphics_layout = unsafe {
                device
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
                    .unwrap()
            };
            let common_pipeline_compute_layout = unsafe {
                device
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
        buffer.upload(0, data);
        buffer
    }
    pub fn create_semaphore(&self, value: u64) -> GPUSemaphore {
        GPUSemaphore::create_raw(&self.device, &self.weak, value)
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
                GPUShaderKind::Vertex,
                specializations,
            )?,
            fragment: self.create_shader_spirv(
                code,
                entry_point_fragment,
                GPUShaderKind::Fragment,
                specializations,
            )?,
        })
    }
    pub fn create_shader_spirv(
        &self,
        code: &[u8],
        entry_point: &str,
        kind: GPUShaderKind,
        specializations: &HashMap<u32, u32>,
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
                size: self.settings.max_push_constant_size as u32,
                stage_flags: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
            },
            GPUShaderKind::Compute { .. } => vk::PushConstantRange {
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
    pub fn create_texture(
        &self,
        desc: TextureDesc,
        queue: QueueKind,
        semaphore: &GPUSemaphore,
        wait_value: u64,
    ) -> GPUTexture {
        let mut usage = vk::ImageUsageFlags::TRANSFER_DST;
        if desc.usage & TextureUsage::TRANSFER_SRC != 0 {
            usage |= vk::ImageUsageFlags::TRANSFER_SRC;
        }
        if desc.usage & TextureUsage::SAMPLED != 0 {
            usage |= vk::ImageUsageFlags::SAMPLED;
        }
        if desc.usage & TextureUsage::STORAGE != 0 {
            usage |= vk::ImageUsageFlags::STORAGE;
        }
        if desc.usage & TextureUsage::COLOR_ATTACHMENT != 0 {
            usage |= vk::ImageUsageFlags::COLOR_ATTACHMENT;
        }
        if desc.usage & TextureUsage::DEPTH_STENCIL_ATTACHMENT != 0 {
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
                aspect_mask: desc.format.to_vk_aspect_flags(),
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
            format: desc.format,
            mip_count: desc.mip_count,
            layer_count: desc.layer_count,
            sample_count: desc.sample_count,
            image,
            allocation: Some(allocation),
            view_cache: Default::default(),
            gpu: self.weak.clone(),
        }
    }
    pub fn create_swapchain(
        &self,
        display_handle: RawDisplayHandle,
        window_handle: RawWindowHandle,
    ) -> GPUSwapchain {
        unsafe {
            let surface = ash_window::create_surface(
                &self.entry,
                &self.instance,
                display_handle,
                window_handle,
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
                        allocation: None,
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
                display_handle,
                window_handle,
            }
        }
    }
    pub fn allocate_arena(&self, block_size: usize) -> GPUArena {
        let memory = MemoryKind::Default;
        GPUArena {
            gpu: self.weak.clone(),
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
            gpu: self.weak.clone(),
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
    pub usage: u8,
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
            usage: 0,
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
            let stencil_attachment = desc
                .stencil_attachment
                .map(|stencil_attachment| create_vk_attachment(&stencil_attachment));
            if let Some(stencil_attachment) = &stencil_attachment {
                info = info.stencil_attachment(stencil_attachment);
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
    pub fn copy_to_texture(
        &mut self,
        texture: &GPUTexture,
        texture_region: TextureRegion,
        src: GPUPointer,
    ) {
        let gpu = self.gpu.upgrade().unwrap();
        let aspect_mask = texture.format.to_vk_aspect_flags();
        let is_compressed = match texture.format {
            TextureFormat::BC1RGBAUnorm
            | TextureFormat::BC3RGBAUnorm
            | TextureFormat::BC7RGBAUnorm
            | TextureFormat::ASTC4x4RGBAUnorm
            | TextureFormat::ETC2RGB8Unorm
            | TextureFormat::ETC2RGBA8Unorm
            | TextureFormat::EACR11Unorm
            | TextureFormat::EACRG11Unorm => true,
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
            gpu.device.cmd_copy_buffer_to_image(
                self.command_buffer,
                src.buffer,
                texture.image,
                vk::ImageLayout::GENERAL,
                std::slice::from_ref(&copy_info),
            )
        };
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
    pub fn bind_descriptor_heap(&self, heap: &GPUDescriptorHeap) {
        let gpu = self.gpu.upgrade().unwrap();
        let descriptor_sets = [heap.textures, heap.textures_rw, heap.samplers];
        unsafe {
            gpu.device.cmd_bind_descriptor_sets(
                self.command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                gpu.common_pipeline_graphics_layout,
                0,
                &descriptor_sets,
                &[],
            );
            gpu.device.cmd_bind_descriptor_sets(
                self.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                gpu.common_pipeline_compute_layout,
                0,
                &descriptor_sets,
                &[],
            );
        };
    }
}
impl Drop for GPUCommandBuffer {
    fn drop(&mut self) {
        if !self.submitted {
            println!("buffer not submitted")
        }
    }
}
pub struct GPURenderPass<'a> {
    command_buffer: &'a mut GPUCommandBuffer,
}
impl GPURenderPass<'_> {
    pub fn set_shaders_individual(&mut self, vertex: &GPUShader, fragment: &GPUShader) {
        let gpu = self.command_buffer.gpu.upgrade().unwrap();
        unsafe {
            gpu.shader_object_device.cmd_bind_shaders(
                self.command_buffer.command_buffer,
                &[vk::ShaderStageFlags::VERTEX, vk::ShaderStageFlags::FRAGMENT],
                &[vertex.shader, fragment.shader],
            )
        };
    }
    pub fn set_shaders(&mut self, shaders: &GraphicsShaders) {
        self.set_shaders_individual(&shaders.vertex, &shaders.fragment);
    }
    pub fn draw<D: bytemuck::Pod>(
        &mut self,
        draw_data: D,
        vertex_count: u32,
        instance_count: u32,
        draw_options: DrawOptions,
    ) {
        let gpu = self.command_buffer.gpu.upgrade().unwrap();
        assert!(
            std::mem::size_of::<D>() <= gpu.settings.max_push_constant_size,
            "draw data size greater than set size"
        );
        unsafe {
            gpu.device.cmd_push_constants(
                self.command_buffer.command_buffer,
                gpu.common_pipeline_graphics_layout,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                0,
                bytemuck::bytes_of(&draw_data),
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
pub struct DrawOptions {}
impl Default for DrawOptions {
    fn default() -> Self {
        DrawOptions {}
    }
}
pub struct RenderPassDesc<'a> {
    render_area_offset: [i32; 2],
    render_area_size: [u32; 2],
    layer_count: u32,
    view_mask: u32,
    color_attachments: &'a [RenderPassAttachment<'a, [f32; 4]>],
    depth_attachment: Option<RenderPassAttachment<'a, f32>>,
    stencil_attachment: Option<RenderPassAttachment<'a, u32>>,
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
impl IntoClearValue for u32 {
    fn into_clear_value(self, format: TextureFormat) -> vk::ClearValue {
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
    fn to_vk_aspect_flags(self) -> vk::ImageAspectFlags {
        match self {
            TextureFormat::D16Unorm
            | TextureFormat::D16UnormS8Uint
            | TextureFormat::D24UnormPack32
            | TextureFormat::D24UnormS8Uint
            | TextureFormat::D32Float => vk::ImageAspectFlags::DEPTH,
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
    allocation: Option<Allocation>,
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
                                    aspect_mask: self.format.to_vk_aspect_flags(),
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
    textures: Vec<Arc<GPUTexture>>,
    gpu: Weak<GPU>,
    present_semaphores: Vec<GPUSemaphore>,
    swapchain: vk::SwapchainKHR,
    next_swap_image: u32,
    display_handle: RawDisplayHandle,
    window_handle: RawWindowHandle,
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
    pub fn resize(&mut self) {
        let gpu = self.gpu.upgrade().unwrap();
        *self = gpu.create_swapchain(self.display_handle, self.window_handle);
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
    gpu: Weak<GPU>,
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
            let gpu = self.gpu.upgrade().unwrap();
            self.blocks.push(gpu.allocate_buffer(
                Layout::from_size_align(self.block_size, 16).unwrap(),
                self.memory,
            ));
            self.block_index += 1;
            let block = self.blocks.last_mut().unwrap();
            block.upload(0, slice);
            self.block_offset = layout.size();
            GPUPointer {
                buffer: block.buffer,
                offset: 0,
                pointer: block.gpu(),
            }
        } else {
            let block = &mut self.blocks[self.block_index];
            block.upload(next_addr_align, slice);
            self.block_offset = next_addr_align + layout.size();
            GPUPointer {
                buffer: block.buffer,
                offset: next_addr_align as u64,
                pointer: block.gpu() + next_addr_align as u64,
            }
        }
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
    gpu: Weak<GPU>,
    pool: vk::DescriptorPool,
    textures: vk::DescriptorSet,
    textures_rw: vk::DescriptorSet,
    samplers: vk::DescriptorSet,
    sampler_cache: HashMap<TextureSampler, vk::Sampler>,
}
impl GPUDescriptorHeap {
    pub fn set_textures(&mut self, offset: usize, textures: &[(&GPUTexture, TextureViewDesc)]) {
        let gpu = self.gpu.upgrade().unwrap();
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
        unsafe { gpu.device.update_descriptor_sets(&[write], &[]) };
    }
    pub fn set_textures_rw(&mut self, offset: usize, textures: &[(&GPUTexture, TextureViewDesc)]) {
        let gpu = self.gpu.upgrade().unwrap();
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
        unsafe { gpu.device.update_descriptor_sets(&[write], &[]) };
    }
    pub fn set_samplers(&mut self, offset: usize, samplers: &[TextureSampler]) {
        let gpu = self.gpu.upgrade().unwrap();
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
                    unsafe { gpu.device.create_sampler(&create_info, None).unwrap() }
                }),
            ));
        }
        let write = vk::WriteDescriptorSet::default()
            .dst_set(self.samplers)
            .dst_binding(0)
            .dst_array_element(offset as u32)
            .image_info(&infos)
            .descriptor_type(vk::DescriptorType::SAMPLER);
        unsafe { gpu.device.update_descriptor_sets(&[write], &[]) };
    }
}
impl Drop for GPUDescriptorHeap {
    fn drop(&mut self) {
        if let Some(gpu) = self.gpu.upgrade() {
            unsafe {
                gpu.device
                    .free_descriptor_sets(
                        self.pool,
                        &[self.textures, self.textures_rw, self.samplers],
                    )
                    .unwrap();
                gpu.device.destroy_descriptor_pool(self.pool, None);
            };
        }
    }
}
#[derive(Hash, Clone, Copy, PartialEq, Eq)]
pub struct TextureSampler {
    min: SamplerFilter,
    mag: SamplerFilter,
    mip: SamplerFilter,
    u: SamplerAddressMode,
    v: SamplerAddressMode,
    w: SamplerAddressMode,
    max_anisotropy: Option<u8>, //should be float, but then not eq
    border: SamplerBorder,
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
struct GPUSettings {
    desc_heap_max_textures: usize,
    desc_heap_max_samplers: usize,
    max_push_constant_size: usize,
}
impl Default for GPUSettings {
    fn default() -> Self {
        GPUSettings {
            desc_heap_max_textures: 32768,
            desc_heap_max_samplers: 32,
            max_push_constant_size: 128,
        }
    }
}
struct TextureRegion {
    offset: [i32; 3],
    size: [u32; 3],
    mip_level: u32,
    base_layer: u32,
    layer_count: u32,
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
}
impl GPUPointer {
    pub fn gpu(self) -> u64 {
        self.pointer
    }
}
impl Into<GPUPointer> for &GPUBuffer {
    fn into(self) -> GPUPointer {
        GPUPointer {
            buffer: self.buffer,
            offset: self.allocation.as_ref().unwrap().offset(),
            pointer: self.gpu_pointer,
        }
    }
}
pub mod TextureUsage {
    pub const SAMPLED: u8 = 1;
    pub const STORAGE: u8 = 2;
    pub const TRANSFER_SRC: u8 = 4;
    pub const COLOR_ATTACHMENT: u8 = 8;
    pub const DEPTH_STENCIL_ATTACHMENT: u8 = 16;
}
