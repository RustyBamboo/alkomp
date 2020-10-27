use colored::*;
use futures::executor::block_on;
use std::fmt;
use std::marker::PhantomData;
use wgpu::util::DeviceExt;
use std::convert::TryInto;


pub fn query() -> bool {
    println!("{}", "Query compatible devices".bold());
    let instance = wgpu::Instance::new(wgpu::BackendBit::PRIMARY);
    let adapter = instance.enumerate_adapters(wgpu::BackendBit::PRIMARY);
    let mut found = false;
    for a in adapter {
        // println!("{:?}", a.get_info());
        let device_info = DeviceInfo { info: a.get_info() };
        println!(
            "\t- {}\t{}",
            device_info.name().bright_white(),
            device_info.backend().green()
        );
        found = true;
    }
    found
}

pub struct Device {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub info: Option<DeviceInfo>,
}

pub struct GPU<T: ?Sized> {
    pub staging_buffer: wgpu::Buffer,
    pub storage_buffer: wgpu::Buffer,
    pub size: u64,
    pub phantom: PhantomData<T>
}

impl Device {
    pub fn new(device_index: usize) -> Self {
        let instance = wgpu::Instance::new(wgpu::BackendBit::PRIMARY);
        let mut adapter = instance.enumerate_adapters(wgpu::BackendBit::PRIMARY);
        let adapter = adapter.nth(device_index).unwrap();
        let (device, queue) = block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                features: wgpu::Features::MAPPABLE_PRIMARY_BUFFERS,
                limits: wgpu::Limits::default(),
                shader_validation: false,
            },
            None,
        ))
        .unwrap();
        let info = adapter.get_info().clone();
        let info = DeviceInfo { info };

        println!(
            "{} {}\t{}",
            "Selected:".on_purple(),
            info.name(),
            info.backend().green()
        );

        Device {
            device,
            queue,
            info: Some(info),
        }
    }

    pub fn to_device_t<T: bytemuck::Pod> (&mut self, data: &[T]) -> GPU<[T]>{
        let bytes = bytemuck::cast_slice(data);

        let staging_buffer = self
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Staging Buffer"),
            contents: &bytes,
            usage: wgpu::BufferUsage::MAP_READ | wgpu::BufferUsage::COPY_DST | wgpu::BufferUsage::COPY_SRC,
        });

        let storage_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: bytes.len() as u64,
            usage: wgpu::BufferUsage::STORAGE
            | wgpu::BufferUsage::COPY_DST
            | wgpu::BufferUsage::COPY_SRC,
            mapped_at_creation: false,
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(&staging_buffer, 0, &storage_buffer, 0, bytes.len() as u64);

        self.queue.submit(Some(encoder.finish()));

        GPU {
            staging_buffer,
            storage_buffer,
            size: bytes.len() as u64,
            phantom: PhantomData,
        }
    }

    pub async fn get<T>(&mut self, gpu: &GPU<[T]>) -> Option<Box<[T]>> 
        where T: bytemuck::Pod
    {
        let mut encoder = self
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_buffer_to_buffer(
        &gpu.storage_buffer,
        0,
        &gpu.staging_buffer,
        0,
        gpu.size,
    );
    self.queue.submit(Some(encoder.finish()));

    let buffer_slice = gpu.staging_buffer.slice(0..);
    let buffer_future = buffer_slice.map_async(wgpu::MapMode::Read);

    self.device.poll(wgpu::Maintain::Wait);


    
        // Gets contents of buffer
        if let Ok(()) = buffer_future.await {
            let data = buffer_slice.get_mapped_range();
            let result = data
                .chunks_exact(std::mem::size_of::<T>())
                .map(|b| bytemuck::from_bytes::<T>(b).clone()).collect();
            return Some(result);
        }
        None

    }


    pub fn to_device_matrix<T: bytemuck::Pod, S:ndarray::Dimension>(&mut self, data: &ndarray::Array<T,S>) -> (wgpu::Buffer, wgpu::Buffer, u64) {
        let shape = data.shape();
        
        // TODO: generale shape
        let mut size = shape[0] * shape[1] * std::mem::size_of::<T>();
        // size += shape.len() * std::mem::size_of::<u32>();


        let shape: &[u8] = bytemuck::cast_slice(shape);
        let mut shape = Vec::from(shape);
        
        let arr = data.as_slice().unwrap();
        let arr: &[u8] = bytemuck::cast_slice(arr);
        let arr = Vec::from(arr);

        let size = size as u64;

        shape.extend(arr);

        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsage::MAP_READ | wgpu::BufferUsage::COPY_DST | wgpu::BufferUsage::COPY_SRC,
            mapped_at_creation: false,
        });
        let storage_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Storage Buffer"),
                contents: &shape,
                usage: wgpu::BufferUsage::STORAGE
                    | wgpu::BufferUsage::COPY_DST
                    | wgpu::BufferUsage::COPY_SRC,
            });
        (staging_buffer, storage_buffer, size)
        // let arr = shape.extend());

    }

    pub fn to_device<T: bytemuck::Pod>(&mut self, data: &[T]) -> (wgpu::Buffer, wgpu::Buffer, u64) {
        let size = data.len() * std::mem::size_of::<T>();
        let size = size as u64;

        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsage::MAP_READ | wgpu::BufferUsage::COPY_DST,
            mapped_at_creation: false,
        });
        let storage_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Storage Buffer"),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsage::STORAGE
                    | wgpu::BufferUsage::COPY_DST
                    | wgpu::BufferUsage::COPY_SRC,
            });
        (staging_buffer, storage_buffer, size)
    }
    pub fn bind_groups(
        &mut self,
        gpu_buffer: &wgpu::Buffer,
    ) -> (wgpu::BindGroupLayout, wgpu::BindGroup) {
        let bind_group_layout =
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: None,
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStage::COMPUTE,
                        ty: wgpu::BindingType::StorageBuffer {
                            dynamic: false,
                            readonly: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }],
                });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(gpu_buffer.slice(0..)),
            }],
        });
        (bind_group_layout, bind_group)
    }
    pub fn create_pipeline(&mut self, bind_group_layout: &wgpu::BindGroupLayout, cs_module: &wgpu::ShaderModule) -> wgpu::ComputePipeline{
        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
    
        let compute_pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            compute_stage: wgpu::ProgrammableStageDescriptor {
                module: &cs_module,
                entry_point: "main",
            },
        });
        compute_pipeline
    }
    pub fn create_encoder(&mut self) -> wgpu::CommandEncoder {
        self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None })
    }
    pub fn run(&mut self, encoder: wgpu::CommandEncoder) {
        self.queue.submit(Some(encoder.finish()))
    }
}

pub struct DeviceInfo {
    pub info: wgpu::AdapterInfo,
}

impl DeviceInfo {
    pub fn name(&self) -> String {
        self.info.name.clone()
    }
    pub fn device_type(&self) -> wgpu::DeviceType {
        self.info.device_type.clone()
    }
    pub fn backend(&self) -> &str {
        match self.info.backend {
            wgpu::Backend::Vulkan => "Vulkan",
            wgpu::Backend::Metal => "Metal",
            wgpu::Backend::Dx11 => "Dx11",
            wgpu::Backend::Dx12 => "Dx12",
            wgpu::Backend::BrowserWebGpu => "Browse",
            _ => "Unknown",
        }
    }
    pub fn vendor_id(&self) -> usize {
        self.info.vendor
    }
    pub fn device_id(&self) -> usize {
        self.info.device
    }
}

impl fmt::Debug for DeviceInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{{ name: {:?}, vendor_id: {:?}, device_id: {:?}, device_type: {:?} }}",
            self.name(),
            self.vendor_id(),
            self.device_id(),
            self.device_type()
        )
    }
}
