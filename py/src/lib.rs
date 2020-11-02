use numpy::{DataType, PyArray, PyArrayDescr, PyArrayDyn};
use pyo3::prelude::*;
use std::convert::From;

#[pyclass]
struct Device {
    device: alkomp::Device,
}

#[pyclass]
struct GPUDataNumpy {
    pub data: alkomp::GPUData<[u8]>,
    dtype: DataType,
    shape: Vec<usize>,
}

#[pymethods]
impl Device {
    #[new]
    fn new(idx: usize) -> Self {
        Device {
            device: alkomp::Device::new(idx),
        }
    }

    fn to_device<'py>(&mut self, _py: Python<'py>, arr: &'py PyAny) -> GPUDataNumpy {
        let dtype: &PyArrayDescr = arr.getattr("dtype").unwrap().downcast().unwrap();
        let dtype = dtype.get_datatype().unwrap();

        let shape: Option<&[usize]>;

        let bytes: &'py [u8] = match dtype {
            DataType::Float32 => {
                let slice: &PyArrayDyn<f32> = arr.downcast().unwrap();
                shape = Some(slice.shape());
                let slice = unsafe { slice.as_slice().unwrap() };
                bytemuck::cast_slice(slice)
            }
            DataType::Uint32 => {
                let slice: &PyArrayDyn<u32> = arr.downcast().unwrap();
                shape = Some(slice.shape());
                let slice = unsafe { slice.as_slice().unwrap() };
                bytemuck::cast_slice(slice)
            }
            _ => panic!(""),
        };

        let shape = Vec::from(shape.unwrap());

        GPUDataNumpy {
            data: self.device.to_device(bytes),
            dtype,
            shape,
        }
    }

    fn run<'py>(
        &mut self,
        entry: String,
        shader: Vec<u32>,
        workspace: (u32, u32, u32),
        layers: Vec<&PyCell<GPUDataNumpy>>,
    ) {
        let mut args = alkomp::ParamsBuilder::new();

        let layers = unsafe {
            layers
                .iter()
                .map(|x| Some(&x.try_borrow_unguarded().unwrap().data))
        };

        for l in layers {
            args = args.param(l);
        }

        let args = args.build(Some(0));

        let compute = self
            .device
            .compile(entry.as_str(), &shader, &args.0)
            .unwrap();

        self.device.call(compute, workspace, &args.1);
    }

    fn get<'py>(
        &mut self,
        py: Python<'py>,
        data: &'py PyCell<GPUDataNumpy>,
    ) -> &'py PyArrayDyn<DataType> {
        let data = data.borrow();

        let dtype = &data.dtype;
        let shape = &data.shape;
        let data = futures::executor::block_on(self.device.get(&data.data)).unwrap();

        //TODO: this needs to copy data in order to reshape?
        let data: &PyArrayDyn<DataType> = match dtype {
            DataType::Float32 => {
                let d: Vec<f32> = data[..data.len() - 1]
                    .chunks_exact(std::mem::size_of::<f32>())
                    .map(|b| bytemuck::from_bytes::<f32>(b).clone())
                    .collect();
                let x: &PyArrayDyn<DataType> = PyArray::from_slice(py, d.as_slice())
                    .reshape(shape.as_slice())
                    .unwrap()
                    .downcast()
                    .unwrap();
                x
            }
            DataType::Uint32 => {
                let d: Vec<u32> = data[..data.len() - 1]
                    .chunks_exact(std::mem::size_of::<u32>())
                    .map(|b| bytemuck::from_bytes::<u32>(b).clone())
                    .collect();
                let x: &PyArrayDyn<DataType> = PyArray::from_slice(py, d.as_slice())
                    .reshape(shape.as_slice())
                    .unwrap()
                    .downcast()
                    .unwrap();
                x
            }
            _ => panic!(""),
        };

        data
    }
}

#[pymodule]
fn alkompy(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<Device>().unwrap();

    #[pyfn(m, "query")]
    fn query<'py>(_py: Python<'_>) -> PyResult<Vec<(String, String)>> {
        let res = alkomp::query()
            .iter()
            .map(|x| (x.name(), x.backend().to_string()))
            .collect();
        Ok(res)
    }

    #[pyfn(m, "compile_glsl")]
    fn compile_glsl<'py>(_py: Python<'_>, code: String) -> PyResult<Vec<u32>> {
        let mut spirv = alkomp::glslhelper::GLSLCompile::new(code.as_str());
        let shader = spirv.compile("main").unwrap();
        Ok(shader)
    }

    Ok(())
}
