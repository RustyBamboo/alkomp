use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::prelude::*;

#[pyclass]
struct Device {
    device: alkomp::Device,
    data: Vec<alkomp::GPUData<[f32]>>,
    size: Vec<alkomp::GPUData<[u32]>>,
}

#[pymethods]
impl Device {
    #[new]
    fn new(idx: usize) -> Self {
        Device { device: alkomp::Device::new(idx), data: Vec::new(), size: Vec::new()}
    }


    fn to_device<'py>(&mut self, _py: Python<'py>, arr: PyReadonlyArrayDyn<f32>) -> PyResult<()> {
        let arr = arr.as_array();
        let s: Vec<u32> = arr.shape().iter().map(|x| *x as u32).collect();

        let size_gpu = self.device.to_device(s.as_slice());
        let data_gpu = self.device.to_device(&arr.as_slice().unwrap());

        self.size.push(size_gpu);
        self.data.push(data_gpu);
    
        Ok(())
    }

    fn run<'py>(&mut self, py: Python<'py>, entry: String, code: String, workspace: (u32, u32, u32)) -> Vec<&'py PyArrayDyn<f32>> {
        let mut args = alkomp::ParamsBuilder::new();

        for (d,s) in self.data.iter().zip(self.size.iter()) {
            args = args.param(Some(d)).param(Some(s));
        }

        let args = args.build(Some(0));

        let compute = self.device.compile(entry.as_str(), code.as_str(), args.0).unwrap();

        self.device.call(compute, workspace, args.1);

        let mut result = Vec::new();
        for (s,d) in self.size.iter().zip(self.data.iter()) {
            let data = futures::executor::block_on(self.device.get(&d)).unwrap();
            let shape = futures::executor::block_on(self.device.get(&s)).unwrap();
    
            let shape = &shape[0..shape.len() - 1];
            let data = &data[0..data.len() - 1];
    
            let shape: Vec<usize> = shape.iter().map(|x| *x as usize).collect();
    
            let x : ndarray::ArrayD<f32> = ndarray::ArrayView::from_shape(shape.as_slice(), data).unwrap().to_owned();
            result.push(x.into_pyarray(py));
        }
        result
    }
}

#[pymodule]
fn alkompy(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<Device>().unwrap();

    #[pyfn(m, "query")]
    fn query<'py>(_py: Python<'_>) -> PyResult<Vec<(String, String)>> {
        let res = alkomp::query().iter().map(|x| (x.name(), x.backend().to_string())).collect();
        Ok(res)
    }

    Ok(())
}