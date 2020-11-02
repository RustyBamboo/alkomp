use alkomp;
use ndarray as nd;

#[test]
fn ndarray_to_device() {
    let mut device = alkomp::Device::new(0);

    let arr: nd::Array<f32, _> = nd::Array::ones((5, 2, 2));

    let size_gpu = device.to_device(arr.shape());
    let data_gpu = device.to_device(&arr.as_slice().unwrap());

    let shape = futures::executor::block_on(device.get(&size_gpu)).unwrap();
    let data = futures::executor::block_on(device.get(&data_gpu)).unwrap();

    let shape = &shape[..];
    let data = &data[0..data.len() - 1];

    let x = nd::ArrayView::from_shape(shape, data).unwrap();

    assert!(x == arr.into_dyn());
}

#[cfg(feature = "shaderc")]
#[test]
fn ndarray_compute_device() {
    let code = "
    #version 450
    layout(local_size_x = 1) in;
    
    layout(set = 0, binding = 0) buffer ArrayData {
        float[] data;
    };

    layout(set = 0, binding = 1) buffer DimData {
        uint[] dim;
    };
    
    void main() {
        uint index = gl_GlobalInvocationID.x;
        data[0] = dim[0];
        data[1] = dim[1];
    }";
    let mut spirv = alkomp::glslhelper::GLSLCompile::new(&code);
    let shader = spirv.compile("main").unwrap();

    let mut device = alkomp::Device::new(0);

    let arr: nd::Array<f32, _> = nd::Array::ones((2, 3));

    let s: Vec<u32> = arr.shape().iter().map(|x| *x as u32).collect();

    let size_gpu = device.to_device(s.as_slice());
    let data_gpu = device.to_device(&arr.as_slice().unwrap());

    let args = alkomp::ParamsBuilder::new()
        .param(Some(&data_gpu))
        .param(Some(&size_gpu))
        .build(Some(0));

    let compute = device.compile("main", &shader, &args.0).unwrap();

    device.call(compute, (1, 1, 1), &args.1);

    let shape = futures::executor::block_on(device.get(&size_gpu)).unwrap();
    let data = futures::executor::block_on(device.get(&data_gpu)).unwrap();

    let _shape = &shape[0..shape.len() - 1];
    let data = &data[0..data.len() - 1];

    let x = nd::ArrayView::from_shape(arr.shape(), data).unwrap();

    let expected = nd::array![[2.0, 3.0, 1.0], [1.0, 1.0, 1.0]];
    assert!(x == expected.into_dyn());
}
