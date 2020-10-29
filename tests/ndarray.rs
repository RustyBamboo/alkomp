use ndarray as nd;
use vulkomp;

#[test]
fn ndarray_to_device() {
    let mut device = vulkomp::Device::new(0);

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

    let mut device = vulkomp::Device::new(0);

    let arr: nd::Array<f32, _> = nd::Array::ones((2, 3));

    let size_gpu = device.to_device(arr.shape());
    let data_gpu = device.to_device(&arr.as_slice().unwrap());

    let args = vulkomp::ParamsBuilder::new()
        .param(Some(&data_gpu))
        .param(Some(&size_gpu))
        .build(Some(0));

    let compute = device.compile("main", code, args.0).unwrap();

    device.call(compute, (1, 1, 1), args.1);


    let shape = futures::executor::block_on(device.get(&size_gpu)).unwrap();
    let data = futures::executor::block_on(device.get(&data_gpu)).unwrap();

    let shape = &shape[..];
    let data = &data[0..data.len() - 1];

    let x = nd::ArrayView::from_shape(shape, data).unwrap();

    println!("{:?}", x);
    // assert!(x == arr.into_dyn());
}