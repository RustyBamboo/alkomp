use vulkomp;
use ndarray as nd;
use futures::executor::block_on;

#[test]
fn test_to_device() {
    let mut device = vulkomp::Device::new(0);

    let arr: nd::Array<f32, _> = nd::Array::ones((1,1));

    let arr = arr.as_slice().unwrap();

    println!("{:?}", arr);

    let gpu : vulkomp::GPU<[f32]> = device.to_device_t(&arr);

    println!("{:?}", futures::executor::block_on(device.get(&gpu)));

}