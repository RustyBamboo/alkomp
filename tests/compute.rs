use ndarray as nd;
use vulkomp;

#[test]
fn compute_on_device() {
    let code = "
    #version 450
    layout(local_size_x = 1) in;
    
    layout(set = 0, binding = 0) buffer PrimeIndices {
        uint[] indices;
    };

    uint collatz_iterations(uint n) {
        uint i = 0;
        while(n != 1) {
            if (mod(n, 2) == 0) {
                n = n / 2;
            }
            else {
                n = (3 * n) + 1;
            }
            i++;
        }
        return i;
    }
    
    void main() {
        uint index = gl_GlobalInvocationID.x;
        indices[index] = collatz_iterations(indices[index]);
    }";

    let arr: Vec<u32> = vec![1, 2, 3, 4];

    let mut device = vulkomp::Device::new(0);
    let data_gpu = device.to_device(arr.as_slice());

    let args = vulkomp::ParamsBuilder::new()
        .param(Some(&data_gpu))
        .build(Some(0));

    let compute = device.compile("main", code, args.0).unwrap();

    device.call(compute, (arr.len() as u32, 1, 1), args.1);

    let collatz = futures::executor::block_on(device.get(&data_gpu)).unwrap();
    let collatz = &collatz[0..collatz.len() - 1];

    assert_eq!(&[0, 1, 7, 2], &collatz[..]);
}
