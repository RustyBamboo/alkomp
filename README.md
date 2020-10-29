# alkomp

alkomp is a GPGPU library written in Rust for preforming compute operations. It's designed to work over [WebGPU](https://www.w3.org/community/gpu/), enabling compute code to work on DirectX, Vulkan, Metal, and eventually OpenCL and the browser.

Currently, compute kernel codes, which run on GPU, are not natively written in Rust. [Shaderc](https://github.com/google/shaderc) is used to compile `GLSL` to `SPIR-V`.

Planned:

- [ ] Build a crate of common operations for GPU
- [ ] Bindings for C and Python
- [ ] Integrate [rust-gpu](https://github.com/EmbarkStudios/rust-gpu) to write native computer shaders

## Get Started

Create your project: `cargo new --bin "gpuproject"`

Add to `cargo.toml`:
```
[dependencies]
alkomp = {git = "https://github.com/RustyBamboo/alkomp", branch = "main"}
```

Modify `src/main.rs`.

As an example, this code runs the [Collatz](https://en.wikipedia.org/wiki/Collatz_conjecture) sequence on the GPU.
```rust
use alkomp;
fn main() {
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

    let mut device = alkomp::Device::new(0);
    let data_gpu = device.to_device(arr.as_slice());

    let args = alkomp::ParamsBuilder::new()
        .param(Some(&data_gpu))
        .build(Some(0));

    let compute = device.compile("main", code, args.0).unwrap();

    device.call(compute, (arr.len() as u32, 1, 1), args.1);

    let collatz = futures::executor::block_on(device.get(&data_gpu)).unwrap();
    let collatz = &collatz[0..collatz.len() - 1];

    assert_eq!(&[0, 1, 7, 2], &collatz[..]);
}
```
