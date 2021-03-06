<p align="center">
  <img width="25%" src="https://github.com/RustyBamboo/alkomp/raw/main/docs/alkomp-logo.png">
</p>

--------------------------------------------------------------------

![crates.io](https://img.shields.io/crates/v/alkomp) ![pypi.org](https://img.shields.io/pypi/v/alkompy?color=blue)

A GPGPU library written in Rust for performing compute operations on native hardware or on the web. It's designed to work over [WebGPU](https://www.w3.org/community/gpu/), enabling compute code to work on DirectX, Vulkan, Metal, and eventually OpenCL and the browser.

Python bindings work around `numpy` arrays, with an example provided below.

## Installation

### Packaged

Rust (`Cargo.toml`):
```
alkomp = {version = "*", features = "shaderc"}
```

Python:
```
pip3 install alkompy
```

### Latest

*straight from the faucet*

Rust (`Cargo.toml`)
```
alkomp = {git = "https://github.com/RustyBamboo/Alkomp", branch = "main", features = "shaderc"}
```

Python
```
git clone https://github.com/RustyBamboo/Alkomp
cd py
pip3 install -r requirements-dev.txt
python3 setup.py install --user
python3 test/test.py
```

## Examples

In this example, we run the [Collatz](https://en.wikipedia.org/wiki/Collatz_conjecture) sequence on the GPU.

### Rust

```
cargo new --bin collatz
cd collatz
echo 'alkomp = {git = "https://github.com/RustyBamboo/Alkomp", branch = "main", features = "shaderc"}' >> Cargo.toml
echo 'futures = "*"' >> Cargo.toml
```

`src/main.rs`
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

        let mut spirv = alkomp::glslhelper::GLSLCompile::new(&code);
        let shader = spirv.compile("main").unwrap();

        let arr: Vec<u32> = vec![1, 2, 3, 4];

        let mut device = alkomp::Device::new(0);
        let data_gpu = device.to_device(arr.as_slice());

        let args = alkomp::ParamsBuilder::new()
            .param(Some(&data_gpu))
            .build(Some(0));

        let compute = device.compile("main", &shader, &args.0).unwrap();

        device.call(compute, (arr.len() as u32, 1, 1), &args.1);

        let collatz = futures::executor::block_on(device.get(&data_gpu)).unwrap();
        let collatz = &collatz[0..collatz.len() - 1];

        assert_eq!(&[0, 1, 7, 2], &collatz[..]);
}
```

### Python

```
touch collatz.py
```

`collatz.py`
```python
#!python3

import alkompy
import numpy as np

arr = np.array(range(1,5), dtype=np.uint32)

# Retrieve a GPU device
dev = alkompy.Device(0)

# Send data to a GPU
data_gpu = dev.to_device(arr)

# GLSL code to compile
code = """
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
    }"""

shader = alkompy.compile_glsl(code)

# Run the shader and specifying the order of the bindings
dev.run("main", shader, (len(arr), 1, 1), [data_gpu])

result = dev.get(data_gpu)
assert((result == np.array([0, 1, 7, 2])).all())
```

```
python3 collatz.py
```

### TODO

- [ ] Build a crate of common operations for GPU
- [X] Bindings for Python
- [ ] Integrate [rust-gpu](https://github.com/EmbarkStudios/rust-gpu) to write native computer shaders

Currently, compute kernel codes, which run on GPU, are not natively written in Rust. [Shaderc](https://github.com/google/shaderc) is used to compile `GLSL` to `SPIR-V`.
