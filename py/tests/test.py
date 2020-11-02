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

# Compile and run the code, and specifying the order of the layout
dev.run("main", code, (len(arr), 1, 1), [data_gpu])

result = dev.get(data_gpu)
assert((result == np.array([0, 1, 7, 2])).all())