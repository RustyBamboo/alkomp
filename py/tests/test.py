#!python3

import alkompy
import numpy as np

arr = np.array(range(1,100), dtype=np.float32)

dev = alkompy.Device(0)
dev.to_device(arr)

code = """
    #version 450
    layout(local_size_x = 1) in;
    
    layout(set = 0, binding = 0) buffer PrimeIndices {
        float[] indices;
    };

    uint collatz_iterations(float n) {
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
        indices[index] = collatz_iterations(float(indices[index]));
    }"""

out = dev.run("main", code, (len(arr), 1, 1))

print(out)