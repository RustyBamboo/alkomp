<p align="center">
  <img width="25%" src="https://github.com/RustyBamboo/alkomp/raw/main/docs/alkomp-logo.png">
</p>

--------------------------------------------------------------------

Python bindings for [alkomp](https://github.com/RustyBamboo/alkomp), a GPGPU library written in Rust for performing compute operations.

```
pip3 install alkompy
```

At this time, the Python interface is designed to specifically work with `numpy ndarrays`. This means you can quickly send a numpy array to a GPU with `data_gpu = device.to_device(my_np_array)` and run a computation using `device.call(...)`. `to_device` returns an object that records the memory location of a GPU buffer, as well shape and type. In order to retrieve the contents of the buffer: `device.get(data_gpu)`. `get` function returns a numpy in the same shape as `my_np_array`.

### Build from source

`git clone https://github.com/RustyBamboo/alkomp && cd alkomp`

`pip3 install -r requirements-dev.txt`

`python3 setup.py develop --user`

`python3 test/test.py`