from setuptools import find_packages, setup
from setuptools_rust import RustExtension

with open("README.md", "r") as fh:
    long_description = fh.read()

setup_requires = ['setuptools-rust>=0.10.2']
install_requires = ['numpy']
test_requires = install_requires + ['pytest']

setup(
    name='alkompy',
    version='0.2.0.post3',
    description='A compute library written in Rust with WebGPU',
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="bsd-3-clause",
    rust_extensions=[RustExtension(
        'alkompy',
        './Cargo.toml',
    )],
    install_requires=install_requires,
    setup_requires=setup_requires,
    test_requires=test_requires,
    packages=find_packages(),
    url="https://github.com/RustyBamboo/alkomp",
    zip_safe=False,
    python_requires = '>=3.4',
    include_package_data = True,
)
