from setuptools import setup, find_packages

setup(
    name='wayve',
    version='2.0.0',
    packages=find_packages(exclude=['examples']),
    url='https://gitlab.kuleuven.be/TFSO-software/wayve',
    license='GNU AGPLv3',
    author='TFSO group',
    author_email='koen.devesse@kuleuven.be',
    description='A framework for atmospheric perturbation models.',
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "llvmlite",
        "numba",
        "mpmath",
        "netCDF4"
    ],
    zip_safe=False
)
