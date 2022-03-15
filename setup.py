from setuptools import setup, find_packages

description = \
"""A multi-wavelength SED model based on radiative transfer simulations and
deep learning"""

install_requires = [
    'astropy>=3',
    'numpy>=1.17',
    'astro-sedpy>=0.2',
    'torch>=1.9',
    'tqdm',
]

# Get version
exec(open('starduster/version.py', 'r').read())

setup(
    name='starduster',
    version=__version__,
    author="Yisheng Qiu",
    author_email="hpc_yqiuu@163.com",
    description=description,
    license="GPLv3",
    url='https://github.com/yqiuu/starduster',
    install_requires=install_requires,
    packages=find_packages(),
    include_package_data=True,
)
