from setuptools import setup, find_packages

description = \
    "A multi-wavelength SED model based on radiative transfer simulations and deep learning"
install_requires = [
    'astropy>=4.2',
    'numpy>=1.17.0',
    'astro-sedpy>=0.2.0',
    'torch>=1.9.0',
    'tqdm',
]

setup(
    name='starduster',
    version='beta',
    author="Yisheng Qiu",
    author_email="hpc_yqiuu@163.com",
    description=description,
    license="GPLv3",
    url='https://github.com/yqiuu/starduster',
    install_requires=install_requires,
    packages=find_packages(),
)
