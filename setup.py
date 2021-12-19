from setuptools import setup, find_packages

install_requires = [
    'astropy>=4.2',
    'numpy>=1.17.0',
    'astro-sedpy>=0.2.0',
    'torch>=1.9.0',
    'tqdm',
]

setup(
    name='starduster',
    version='alpha',
    description='A dust extinction curve generator',
    author="Yisheng Qiu",
    author_email="hpc_yqiuu@163.com",
    url='https://github.com/yqiuu/starduster',
    install_requires=install_requires,
    packages=find_packages(),
)
