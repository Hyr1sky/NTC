from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
      name='pf',
      version='0.1.0',
      ext_modules=[
            CppExtension('pf', ['pf.cpp'])
      ],
      cmdclass={'build_ext': BuildExtension}
)
