from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy


extensions = [
    Extension(
        "QuickshiftPP",
        sources=["quickshift_pp.pyx"],
        language="c++",
        include_dirs=[numpy.get_include()],
    )
]

setup(
    name="dcsc-pu-quickshiftpp",
    version="1.0.0",
    description="Modified Quickshift++ core extractor used by DCSC-PU",
    ext_modules=cythonize(extensions, compiler_directives={"language_level": "3"}),
    py_modules=[],
)
