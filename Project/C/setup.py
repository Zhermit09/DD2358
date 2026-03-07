from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "*",
        ["C/*.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-O3"],   # compiler optimization
    ),
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": 3,
            "boundscheck": False,
            "initializedcheck": False,
            "nonecheck": False
        },
        annotate=True,   # generates HTML performance report
    )
)