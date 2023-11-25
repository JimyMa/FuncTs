import os
import sys
import sysconfig

from setuptools import Extension, find_packages, setup
import setuptools.command.build_ext
from setuptools.dist import Distribution

from subprocess import check_call

import torch

base_dir = os.path.realpath(os.path.dirname(__file__))
build_dir = os.path.join(base_dir, "build")
install_dir = os.path.join(base_dir, "functs")
torch_dir = os.path.dirname(torch.__file__)
torch_lib = os.path.join(torch_dir, "lib")
torch_cmake_prefix_path = os.path.join(torch_dir,
                                       "share",
                                       "cmake",
                                       "Torch")
cmake_path = "/usr/local/bin/cmake"
cmake_python_include_dir = sysconfig.get_path("include")
cmake_python_library = "{}/{}".format(
    sysconfig.get_config_var("LIBDIR"), sysconfig.get_config_var("INSTSONAME")
)
cmake_torchvision_dir = "~/src/meta/vision_install/share/cmake/TorchVision/"

def build_cmake(build_type="Release", generate_command=1):
    os.makedirs(build_dir, exist_ok=True)
    os.makedirs(install_dir, exist_ok=True)

    # Use ninja to build
    gen_args = ["-G Ninja"]

    # CMakeLists PATH
    gen_args += [".."]

    gen_args += ["-DCMAKE_INSTALL_PREFIX={}".format(install_dir)]
    gen_args += ["-DCMAKE_PREFIX_PATH={}".format(torch_cmake_prefix_path)]
    gen_args += ["-DCMAKE_EXPORT_COMPILE_COMMANDS={}".format(generate_command)]
    gen_args += ["-DPYTHON_INCLUDE_DIR={}".format(cmake_python_include_dir)]
    gen_args += ["-DPYTHONLIBRARIES={}".format(cmake_python_library)]
    gen_args += ["-DTorchVision_DIR={}".format(cmake_torchvision_dir)]

    # if fait backend is used, cmake version must be >= 3.27.0!!!
    gen_command = [cmake_path] + gen_args
    check_call(gen_command, cwd=build_dir)

    build_args = ["--build",
                  ".",
                  "--target",
                  "install",
                  "--config",
                  build_type]

    max_jobs = os.getenv("MAX_JOBS")
    if max_jobs:
        build_args += ["-j", max_jobs]

    build_command = [cmake_path] + build_args
    check_call(build_command, cwd=build_dir)


def main():
    build_cmake()
    setup(
        name="functs",
        version="0.0.4",
        ext_modules=[
            Extension(
                "functs._C",
                libraries=["functs_python"],
                library_dirs=[os.path.join(install_dir, "lib")],
                sources=['functs/csrc/stub.c'],
                include_dirs=[],
                language="c",
                extra_link_args=['-Wl,-rpath,$ORIGIN/lib']
            )
        ]
    )


if __name__ == "__main__":
    main()
