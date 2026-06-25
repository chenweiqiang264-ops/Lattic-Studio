"""
编译 cpp_periodic_sdf C++ 扩展

使用方法:
    python core/cpp/build_periodic_sdf.py
"""

from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
import sys
import os

# 检测 OpenMP 支持
def has_openmp():
    """检测编译器是否支持 OpenMP"""
    import subprocess
    import tempfile
    
    test_code = """
    #include <omp.h>
    int main() { return 0; }
    """
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as f:
        f.write(test_code)
        test_file = f.name
    
    try:
        if sys.platform == 'win32':
            # Windows MSVC
            result = subprocess.run(
                ['cl', '/openmp', test_file, '/Fe:test.exe'],
                capture_output=True,
                timeout=10
            )
        else:
            # Linux/Mac GCC/Clang
            result = subprocess.run(
                ['g++', '-fopenmp', test_file, '-o', 'test'],
                capture_output=True,
                timeout=10
            )
        
        success = result.returncode == 0
    except:
        success = False
    finally:
        try:
            os.unlink(test_file)
            if sys.platform == 'win32':
                if os.path.exists('test.exe'):
                    os.unlink('test.exe')
            else:
                if os.path.exists('test'):
                    os.unlink('test')
        except:
            pass
    
    return success

# 编译选项
extra_compile_args = []
extra_link_args = []

if sys.platform == 'win32':
    # Windows MSVC
    extra_compile_args = ['/O2', '/std:c++17']
    if has_openmp():
        print("✓ OpenMP 支持已启用（Windows）")
        extra_compile_args.append('/openmp')
    else:
        print("⚠ OpenMP 不可用，将使用单线程版本")
else:
    # Linux/Mac GCC/Clang
    extra_compile_args = ['-O3', '-std=c++17']
    if has_openmp():
        print("✓ OpenMP 支持已启用（Linux/Mac）")
        extra_compile_args.append('-fopenmp')
        extra_link_args.append('-fopenmp')
    else:
        print("⚠ OpenMP 不可用，将使用单线程版本")

# 定义扩展模块
ext_modules = [
    Pybind11Extension(
        "cpp_periodic_sdf",
        ["cpp_periodic_sdf.cpp"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        cxx_std=17,
    ),
]

# 编译
setup(
    name="cpp_periodic_sdf",
    version="1.0.0",
    author="Your Name",
    description="C++ 周期性 SDF 采样加速模块",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)

print("\n" + "=" * 70)
print("编译完成！")
print("=" * 70)
print("\n使用方法:")
print("  from core.cpp.cpp_periodic_sdf import sample_periodic_sdf")
print("  lattice_sdf = sample_periodic_sdf(...)")
print()
