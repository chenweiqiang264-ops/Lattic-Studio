@echo off
echo ======================================================================
echo 编译 C++ 八叉树 SDF 加速模块
echo ======================================================================
echo.

cd core\cpp
python build_octree_sdf.py build_ext --inplace

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ======================================================================
    echo 编译成功！
    echo ======================================================================
    echo.
    echo 生成的文件：
    dir /b cpp_octree_sdf*.pyd
    echo.
    echo 现在可以在程序中使用 C++ 加速的八叉树构建了
    echo.
) else (
    echo.
    echo ======================================================================
    echo 编译失败！
    echo ======================================================================
    echo.
    echo 请检查：
    echo 1. 是否安装了 Visual Studio 或 Build Tools
    echo 2. 是否安装了 pybind11: pip install pybind11
    echo 3. 编译器是否支持 C++17
    echo.
)

cd ..\..
pause
