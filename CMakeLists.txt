cmake_minimum_required ( VERSION 3.5 )
project ( multiview_deconvolution )

# NOTE (2026): keep the CMake minimum at 3.5. The build relies on the legacy FindCUDA
# module, which newer CMake policy removes (CMP0146, CMake >= 3.27); requesting a higher
# minimum would switch that policy to NEW and break the CUDA discovery. If you run CMake
# 4.x, pass -DCMAKE_POLICY_VERSION_MINIMUM=3.5 (the sub-directory CMakeLists still
# declare 2.8). See BUILD_2026.md for the full build procedure.

# contains main source files (executables are produced under build/src/Release)
add_subdirectory("${PROJECT_SOURCE_DIR}/src")
