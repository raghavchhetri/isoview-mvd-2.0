#CMake file to build multi-view deconvolution in CUDA

#define important variables for the rest of the project

# refer to the root source directory of the project as ${TGMM_SOURCE_DIR} and
# to the root binary directory of the project as ${TGMM_BINARY_DIR}. 
#The FindCUDA script is distributed since version 2.8

cmake_minimum_required (VERSION 2.8)
project (multi-view_deconvolution_src)



#
#Set important flags
#
#add flags for the C++11 
if(CMAKE_COMPILER_IS_GNUCXX) 
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread")
	LINK_DIRECTORIES("/lib64")
elseif ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++ -std=c++11")
endif()


#to locate scripts
set (CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake") 

#
# Find necessary libraries
#
#Cuda
SET(CUDA_NVCC_FLAGS "-gencode arch=compute_86,code=sm_86;-gencode arch=compute_86,code=compute_86;-allow-unsupported-compiler" CACHE STRING "Semi-colon delimit multiple arguments")
SET(CUDA_NVCC_FLAGS_DEBUG -g; -G;-O0 CACHE STRING "Semi-colon delimit multiple arguments") #set before FIND_PACKAGE(CUDA) in order to avoid FORCE to show them in GUI. So user can modify them
SET(CUDA_NVCC_FLAGS_RELEASE -O3 CACHE STRING "Semi-colon delimit multiple arguments")
SET(CUDA_NVCC_FLAGS_RELWITHDEBINFO -O3 CACHE STRING "Semi-colon delimit multiple arguments")

FIND_PACKAGE(CUDA REQUIRED)
SET(CUDA_VERBOSE_BUILD ON FORCE)
INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS}) 
#lcudart and lcuda  are already added as variables with find package
#SET(LIBS_CUDA_CUSPARSE "cusparse" CACHE STRING "")#add CUDA libraries ignored in CUDA.cmake

#
#submodules: KLB I/O
#
include_directories("${PROJECT_SOURCE_DIR}/external/klb/src")
add_subdirectory("${PROJECT_SOURCE_DIR}/external/klb/src" "${CMAKE_CURRENT_BINARY_DIR}/KLB")
LINK_DIRECTORIES("${PROJECT_SOURCE_DIR}/external/klb/src")

#zlib headers (KLB already vendors and builds zlib; n5IO.cpp reuses that same build
#for N5's gzip-compressed PSF blocks instead of vendoring a second copy)
include_directories("${PROJECT_SOURCE_DIR}/external/klb/src/external/zlib-1.2.8")
include_directories("${CMAKE_CURRENT_BINARY_DIR}/KLB/external/zlib-1.2.8")

#
# separable convolution
#
include_directories("${PROJECT_SOURCE_DIR}/separableConvolution3D_texture")
add_subdirectory("${PROJECT_SOURCE_DIR}/separableConvolution3D_texture" "${CMAKE_CURRENT_BINARY_DIR}/separableConv")
LINK_DIRECTORIES("${PROJECT_SOURCE_DIR}/separableConvolution3D_texture")

#
# dct8x8 to calculate weigths
#
include_directories("${PROJECT_SOURCE_DIR}/dct8x8")
add_subdirectory("${PROJECT_SOURCE_DIR}/dct8x8" "${CMAKE_CURRENT_BINARY_DIR}/dct8x8")
LINK_DIRECTORIES("${PROJECT_SOURCE_DIR}/dct8x8")

#
# code to calculate affine transformatiosn of 3D images fast
#
include_directories("${PROJECT_SOURCE_DIR}/external/imWarpFast")
add_subdirectory("${PROJECT_SOURCE_DIR}/external/imWarpFast" "${CMAKE_CURRENT_BINARY_DIR}/imWarpFast")
LINK_DIRECTORIES("${PROJECT_SOURCE_DIR}/external/imWarpFast")

#
# single-header JSON parser (Zarr v3 / N5 metadata)
#
include_directories("${PROJECT_SOURCE_DIR}/external/json")

#
# c-blosc2: codec used by the Zarr v3 chunks in the BigStitcher-exported datasets.
# Only the static library is needed; tests/benchmarks/examples/plugins are disabled
# to keep the build surface (and risk under the pinned MSVC v14.38 toolset) minimal.
#
set(BUILD_STATIC ON CACHE BOOL "" FORCE)
set(BUILD_SHARED OFF CACHE BOOL "" FORCE)
set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_FUZZERS OFF CACHE BOOL "" FORCE)
set(BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_PLUGINS OFF CACHE BOOL "" FORCE)
#we only need the zstd codec (what the Zarr v3 chunks in this dataset use); disabling
#c-blosc2's own bundled zlib avoids a CMake target-name collision with KLB's own
#vendored zlib-1.2.8 (both define a target literally named "zlib")
set(DEACTIVATE_ZLIB ON CACHE BOOL "" FORCE)
include_directories("${PROJECT_SOURCE_DIR}/external/c-blosc2/include")
add_subdirectory("${PROJECT_SOURCE_DIR}/external/c-blosc2" "${CMAKE_CURRENT_BINARY_DIR}/c-blosc2")

#
# classic c-blosc (the pre-blosc2 library): c-blosc2 can only WRITE its own newer
# "blosc2" chunk header format (verified against its source -- every compress entry
# point hardcodes the new format version), which numcodecs (zarr-python) and Fiji's
# n5-zarr/n5-blosc cannot decode. Classic c-blosc always writes the older format that
# both of those expect, so it's used only for the Zarr *write* path; c-blosc2 (which
# can read both formats) still handles all reading.
#
#classic c-blosc and c-blosc2 each bundle their own Windows pthread-emulation shim
#(needed because this toolchain has no real pthreads), and both give those symbols
#external linkage -- linking both as static libraries into the same executable causes
#LNK2005 duplicate-symbol errors (verified: classic c-blosc's blosc.c textually
#`#include`s win32/pthread.c into itself, so it can't be split out as a separate,
#excludable translation unit). Building classic c-blosc as a shared library (DLL)
#isolates its symbols in their own namespace and avoids the collision entirely.
set(BUILD_STATIC OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED ON CACHE BOOL "" FORCE)
set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_FUZZERS OFF CACHE BOOL "" FORCE)
set(BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(DEACTIVATE_ZLIB ON CACHE BOOL "" FORCE)
include_directories("${PROJECT_SOURCE_DIR}/external/c-blosc/blosc")
add_subdirectory("${PROJECT_SOURCE_DIR}/external/c-blosc" "${CMAKE_CURRENT_BINARY_DIR}/c-blosc")

#blosc.dll needs to sit next to the executables (or be on PATH) to load at runtime,
#the same way cudart64_110.dll/cufft64_10.dll already do
add_custom_command(TARGET blosc_shared POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:blosc_shared> "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>")



#
# submodules: add source and header files from subfolders
#
file(GLOB CUDA_HDRS *.h)
file(GLOB CUDA_CUDA_SRCS *.cu)

file(GLOB CPP_SRCS *.cpp)

file(GLOB External_xmlParser_SRCS external/xmlParser/*.cpp)
file(GLOB External_xmlParser_HDRS external/xmlParser/*.h)

CUDA_COMPILE(CUDA_CUDA_OBJ STATIC ${CUDA_CUDA_SRCS} OPTIONS ${CUOPTS})


#generate library
add_library(multiview_deconvolution_GPU ${CPP_SRCS} ${CUDA_SRCS} ${CUDA_CUDA_OBJ} ${CUDA_HDRS} ${CUDA_CUDA_SRCS} ${External_xmlParser_SRCS} ${External_xmlParser_HDRS})

#link libraries
TARGET_LINK_LIBRARIES(multiview_deconvolution_GPU ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY} ${CUDA_cufft_LIBRARY})
TARGET_LINK_LIBRARIES(multiview_deconvolution_GPU SeparableConvolution3D_CUDA_lib DCT8x8_CUDA_lib imwarpFastLib)
TARGET_LINK_LIBRARIES(multiview_deconvolution_GPU klblib)
TARGET_LINK_LIBRARIES(multiview_deconvolution_GPU blosc2_static blosc_shared)
SET_TARGET_PROPERTIES(multiview_deconvolution_GPU PROPERTIES LINKER_LANGUAGE CXX)


#executable for multiview deconvolution
ADD_EXECUTABLE(multiview_deconvolution_LR_multiGPU main_multiviewDeconvLR_multiGPU.cxx )
TARGET_LINK_LIBRARIES(multiview_deconvolution_LR_multiGPU multiview_deconvolution_GPU)

#executable for multiview deconvolution with low memory computers (using I/O)
ADD_EXECUTABLE(multiview_deconvolution_LR_multiGPU_low_mem main_multiviewDeconvLR_multiGPU_low_mem.cxx )
TARGET_LINK_LIBRARIES(multiview_deconvolution_LR_multiGPU_low_mem multiview_deconvolution_GPU)

#executable for multiview deconvolution with very large datasets. Deconvolution is performed in blocks along z to fit in CPU and GPU memory
ADD_EXECUTABLE(main_multiviewDeconvLR_multiGPU_blocksZ main_multiviewDeconvLR_multiGPU_blocksZ.cxx )
TARGET_LINK_LIBRARIES(main_multiviewDeconvLR_multiGPU_blocksZ multiview_deconvolution_GPU)

#standalone diagnostic tool for the Zarr v3 reader (no GPU/deconvolution involved)
ADD_EXECUTABLE(test_zarr_read test_zarr_read.cxx)
TARGET_LINK_LIBRARIES(test_zarr_read multiview_deconvolution_GPU)

#standalone dev tool: KLB -> Zarr v3 conversion, for validating the Zarr read path
#against existing KLB-based ground-truth deconvolution results
ADD_EXECUTABLE(tool_klb2zarr tool_klb2zarr.cxx)
TARGET_LINK_LIBRARIES(tool_klb2zarr multiview_deconvolution_GPU)

#standalone dev tool: compares two images (KLB or Zarr v3) voxel-by-voxel
ADD_EXECUTABLE(tool_compare_images tool_compare_images.cxx)
TARGET_LINK_LIBRARIES(tool_compare_images multiview_deconvolution_GPU)

#blosc.dll needs to sit next to each executable to load at runtime, the same way
#cudart64_110.dll/cufft64_10.dll already do. A POST_BUILD command
#attached to blosc_shared itself doesn't reliably fire under the Visual Studio
#generator when the target lives in a different directory scope (verified: the
#generated blosc_shared.vcxproj never gained the PostBuildEvent), so each executable
#copies it for itself instead.
foreach(exeTarget multiview_deconvolution_LR_multiGPU multiview_deconvolution_LR_multiGPU_low_mem main_multiviewDeconvLR_multiGPU_blocksZ test_zarr_read tool_klb2zarr tool_compare_images)
	add_custom_command(TARGET ${exeTarget} POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:blosc_shared> "$<TARGET_FILE_DIR:${exeTarget}>")
endforeach()
