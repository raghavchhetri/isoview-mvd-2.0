# IsoView Multi-View Lucy-Richardson Deconvolution (multi-GPU, CUDA)

Multi-view Lucy-Richardson deconvolution for multi-GPU architectures, originally
released as Supplementary Software for the IsoView light-sheet paper
(Chhetri et al., *Nature Methods*, 2015). It compiles on Windows and Linux via CMake.

> **2026 update.** This package has been modernized to build and run on current NVIDIA
> hardware — verified on an **RTX A5000 (Ampere, sm_86) with CUDA 11.8**. The original
> precompiled binaries only targeted pre-Pascal GPUs and no longer run. For the complete
> build procedure, toolchain/toolset requirements, and the record of the port (arch
> flags, MSVC toolset pin, and the in-place → out-of-place cuFFT fix), see
> **`BUILD_2026.md`** in this folder. The notes below are the quick reference.

## 1. Contents

- **`src`** — all C++/CUDA source. Build with CMake (see `BUILD_2026.md`).
- **`test`** — a cropped IsoView test data set (anterior region of a *Drosophila* first
  instar larva expressing GCaMP6s pan-neuronally): four registered views
  (`imReg_1..4.klb`) with matching PSFs (`psfReg_1..4.klb`), plus the example
  configuration `config_2026.xml`.
- **`CMakeLists.txt`** — top-level CMake project file.
- **`BUILD_2026.md`** — modern build/run guide and porting notes.

The compiled executables are produced under **`build/src/Release/`** when you build.

## 2. Installation and requirements

Building from source is required (the original precompiled binaries do not run on modern
GPUs). You need:

- **CUDA Toolkit 11.8** — fully supports Ampere/sm_86 and installs side-by-side with
  newer toolkits. Archive: https://developer.nvidia.com/cuda-toolkit-archive
- **CMake 3.x** and a supported host compiler. On Windows: Visual Studio 2022 with the
  **MSVC v14.38** toolset, pinned at configure time via `-T version=14.38`.
- The **KLB** image-format library is bundled under `src/external/klb`
  (https://bitbucket.org/fernandoamat/keller-lab-block-filetype).

Do **not** build against CUDA 7.0 (a known cuFFT bug can produce NaNs). The exact
configure/build commands and troubleshooting live in `BUILD_2026.md`.

## 3. XML configuration file

Parameters and file names are passed via an XML file; see the example
`test/config_2026.xml`. Each `<view>` entry describes one image, its PSF, and the affine
transform that registers it to the others; a single `<deconvolution>` entry gives the
algorithm parameters.

**`<view>`** (one per image):

- `A` — 16 floating-point numbers: the 3D affine transform (4x4 matrix) that registers
  this image to the others.
- `imgFilename` — full path to the image file (KLB format).
- `psfFilename` — full path to the PSF file (KLB format; typically much smaller than the
  image).

**`<deconvolution>`** (exactly one):

- `imBackground` — background intensity subtracted from every image before deconvolution.
  For the test data use **100**; with 0 the Lucy-Richardson iteration is prone to
  ringing-driven divergence.
- `lambdaTV` — weight for total-variation regularization; a value <= 0 disables TV.
- `numIter` — number of Lucy-Richardson iterations.
- `verbose` — if nonzero, intermediate images (affine-transformed views, weights, PSFs)
  are written for debugging; 0 writes only the final result.
- `blockZsize` — Z-block size for partitioning very large images (only used by
  `main_multiviewDeconvLR_multiGPU_blocksZ.exe`). A power of 2 or 3 is recommended.

## 4. Running the test

Input stacks must be in **KLB** format, which allows efficient random access to regions
of interest.

1. Build the project (see `BUILD_2026.md`).
2. Open a terminal in the build output folder, e.g.
   `D:\2026_multiview-deconvolution\2026_IsoView-MVD-Software\build\src\Release`.
3. Run:

   ```
   .\multiview_deconvolution_LR_multiGPU.exe <full_path>\test\config_2026.xml
   ```

Make sure the paths inside the XML are correct for your machine, and keep the XML in a
folder path **without blank spaces**. The result is written next to the first input
image as `imReg_1.klb_dec_LR_multiGPU__iter<N>_lambdaTV<...>.klb`.
