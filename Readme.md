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
>
> **2026-09-16:** the pipeline can now also read **Zarr v3** views and **N5** PSFs
> directly (e.g. a BigStitcher/SpimData2 export) and write **Zarr v3** output — see
> §3 below and `CLAUDE.md` §10 for details.

## 1. Contents

- **`src`** — all C++/CUDA source. Build with CMake (see `BUILD_2026.md`).
- **`tools`** — `convert_spimdata_xml.py`: converts a BigStitcher/SpimData2 XML export
  (Zarr v3 views + N5 PSFs) into this project's native `<document>` XML, one file per
  tile. See §3 below.
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
- `imgFilename` — full path to the image file. **KLB**, or (2026-09-16) a **Zarr v3**
  array path or an **N5** array path.
- `psfFilename` — full path to the PSF file (same format options as `imgFilename`;
  typically much smaller than the image).
- `psfNeedsAffineTransform` — 0 (default) if the PSF is already pre-registered to this
  view (the original workflow). 1 if it's a raw/un-warped PSF (e.g. straight out of a
  BigStitcher `psf.n5`) that still needs the (linear part of the) view's own `A` applied.

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
- `saveAsZarr` — 0 (default) writes the result as KLB; 1 writes **Zarr v3** (requires
  `saveAsUINT16="1"`). See §5.
- `anisotropyZ` — optional; overrides the affine-derived Z-anisotropy estimate. Needed
  when `A` is a real composed registration matrix rather than the identity-except-Z
  pattern the default heuristic recognizes (see `CLAUDE.md` §5/§10).

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

## 5. Running on a BigStitcher/SpimData2 export (Zarr v3 + N5)

For a dataset exported from BigStitcher (a SpimData2 XML with `bdv.multimg.zarr` views
and raw N5 PSFs), first convert it to this project's native XML:

```
python tools\convert_spimdata_xml.py <path-to-spimdata.xml> --tile 0
```

This composes each view's registration transform chain into a single `A`, resolves the
Zarr view and N5 PSF paths, computes `anisotropyZ` from the XML's own voxel size, and
writes one native XML per tile (`--all-tiles` to convert every tile at once) with
`psfNeedsAffineTransform="1"` set — the PSFs from a fresh BigStitcher export are raw and
not yet warped to each view. Then run the resulting XML exactly as in §4. Add
`--num-iter`, `--background`, `--lambda-tv` to override the defaults it fills in (the
default is `numIter=1`, meant for a first correctness check before a full run). See
`CLAUDE.md` §10 for the on-disk format details and current limitations (only the
whole-volume executable has the PSF-warp step; no ROI/partial reads yet).

## 6. License

Released under the Janelia Open-Source Software License (3-clause BSD) — see
`LICENSE.txt`; bundled third-party components and their licenses are listed in
`THIRD_PARTY_NOTICES.md`.
