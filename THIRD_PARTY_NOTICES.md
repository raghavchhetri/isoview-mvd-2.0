# Third-party notices

This project's own license is in [`LICENSE.txt`](LICENSE.txt) (Janelia Open-Source
Software License, 3-clause BSD; copyright HHMI, with 2026 modifications by Raghav
Chhetri).

It bundles the third-party components below. Each keeps its own license text in-tree at
the path shown; those texts govern that component and must be retained on redistribution.

## Bundled libraries

| Component | Version | Location | License | Copyright holder | License text |
|---|---|---|---|---|---|
| KLB (Keller lab block file type) | — | `src/external/klb` | BSD-style, 3-clause (HHMI) | Fernando Amat and Howard Hughes Medical Institute | `src/external/klb/LICENSE.txt` |
| zlib (bundled inside KLB) | 1.2.8 | `src/external/klb/src/external/zlib-1.2.8` | zlib license | Jean-loup Gailly and Mark Adler | `zlib.h` header / `README` in that directory |
| bzip2 (bundled inside KLB) | 1.0.6 | `src/external/klb/src/external/bzip2-1.0.6` | bzip2 license (BSD-style) | Julian Seward | `src/external/klb/src/external/bzip2-1.0.6/LICENSE` |
| imWarpFast (affine transform) | — | `src/external/imWarpFast` | BSD | Dirk-Jan Kroon, 2009 | `src/external/imWarpFast/license.txt` |
| XMLParser | V2.24 | `src/external/xmlParser` | BSD | Frank Vanden Berghen, 2002 | notice in `src/external/xmlParser/xmlParser.h` |
| nlohmann/json | 3.11.3 | `src/external/json/json.hpp` | MIT | Niels Lohmann | SPDX/copyright header in `json.hpp` |
| c-blosc (classic; writes Zarr chunks) | 1.21.6 | `src/external/c-blosc` | BSD-3-Clause | Francesc Alted; Blosc Development Team | `src/external/c-blosc/LICENSE.txt`, `src/external/c-blosc/LICENSES/` |
| c-blosc2 (reads Zarr chunks) | 2.15.1 | `src/external/c-blosc2` | BSD-3-Clause | Francesc Alted; Blosc Development Team | `src/external/c-blosc2/LICENSE.txt`, `src/external/c-blosc2/LICENSES/` |

c-blosc and c-blosc2 in turn carry their own bundled compression sources, each with its
license text under the `LICENSES/` folders above or inside `internal-complibs/`:
Zstandard (BSD, Facebook/Meta), LZ4 (BSD-2-Clause), BitShuffle (MIT), FastLZ (MIT),
and zlib / zlib-ng (zlib license). zlib and zlib-ng sources are present in the tree but
are compiled out of this project's build (`DEACTIVATE_ZLIB`).

## NVIDIA CUDA sample code

The following files derive from NVIDIA CUDA SDK samples and carry NVIDIA copyright
headers (1993–2014) that refer to the NVIDIA end-user license agreement (EULA):

- `src/dct8x8/` — the NVIDIA-headed source files (including `helper_*.h`), plus the
  sample `data/` and `doc/` assets that ship with that SDK sample
- `src/separableConvolution3D_texture/` — the NVIDIA-headed source files (including
  `helper_*.h` and `book.h`)
- `src/book.h`

These files were already part of the original IsoView supplementary software package and
are retained here unchanged. Their terms are NVIDIA's, not this project's; the BSD license
in `LICENSE.txt` does not apply to them. NVIDIA has since published the current CUDA
samples under a BSD license (github.com/NVIDIA/cuda-samples); replacing these files with
those versions is a possible follow-up.
