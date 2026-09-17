#!/usr/bin/env python3
"""Convert a BigStitcher SpimData2 XML (Zarr v3 views + N5 PSFs) into this
project's native <document> deconvolution XML, one file per tile.

Each SpimData2 ViewRegistration stores a chain of affine transforms that must
be composed in listed order (first-listed applied last, to a column vector):
Net = M_1 * M_2 * ... * M_n. Each <affine> is 12 numbers, row-major 3x4, with
translation in the last column (indices 3, 7, 11) -- padding that to 4x4 with
a bottom row of [0,0,0,1] and flattening row-major gives this project's own
16-float "A" directly (verified against the mex-interface assignment in
src/external/imWarpFast/affine_transform_3d_single.c: A[4*row+col] = M(row,col)).
"""
import argparse
import os
import xml.etree.ElementTree as ET

import numpy as np


def parse_affine_12(text):
    vals = [float(v) for v in text.split()]
    assert len(vals) == 12, f"expected 12 affine values, got {len(vals)}"
    m = np.eye(4, dtype=np.float64)
    m[0, :] = vals[0:4]
    m[1, :] = vals[4:8]
    m[2, :] = vals[8:12]
    return m


def compose_view_registration(reg_elem):
    net = np.eye(4, dtype=np.float64)
    for t in reg_elem.findall("ViewTransform"):
        net = net @ parse_affine_12(t.find("affine").text)
    return net


def format_affine_16(m):
    return " ".join(f"{v:.12f}" for v in m[:4, :4].reshape(-1))


def resolve_zarr_base(seq_desc_elem, xml_dir):
    zarr_elem = seq_desc_elem.find("ImageLoader/zarr")
    zarr_text = zarr_elem.text.strip()
    if zarr_elem.get("type") == "absolute":
        path = zarr_text
        if path.startswith("file:/"):
            path = path[len("file:/"):]
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(xml_dir, zarr_text))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spimdata_xml")
    ap.add_argument("--tile", type=int, default=None, help="tile id to convert (default: first tile found)")
    ap.add_argument("--all-tiles", action="store_true", help="convert every tile instead of just one")
    ap.add_argument("--output-dir", default=None, help="default: same directory as spimdata_xml")
    ap.add_argument("--num-iter", type=int, default=1, help="LR iterations (default 1, for the first validation run)")
    ap.add_argument("--background", type=float, default=100.0)
    ap.add_argument("--lambda-tv", type=float, default=0.0)
    ap.add_argument("--resolution-level", type=int, default=0, help="OME-Zarr multiscale level to read (0 = full res)")
    args = ap.parse_args()

    xml_path = os.path.abspath(args.spimdata_xml)
    xml_dir = os.path.dirname(xml_path)
    out_dir = os.path.abspath(args.output_dir) if args.output_dir else xml_dir
    psf_dir = os.path.join(xml_dir, "psf.n5")

    root = ET.parse(xml_path).getroot()
    seq_desc = root.find("SequenceDescription")
    zarr_base = resolve_zarr_base(seq_desc, xml_dir)

    zgroups = {int(zg.get("setup")): zg.get("path") for zg in seq_desc.findall("ImageLoader/zgroups/zgroup")}

    view_setups = {}
    for vs in seq_desc.findall("ViewSetups/ViewSetup"):
        setup_id = int(vs.findtext("id"))
        attrs = vs.find("attributes")
        voxel_size = [float(v) for v in vs.findtext("voxelSize/size").split()]
        view_setups[setup_id] = {
            "name": vs.findtext("name"),
            "tile": int(attrs.findtext("tile")),
            "angle": int(attrs.findtext("angle")),
            "voxelSize": voxel_size,
        }

    psf_files = {int(b.get("setup")): b.findtext("File") for b in root.findall("PointSpreadFunctions/BoundingBoxDefinition")}

    affines = {int(reg.get("setup")): compose_view_registration(reg) for reg in root.findall("ViewRegistrations/ViewRegistration")}

    tiles = {}
    for setup_id, info in view_setups.items():
        tiles.setdefault(info["tile"], []).append(setup_id)

    if args.all_tiles:
        tile_ids = sorted(tiles.keys())
    elif args.tile is not None:
        tile_ids = [args.tile]
    else:
        tile_ids = [sorted(tiles.keys())[0]]

    os.makedirs(out_dir, exist_ok=True)

    for tile_id in tile_ids:
        setup_ids = sorted(tiles[tile_id])

        # anisotropyZ (z voxel size / xy voxel size) is exact from the XML's own
        # calibration; the C++ side's affine-pattern heuristic
        # (paramDeconvolution::getAnisotropyZfromAffine) only recognizes
        # identity-except-Z matrices, which none of these composed affines are.
        vs0 = view_setups[setup_ids[0]]["voxelSize"]
        anisotropy_z = vs0[2] / vs0[0]

        lines = ['<?xml version="1.0" encoding="utf-8"?>', "<document>"]
        for setup_id in setup_ids:
            img_path = os.path.join(zarr_base, zgroups[setup_id], str(args.resolution_level))
            psf_path = os.path.join(psf_dir, psf_files[setup_id])
            lines.append(
                f'<view imgFilename="{img_path}" psfFilename="{psf_path}" '
                f'A="{format_affine_16(affines[setup_id])}" psfNeedsAffineTransform="1"></view>'
            )
        lines.append(
            f'<deconvolution blockZsize="-1" imBackground="{args.background:.6f}" '
            f'lambdaTV="{args.lambda_tv:.6f}" numIter="{args.num_iter}" verbose="0" '
            f'saveAsUINT16="1" saveAsZarr="1" anisotropyZ="{anisotropy_z:.6f}" prefix="tile{tile_id:02d}"/>'
        )
        lines.append("</document>")

        out_path = os.path.join(out_dir, f"tile{tile_id:02d}_deconv.xml")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"tile {tile_id}: {len(setup_ids)} views -> {out_path}")


if __name__ == "__main__":
    main()
