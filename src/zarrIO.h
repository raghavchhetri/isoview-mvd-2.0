/*
* \brief minimal Zarr v3 array reader/writer.
*
* Reads exactly what the BigStitcher-exported OME-Zarr datasets in this project use:
* a regular chunk grid, default ("c/i/j/k") chunk key encoding, and either a plain
* [bytes, blosc] codec chain or that chain wrapped in "sharding_indexed". Writes a
* single-chunk [bytes, blosc] array plus a minimal OME-Zarr group wrapper.
*
* Not a general-purpose Zarr v3 implementation: no partial/ROI reads, no other codecs
* (gzip/zstd-direct/transpose/...), no multi-chunk writes. See TODO.md for follow-ups.
*/
#ifndef __ZARR_IO_HEADER_H__
#define __ZARR_IO_HEADER_H__

#include <cstdint>
#include <string>
#include <vector>

struct zarrArrayInfo
{
	std::vector<int64_t> shape;      //full array shape, axis order as stored in zarr.json
	std::vector<int64_t> chunkShape; //outer chunk_grid chunk_shape (same axis order/length as shape)
	std::string dtype;               //zarr v3 data_type string: "uint8","uint16","uint32","float32","float64",...
};

//true if path/zarr.json exists and describes a Zarr v3 array (node_type == "array")
bool zarr_isArray(const std::string& path);

//parses path/zarr.json into info. Returns 0 on success.
int zarr_readHeader(const std::string& path, zarrArrayInfo& info);

//reads the whole array into outBuffer (caller-allocated, zero-initialized, native dtype
//matching info.dtype). The array's last 3 shape axes are read as (z,y,x); any leading
//axes (e.g. t,c) must have size 1. Zarr's C-order storage (last axis fastest) already
//matches this project's own dims[0]=x-fastest convention, so no transpose is needed --
//just axis-order bookkeeping. Returns 0 on success.
int zarr_readArray(const std::string& path, void* outBuffer, size_t outBufferBytes, const zarrArrayInfo& info);

//writes a simple single-chunk 3D array (dims[0]=x fastest-varying, dims[1]=y, dims[2]=z,
//matching multiviewImage's own dimsImg convention) using the plain "bytes" codec
//(uncompressed) -- a lightweight utility for one-off array dumps/conversions, not the
//full pyramid writer below. dtype is a zarr v3 data_type string ("uint16" or "float32").
int zarr_writeArray3D(const std::string& path, const void* data, const int64_t dims[3], const std::string& dtype);

//writes a uint16 3D volume as an OME-Zarr (NGFF 0.5) multiscale pyramid, matching the
//on-disk conventions of the BigStitcher-exported datasets this project reads: a group
//zarr.json with "ome"/"multiscales" metadata, one array per resolution level (2x2x2
//box-downsampled from the previous level, continuing while the current level's
//smallest dimension is >= 100), each array chunked (full XY, up to 104 Z-planes per
//chunk) and sharded (one shard per chunk, sub-chunked into single Z-planes), and each
//sub-chunk compressed with zstd+bitshuffle via classic ("blosc1"-format) blosc -- the
//format numcodecs (zarr-python) and Fiji's n5-zarr/n5-blosc both expect; c-blosc2 can
//only write its own newer header format, which neither of those can decode.
//`groupPath` is the output ".zarr" directory; dims[] is dims[0]=x,dims[1]=y,dims[2]=z.
int zarr_writePyramid3D(const std::string& groupPath, const uint16_t* data, const int64_t dims[3]);

#endif
