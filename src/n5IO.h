/*
* \brief minimal N5 array reader, used for the raw (un-warped) PSFs in the
* BigStitcher-exported datasets (psf.n5/psf_t0_vwNNN).
*
* Supports N5's default block layout (mode 0) with gzip or raw (uncompressed)
* compression -- what BigStitcher's PSF export actually writes. See TODO.md for
* other N5 compression codecs (bzip2/xz/blosc) if a future dataset needs them.
*/
#ifndef __N5_IO_HEADER_H__
#define __N5_IO_HEADER_H__

#include <cstdint>
#include <string>

struct n5ArrayHeader
{
	int64_t dims[3]; //fastest-varying first (N5's own "dimensions" convention, index 0 = fastest), matches this project's dims[0]=x convention
	std::string dataType;        //"uint8","uint16","uint32","uint64","int8","int16","int32","int64","float32","float64"
	std::string compressionType; //"raw" or "gzip"
};

//true if path/attributes.json exists and describes an N5 array (has a "dimensions" key)
bool n5_isArray(const std::string& path);

//parses path/attributes.json. Returns 0 on success.
int n5_readHeader(const std::string& path, n5ArrayHeader& header);

//reads the whole array into outBuffer (caller-allocated, zero-initialized, native
//dtype matching header.dataType). Returns 0 on success.
int n5_readArray(const std::string& path, void* outBuffer, size_t outBufferBytes, const n5ArrayHeader& header);

#endif
