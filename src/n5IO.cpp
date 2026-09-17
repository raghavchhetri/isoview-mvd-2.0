#include "n5IO.h"
#include "json.hpp"
#include "zlib.h"

#include <fstream>
#include <sstream>
#include <cstring>
#include <iostream>
#include <vector>
#include <algorithm>

using json = nlohmann::json;
using namespace std;

namespace
{
	bool readJsonFile(const string& path, json& out)
	{
		ifstream f(path, ios::binary);
		if (!f.is_open())
			return false;
		try {
			f >> out;
		}
		catch (const std::exception& e) {
			cout << "ERROR: n5IO: failed to parse " << path << ": " << e.what() << endl;
			return false;
		}
		return true;
	}

	size_t n5DtypeSize(const string& dtype)
	{
		if (dtype == "uint8" || dtype == "int8") return 1;
		if (dtype == "uint16" || dtype == "int16") return 2;
		if (dtype == "uint32" || dtype == "int32" || dtype == "float32") return 4;
		if (dtype == "uint64" || dtype == "int64" || dtype == "float64") return 8;
		return 0;
	}

	uint16_t readU16BE(const uint8_t* p) { return (uint16_t)((p[0] << 8) | p[1]); }
	uint32_t readU32BE(const uint8_t* p) { return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) | ((uint32_t)p[2] << 8) | p[3]; }

	//gzip-wrapped deflate stream, exact decompressed size known ahead of time
	bool gzipInflate(const uint8_t* src, size_t srcSize, uint8_t* dst, size_t dstSize)
	{
		z_stream strm;
		memset(&strm, 0, sizeof(strm));
		if (inflateInit2(&strm, 15 + 16) != Z_OK)
			return false;

		strm.next_in = (Bytef*)src;
		strm.avail_in = (uInt)srcSize;
		strm.next_out = (Bytef*)dst;
		strm.avail_out = (uInt)dstSize;

		int ret = inflate(&strm, Z_FINISH);
		size_t produced = dstSize - strm.avail_out;
		inflateEnd(&strm);

		if (ret != Z_STREAM_END || produced != dstSize)
		{
			cout << "ERROR: n5IO: gzip inflate failed (ret=" << ret << ", produced " << produced << "/" << dstSize << ")" << endl;
			return false;
		}
		return true;
	}
}

bool n5_isArray(const string& path)
{
	json j;
	if (!readJsonFile(path + "/attributes.json", j))
		return false;
	return j.contains("dimensions");
}

int n5_readHeader(const string& path, n5ArrayHeader& header)
{
	json j;
	if (!readJsonFile(path + "/attributes.json", j))
		return 1;

	if (!j.contains("dimensions"))
	{
		cout << "ERROR: n5IO: " << path << " is not an N5 array (no \"dimensions\")" << endl;
		return 2;
	}

	auto dims = j["dimensions"];
	if (dims.size() != 3)
	{
		cout << "ERROR: n5IO: only 3D N5 arrays are supported (got " << dims.size() << "D)" << endl;
		return 3;
	}
	for (int ii = 0; ii < 3; ii++)
		header.dims[ii] = dims[ii].get<int64_t>();

	header.dataType = j.value("dataType", "");
	if (n5DtypeSize(header.dataType) == 0)
	{
		cout << "ERROR: n5IO: unsupported dataType '" << header.dataType << "'" << endl;
		return 4;
	}

	header.compressionType = j["compression"].value("type", "raw");
	if (header.compressionType != "raw" && header.compressionType != "gzip")
	{
		cout << "ERROR: n5IO: unsupported compression type '" << header.compressionType << "'" << endl;
		return 5;
	}

	return 0;
}

int n5_readArray(const string& path, void* outBuffer, size_t outBufferBytes, const n5ArrayHeader& header)
{
	json j;
	if (!readJsonFile(path + "/attributes.json", j))
		return 1;

	int64_t nominalBlockSize[3];
	for (int ii = 0; ii < 3; ii++)
		nominalBlockSize[ii] = j["blockSize"][ii].get<int64_t>();

	size_t itemSize = n5DtypeSize(header.dataType);
	if (header.dims[0] * header.dims[1] * header.dims[2] * (int64_t)itemSize != (int64_t)outBufferBytes)
	{
		cout << "ERROR: n5IO: output buffer size does not match array dims/dtype" << endl;
		return 2;
	}

	int64_t numBlocks[3];
	for (int ii = 0; ii < 3; ii++)
		numBlocks[ii] = (header.dims[ii] + nominalBlockSize[ii] - 1) / nominalBlockSize[ii];

	for (int64_t bx = 0; bx < numBlocks[0]; bx++)
	{
		for (int64_t by = 0; by < numBlocks[1]; by++)
		{
			for (int64_t bz = 0; bz < numBlocks[2]; bz++)
			{
				ostringstream ss;
				ss << path << "/" << bx << "/" << by << "/" << bz;
				ifstream f(ss.str(), ios::binary | ios::ate);
				if (!f.is_open())
					continue;//missing block == leave as whatever the caller pre-filled (typically zero)

				size_t fileSize = (size_t)f.tellg();
				f.seekg(0);
				vector<uint8_t> raw(fileSize);
				f.read((char*)raw.data(), fileSize);

				if (fileSize < 4)
				{
					cout << "ERROR: n5IO: block file " << ss.str() << " too small to contain a header" << endl;
					return 3;
				}
				uint16_t mode = readU16BE(raw.data());
				uint16_t ndims = readU16BE(raw.data() + 2);
				if (mode != 0 || ndims != 3)
				{
					cout << "ERROR: n5IO: only mode-0 3D blocks are supported (" << ss.str() << ")" << endl;
					return 4;
				}

				int64_t blockDims[3];
				for (int ii = 0; ii < 3; ii++)
					blockDims[ii] = readU32BE(raw.data() + 4 + 4 * ii);

				const uint8_t* payload = raw.data() + 4 + 4 * 3;
				size_t payloadSize = fileSize - (4 + 4 * 3);
				size_t blockVoxels = (size_t)(blockDims[0] * blockDims[1] * blockDims[2]);
				vector<uint8_t> blockBuf(blockVoxels * itemSize);

				if (header.compressionType == "raw")
				{
					if (payloadSize != blockBuf.size())
					{
						cout << "ERROR: n5IO: raw block payload size mismatch (" << ss.str() << ")" << endl;
						return 5;
					}
					memcpy(blockBuf.data(), payload, blockBuf.size());
				}
				else
				{
					if (!gzipInflate(payload, payloadSize, blockBuf.data(), blockBuf.size()))
						return 6;
				}

				//place block into the output array (dims[0]=x fastest, matching N5's own
				//fastest-first "dimensions" convention -- no axis reordering needed)
				int64_t blockStart[3] = { bx * nominalBlockSize[0], by * nominalBlockSize[1], bz * nominalBlockSize[2] };
				int64_t validX = min(blockDims[0], header.dims[0] - blockStart[0]);
				int64_t validY = min(blockDims[1], header.dims[1] - blockStart[1]);
				int64_t validZ = min(blockDims[2], header.dims[2] - blockStart[2]);
				uint8_t* out = (uint8_t*)outBuffer;
				for (int64_t z = 0; z < validZ; z++)
				{
					for (int64_t y = 0; y < validY; y++)
					{
						const uint8_t* srcRow = blockBuf.data() + itemSize * (blockDims[0] * (y + blockDims[1] * z));
						uint8_t* dstRow = out + itemSize * ((blockStart[0]) + header.dims[0] * ((blockStart[1] + y) + header.dims[1] * (blockStart[2] + z)));
						memcpy(dstRow, srcRow, itemSize * validX);
					}
				}
			}
		}
	}

	return 0;
}
