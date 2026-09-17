#include "zarrIO.h"
#include "json.hpp"
#include "blosc2.h"
#include "blosc.h"

#include <fstream>
#include <sstream>
#include <cstring>
#include <cmath>
#include <iostream>
#include <vector>
#include <algorithm>

using json = nlohmann::json;
using namespace std;

namespace
{
	bool fileExists(const string& path)
	{
		ifstream f(path, ios::binary);
		return f.good();
	}

	bool readJsonFile(const string& path, json& out)
	{
		ifstream f(path, ios::binary);
		if (!f.is_open())
			return false;
		try {
			f >> out;
		}
		catch (const std::exception& e) {
			cout << "ERROR: zarrIO: failed to parse " << path << ": " << e.what() << endl;
			return false;
		}
		return true;
	}

	vector<uint8_t> readWholeFile(const string& path, bool& ok)
	{
		ifstream f(path, ios::binary | ios::ate);
		ok = f.is_open();
		if (!ok)
			return vector<uint8_t>();
		size_t n = (size_t)f.tellg();
		f.seekg(0);
		vector<uint8_t> buf(n);
		f.read((char*)buf.data(), n);
		return buf;
	}

	size_t zarrDtypeSize(const string& dtype)
	{
		if (dtype == "uint8" || dtype == "int8") return 1;
		if (dtype == "uint16" || dtype == "int16") return 2;
		if (dtype == "uint32" || dtype == "int32" || dtype == "float32") return 4;
		if (dtype == "uint64" || dtype == "int64" || dtype == "float64") return 8;
		return 0;
	}

	//path/c/i0/i1/.../iN , default chunk key encoding
	string chunkKeyPath(const string& arrayPath, const vector<int64_t>& idx, const string& separator)
	{
		ostringstream ss;
		ss << arrayPath << "/c";
		for (size_t ii = 0; ii < idx.size(); ii++)
			ss << separator << idx[ii];
		return ss.str();
	}

	uint64_t readU64LE(const uint8_t* p)
	{
		uint64_t v = 0;
		for (int ii = 7; ii >= 0; ii--)
			v = (v << 8) | p[ii];
		return v;
	}

	void writeU64LE(uint8_t* p, uint64_t v)
	{
		for (int ii = 0; ii < 8; ii++)
		{
			p[ii] = (uint8_t)(v & 0xFF);
			v >>= 8;
		}
	}

	//decodes one blosc-compressed buffer of known decompressed size
	bool bloscDecompress(const uint8_t* src, size_t srcSize, uint8_t* dst, size_t dstSize)
	{
		blosc2_dparams dparams = BLOSC2_DPARAMS_DEFAULTS;
		blosc2_context* ctx = blosc2_create_dctx(dparams);
		int n = blosc2_decompress_ctx(ctx, src, (int32_t)srcSize, dst, (int32_t)dstSize);
		blosc2_free_ctx(ctx);
		if (n < 0 || (size_t)n != dstSize)
		{
			cout << "ERROR: zarrIO: blosc2_decompress_ctx failed (returned " << n << ", expected " << dstSize << ")" << endl;
			return false;
		}
		return true;
	}

	//copies a fully-decoded chunk buffer (size = chunkShape[0..2] voxels, axis order z,y,x,
	//last axis fastest) into the whole-array output buffer, clipping to the array's actual
	//bounds (the last chunk along any axis is typically only partially inside the array).
	void placeChunk(uint8_t* outBuffer, const int64_t arrayShape[3], const int64_t chunkShape[3],
		const int64_t chunkStart[3], const uint8_t* chunkBuffer, size_t itemSize)
	{
		int64_t validZ = min(chunkShape[0], arrayShape[0] - chunkStart[0]);
		int64_t validY = min(chunkShape[1], arrayShape[1] - chunkStart[1]);
		int64_t validX = min(chunkShape[2], arrayShape[2] - chunkStart[2]);
		if (validZ <= 0 || validY <= 0 || validX <= 0)
			return;

		for (int64_t z = 0; z < validZ; z++)
		{
			for (int64_t y = 0; y < validY; y++)
			{
				const uint8_t* srcRow = chunkBuffer + itemSize * (chunkShape[2] * (y + chunkShape[1] * z));
				uint8_t* dstRow = outBuffer + itemSize * ((chunkStart[2]) + arrayShape[2] * ((chunkStart[1] + y) + arrayShape[1] * (chunkStart[0] + z)));
				memcpy(dstRow, srcRow, itemSize * validX);
			}
		}
	}

	struct CodecChain
	{
		bool sharded;
		bool hasBlosc;//false means the plain "bytes" codec alone, i.e. raw/uncompressed
		vector<int64_t> shardChunkShape;//sub-chunk shape, only valid if sharded
	};

	bool parseCodecs(const json& codecsJson, CodecChain& chain)
	{
		chain.sharded = false;
		if (codecsJson.size() == 1 && codecsJson[0].value("name", "") == "sharding_indexed")
		{
			chain.sharded = true;
			const json& cfg = codecsJson[0]["configuration"];
			for (auto& v : cfg["chunk_shape"])
				chain.shardChunkShape.push_back(v.get<int64_t>());
			//inner codecs must decode to raw bytes we can consume directly: bytes + blosc
			bool sawBlosc = false;
			for (auto& c : cfg["codecs"])
			{
				string name = c.value("name", "");
				if (name == "blosc") sawBlosc = true;
				else if (name != "bytes")
				{
					cout << "ERROR: zarrIO: unsupported inner shard codec '" << name << "'" << endl;
					return false;
				}
			}
			if (!sawBlosc)
			{
				cout << "ERROR: zarrIO: sharded array without a blosc inner codec is not supported" << endl;
				return false;
			}
			string indexLocation = cfg.value("index_location", "end");
			if (indexLocation != "end")
			{
				cout << "ERROR: zarrIO: only index_location=\"end\" is supported for sharding_indexed" << endl;
				return false;
			}
			return true;
		}

		chain.hasBlosc = false;
		for (auto& c : codecsJson)
		{
			string name = c.value("name", "");
			if (name == "blosc") chain.hasBlosc = true;
			else if (name != "bytes")
			{
				cout << "ERROR: zarrIO: unsupported codec '" << name << "'" << endl;
				return false;
			}
		}
		return true;
	}

	//CRC32C (Castagnoli), used by the sharding_indexed codec's index_codecs checksum
	uint32_t crc32c(const uint8_t* data, size_t len)
	{
		uint32_t crc = 0xFFFFFFFFu;
		for (size_t i = 0; i < len; i++)
		{
			crc ^= data[i];
			for (int j = 0; j < 8; j++)
				crc = (crc >> 1) ^ ((crc & 1u) ? 0x82F63B78u : 0u);
		}
		return crc ^ 0xFFFFFFFFu;
	}

	//2x downsample (box average) of a uint16 volume, this project's own dims[0]=x
	//fastest-varying convention. zFactor=1 downsamples only x,y (used nowhere currently,
	//kept for symmetry); zFactor=2 downsamples all three axes.
	void downsample2x(const uint16_t* src, const int64_t srcDims[3], uint16_t* dst, const int64_t dstDims[3], int zFactor)
	{
		for (int64_t z = 0; z < dstDims[2]; z++)
		{
			for (int64_t y = 0; y < dstDims[1]; y++)
			{
				for (int64_t x = 0; x < dstDims[0]; x++)
				{
					uint32_t sum = 0;
					int count = 0;
					for (int64_t dz = 0; dz < zFactor; dz++)
					{
						int64_t sz = z * zFactor + dz;
						for (int64_t dy = 0; dy < 2; dy++)
						{
							int64_t sy = y * 2 + dy;
							const uint16_t* row = src + srcDims[0] * (sy + srcDims[1] * sz);
							sum += row[x * 2] + row[x * 2 + 1];
							count += 2;
						}
					}
					dst[x + dstDims[0] * (y + dstDims[1] * z)] = (uint16_t)(sum / count);
				}
			}
		}
	}

	//writes one resolution level: a chunked (full XY, <=104-plane Z chunks), sharded
	//(one shard file per chunk, sub-chunked into single Z-planes) Zarr v3 array,
	//compressed with classic ("blosc1"-format) blosc so numcodecs/n5-zarr can read it.
	int writeShardedLevel(const string& levelPath, const uint16_t* data, const int64_t dimsXYZ[3])
	{
		int64_t nx = dimsXYZ[0], ny = dimsXYZ[1], nz = dimsXYZ[2];
		const int64_t nominalChunkZ = min((int64_t)104, nz);
		int64_t numChunks = (nz + nominalChunkZ - 1) / nominalChunkZ;
		size_t itemSize = 2;
		size_t planeVoxels = (size_t)nx * ny;

		system(("mkdir \"" + levelPath + "\" 2>nul").c_str());
		system(("mkdir \"" + levelPath + "/c\" 2>nul").c_str());

		blosc_set_compressor("zstd");
		vector<uint8_t> compBuf(planeVoxels * itemSize + BLOSC_MAX_OVERHEAD);

		for (int64_t cz = 0; cz < numChunks; cz++)
		{
			int64_t zStart = cz * nominalChunkZ;
			int64_t validZ = min(nominalChunkZ, nz - zStart);

			vector<uint8_t> shardBytes;
			vector<uint64_t> idxOffset(nominalChunkZ, UINT64_MAX);
			vector<uint64_t> idxNbytes(nominalChunkZ, UINT64_MAX);

			for (int64_t lz = 0; lz < validZ; lz++)
			{
				const uint16_t* planeSrc = data + planeVoxels * (zStart + lz);//a full XY plane is contiguous in this project's x-fastest layout
				int csize = blosc_compress(3, BLOSC_BITSHUFFLE, itemSize, planeVoxels * itemSize, planeSrc, compBuf.data(), compBuf.size());
				if (csize <= 0)
				{
					cout << "ERROR: zarrIO: blosc_compress failed (" << csize << ")" << endl;
					return 1;
				}
				idxOffset[lz] = shardBytes.size();
				idxNbytes[lz] = (uint64_t)csize;
				shardBytes.insert(shardBytes.end(), compBuf.begin(), compBuf.begin() + csize);
			}

			size_t indexStart = shardBytes.size();
			uint8_t buf8[8];
			for (int64_t lz = 0; lz < nominalChunkZ; lz++)
			{
				writeU64LE(buf8, idxOffset[lz]); shardBytes.insert(shardBytes.end(), buf8, buf8 + 8);
				writeU64LE(buf8, idxNbytes[lz]); shardBytes.insert(shardBytes.end(), buf8, buf8 + 8);
			}
			uint32_t crc = crc32c(shardBytes.data() + indexStart, shardBytes.size() - indexStart);
			uint8_t crcBuf[4] = { (uint8_t)(crc & 0xFF), (uint8_t)((crc >> 8) & 0xFF), (uint8_t)((crc >> 16) & 0xFF), (uint8_t)((crc >> 24) & 0xFF) };
			shardBytes.insert(shardBytes.end(), crcBuf, crcBuf + 4);

			system(("mkdir \"" + levelPath + "/c/" + to_string(cz) + "/0\" 2>nul").c_str());
			string chunkPath = levelPath + "/c/" + to_string(cz) + "/0/0";
			ofstream chunkFile(chunkPath, ios::binary);
			if (!chunkFile.is_open())
			{
				cout << "ERROR: zarrIO: could not create shard file " << chunkPath << endl;
				return 2;
			}
			chunkFile.write((const char*)shardBytes.data(), shardBytes.size());
			chunkFile.close();
		}

		json arrayMeta;
		arrayMeta["zarr_format"] = 3;
		arrayMeta["node_type"] = "array";
		arrayMeta["shape"] = { nz, ny, nx };
		arrayMeta["data_type"] = "uint16";
		arrayMeta["chunk_grid"] = { {"name", "regular"}, {"configuration", {{"chunk_shape", {nominalChunkZ, ny, nx}}}} };
		arrayMeta["chunk_key_encoding"] = { {"name", "default"}, {"configuration", {{"separator", "/"}}} };
		arrayMeta["fill_value"] = 0;
		arrayMeta["codecs"] = json::array({
			{ {"name", "sharding_indexed"}, {"configuration", {
				{"chunk_shape", {1, ny, nx}},
				{"codecs", json::array({
					{ {"name", "bytes"}, {"configuration", {{"endian", "little"}}} },
					{ {"name", "blosc"}, {"configuration", {{"cname", "zstd"}, {"clevel", 3}, {"shuffle", "bitshuffle"}, {"typesize", (int)itemSize}, {"blocksize", 0}}} }
					})},
				{"index_codecs", json::array({
					{ {"name", "bytes"}, {"configuration", {{"endian", "little"}}} },
					{ {"name", "crc32c"} }
					})},
				{"index_location", "end"}
				}} }
			});
		arrayMeta["attributes"] = json::object();
		arrayMeta["storage_transformers"] = json::array();

		ofstream arrayJson(levelPath + "/zarr.json", ios::binary);
		arrayJson << arrayMeta.dump(2);
		arrayJson.close();

		return 0;
	}
}

bool zarr_isArray(const string& path)
{
	json j;
	if (!readJsonFile(path + "/zarr.json", j))
		return false;
	return j.value("node_type", "") == "array" && j.value("zarr_format", 0) == 3;
}

int zarr_readHeader(const string& path, zarrArrayInfo& info)
{
	json j;
	if (!readJsonFile(path + "/zarr.json", j))
		return 1;

	if (j.value("node_type", "") != "array" || j.value("zarr_format", 0) != 3)
	{
		cout << "ERROR: zarrIO: " << path << " is not a Zarr v3 array" << endl;
		return 2;
	}

	info.shape.clear();
	for (auto& v : j["shape"])
		info.shape.push_back(v.get<int64_t>());

	info.dtype = j.value("data_type", "");
	if (zarrDtypeSize(info.dtype) == 0)
	{
		cout << "ERROR: zarrIO: unsupported data_type '" << info.dtype << "'" << endl;
		return 3;
	}

	if (j["chunk_grid"].value("name", "") != "regular")
	{
		cout << "ERROR: zarrIO: only chunk_grid \"regular\" is supported" << endl;
		return 4;
	}
	info.chunkShape.clear();
	for (auto& v : j["chunk_grid"]["configuration"]["chunk_shape"])
		info.chunkShape.push_back(v.get<int64_t>());

	if (j["chunk_key_encoding"].value("name", "") != "default")
	{
		cout << "ERROR: zarrIO: only chunk_key_encoding \"default\" is supported" << endl;
		return 5;
	}

	if (info.shape.size() < 3 || info.shape.size() != info.chunkShape.size())
	{
		cout << "ERROR: zarrIO: expected shape/chunk_shape with matching rank >= 3" << endl;
		return 6;
	}
	for (size_t ii = 0; ii + 3 < info.shape.size(); ii++)
	{
		if (info.shape[ii] != 1)
		{
			cout << "ERROR: zarrIO: leading axis " << ii << " has size " << info.shape[ii] << " (only size-1 leading axes, e.g. t/c, are supported)" << endl;
			return 7;
		}
	}

	return 0;
}

int zarr_readArray(const string& path, void* outBuffer, size_t outBufferBytes, const zarrArrayInfo& info)
{
	json j;
	if (!readJsonFile(path + "/zarr.json", j))
		return 1;

	CodecChain chain;
	if (!parseCodecs(j["codecs"], chain))
		return 2;

	string separator = j["chunk_key_encoding"].value("configuration", json::object()).value("separator", "/");

	size_t itemSize = zarrDtypeSize(info.dtype);
	size_t ndims = info.shape.size();
	int64_t arrayShape[3] = { info.shape[ndims - 3], info.shape[ndims - 2], info.shape[ndims - 1] };
	int64_t chunkShape[3] = { info.chunkShape[ndims - 3], info.chunkShape[ndims - 2], info.chunkShape[ndims - 1] };

	if ((size_t)(arrayShape[0] * arrayShape[1] * arrayShape[2]) * itemSize != outBufferBytes)
	{
		cout << "ERROR: zarrIO: output buffer size does not match array shape/dtype" << endl;
		return 3;
	}

	//leading (t,c,...) chunk index is always 0 since those axes have size 1 (single chunk)
	vector<int64_t> leadingIdx(ndims - 3, 0);

	int64_t numChunks[3];
	for (int d = 0; d < 3; d++)
		numChunks[d] = (arrayShape[d] + chunkShape[d] - 1) / chunkShape[d];

	vector<uint8_t> outerChunkBuf((size_t)(chunkShape[0] * chunkShape[1] * chunkShape[2]) * itemSize);

	for (int64_t cz = 0; cz < numChunks[0]; cz++)
	{
		for (int64_t cy = 0; cy < numChunks[1]; cy++)
		{
			for (int64_t cx = 0; cx < numChunks[2]; cx++)
			{
				vector<int64_t> idx = leadingIdx;
				idx.push_back(cz); idx.push_back(cy); idx.push_back(cx);
				string filePath = chunkKeyPath(path, idx, separator);
				bool chunkExists = fileExists(filePath);
				if (!chunkExists)
					continue;//missing chunk == fill_value (output buffer is expected to be pre-zeroed)

				bool ok;
				vector<uint8_t> fileBytes = readWholeFile(filePath, ok);
				if (!ok)
				{
					cout << "ERROR: zarrIO: could not read chunk file " << filePath << endl;
					return 4;
				}

				fill(outerChunkBuf.begin(), outerChunkBuf.end(), 0);

				if (!chain.sharded)
				{
					if (chain.hasBlosc)
					{
						if (!bloscDecompress(fileBytes.data(), fileBytes.size(), outerChunkBuf.data(), outerChunkBuf.size()))
							return 5;
					}
					else
					{
						if (fileBytes.size() != outerChunkBuf.size())
						{
							cout << "ERROR: zarrIO: raw chunk file " << filePath << " size mismatch" << endl;
							return 5;
						}
						memcpy(outerChunkBuf.data(), fileBytes.data(), fileBytes.size());
					}
				}
				else
				{
					int64_t subShape[3] = { chain.shardChunkShape[ndims - 3], chain.shardChunkShape[ndims - 2], chain.shardChunkShape[ndims - 1] };
					int64_t subGrid[3];
					for (int d = 0; d < 3; d++)
					{
						if (chunkShape[d] % subShape[d] != 0)
						{
							cout << "ERROR: zarrIO: shard chunk_shape does not evenly divide the outer chunk_shape" << endl;
							return 6;
						}
						subGrid[d] = chunkShape[d] / subShape[d];
					}
					int64_t numSub = subGrid[0] * subGrid[1] * subGrid[2];
					size_t indexBytes = (size_t)numSub * 16;
					if (fileBytes.size() < indexBytes)
					{
						cout << "ERROR: zarrIO: shard file " << filePath << " smaller than its own index" << endl;
						return 7;
					}
					const uint8_t* indexPtr = fileBytes.data() + (fileBytes.size() - indexBytes - 4);//trailing 4 bytes: crc32c of the index, not verified

					size_t subVoxels = (size_t)(subShape[0] * subShape[1] * subShape[2]);
					vector<uint8_t> subBuf(subVoxels * itemSize);

					int64_t subLinear = 0;
					for (int64_t sz = 0; sz < subGrid[0]; sz++)
					{
						for (int64_t sy = 0; sy < subGrid[1]; sy++)
						{
							for (int64_t sx = 0; sx < subGrid[2]; sx++, subLinear++)
							{
								uint64_t offset = readU64LE(indexPtr + subLinear * 16);
								uint64_t nbytes = readU64LE(indexPtr + subLinear * 16 + 8);
								if (offset == UINT64_MAX && nbytes == UINT64_MAX)
									continue;//sub-chunk not written == fill_value
								if (offset + nbytes > fileBytes.size())
								{
									cout << "ERROR: zarrIO: shard index entry out of bounds in " << filePath << endl;
									return 8;
								}

								if (!bloscDecompress(fileBytes.data() + offset, nbytes, subBuf.data(), subBuf.size()))
									return 9;

								int64_t subStart[3] = { sz * subShape[0], sy * subShape[1], sx * subShape[2] };
								placeChunk(outerChunkBuf.data(), chunkShape, subShape, subStart, subBuf.data(), itemSize);
							}
						}
					}
				}

				int64_t chunkStart[3] = { cz * chunkShape[0], cy * chunkShape[1], cx * chunkShape[2] };
				placeChunk((uint8_t*)outBuffer, arrayShape, chunkShape, chunkStart, outerChunkBuf.data(), itemSize);
			}
		}
	}

	return 0;
}

int zarr_writeArray3D(const string& path, const void* data, const int64_t dims[3], const string& dtype)
{
	size_t itemSize = zarrDtypeSize(dtype);
	if (itemSize == 0)
	{
		cout << "ERROR: zarrIO: unsupported output dtype '" << dtype << "'" << endl;
		return 1;
	}

	//dims[] is this project's own convention: dims[0]=x (fastest), dims[1]=y, dims[2]=z.
	//Zarr shape/chunk_shape axis order is (z,y,x) -- last axis fastest, same memory layout.
	int64_t arrayShape[3] = { dims[2], dims[1], dims[0] };//z,y,x

	//cap chunk size per axis: a single chunk covering the whole array works for our own
	//reader, but was found to crash Fiji's N5/Zarr importer -- its parallel bulk reader
	//has every worker thread decode the *same* giant chunk concurrently (a scenario real
	//multi-chunk datasets never hit), which throws inside its cache/codec layer. Chunking
	//normally sidesteps that entirely and is also just better Zarr practice.
	const int64_t maxChunkDim = 256;
	int64_t chunkShape[3], numChunks[3];
	for (int d = 0; d < 3; d++)
	{
		chunkShape[d] = min(maxChunkDim, arrayShape[d]);
		numChunks[d] = (arrayShape[d] + chunkShape[d] - 1) / chunkShape[d];
	}

	system(("mkdir \"" + path + "\" 2>nul").c_str());
	system(("mkdir \"" + path + "/c\" 2>nul").c_str());

	//written with the plain "bytes" codec (no blosc): c-blosc2 produces modern
	//blosc2-frame-format chunks that this project's own reader and the official
	//blosc2 Python package decode correctly, but that numcodecs' blosc decoder
	//(what the mainstream zarr-python package uses) fails on -- a known ecosystem
	//gap, not a correctness issue. Raw bytes sidestep that entirely for universal
	//readability, at the cost of larger files (see TODO.md for a compressed follow-up).
	size_t chunkVoxels = (size_t)(chunkShape[0] * chunkShape[1] * chunkShape[2]);
	vector<uint8_t> chunkBuf(chunkVoxels * itemSize);
	const uint8_t* src = (const uint8_t*)data;

	for (int64_t cz = 0; cz < numChunks[0]; cz++)
	{
		system(("mkdir \"" + path + "/c/" + to_string(cz) + "\" 2>nul").c_str());
		for (int64_t cy = 0; cy < numChunks[1]; cy++)
		{
			system(("mkdir \"" + path + "/c/" + to_string(cz) + "/" + to_string(cy) + "\" 2>nul").c_str());
			for (int64_t cx = 0; cx < numChunks[2]; cx++)
			{
				int64_t chunkStart[3] = { cz * chunkShape[0], cy * chunkShape[1], cx * chunkShape[2] };
				int64_t validZ = min(chunkShape[0], arrayShape[0] - chunkStart[0]);
				int64_t validY = min(chunkShape[1], arrayShape[1] - chunkStart[1]);
				int64_t validX = min(chunkShape[2], arrayShape[2] - chunkStart[2]);

				fill(chunkBuf.begin(), chunkBuf.end(), 0);
				for (int64_t z = 0; z < validZ; z++)
				{
					for (int64_t y = 0; y < validY; y++)
					{
						const uint8_t* srcRow = src + itemSize * ((chunkStart[2]) + arrayShape[2] * ((chunkStart[1] + y) + arrayShape[1] * (chunkStart[0] + z)));
						uint8_t* dstRow = chunkBuf.data() + itemSize * (chunkShape[2] * (y + chunkShape[1] * z));
						memcpy(dstRow, srcRow, itemSize * validX);
					}
				}

				string chunkPath = path + "/c/" + to_string(cz) + "/" + to_string(cy) + "/" + to_string(cx);
				ofstream chunkFile(chunkPath, ios::binary);
				if (!chunkFile.is_open())
				{
					cout << "ERROR: zarrIO: could not create chunk file " << chunkPath << endl;
					return 3;
				}
				chunkFile.write((const char*)chunkBuf.data(), chunkBuf.size());
				chunkFile.close();
			}
		}
	}

	int64_t nz = arrayShape[0], ny = arrayShape[1], nx = arrayShape[2];

	//plain 3D Zarr v3 array (shape order z,y,x, chunked). No OME-Zarr group wrapper --
	//that's viewer-interop polish, not required for the array itself to be a valid,
	//round-trippable Zarr v3 dataset.
	json arrayMeta;
	arrayMeta["zarr_format"] = 3;
	arrayMeta["node_type"] = "array";
	arrayMeta["shape"] = { nz, ny, nx };
	arrayMeta["data_type"] = dtype;
	arrayMeta["chunk_grid"] = { {"name", "regular"}, {"configuration", {{"chunk_shape", {chunkShape[0], chunkShape[1], chunkShape[2]}}}} };
	arrayMeta["chunk_key_encoding"] = { {"name", "default"}, {"configuration", {{"separator", "/"}}} };
	arrayMeta["fill_value"] = 0;
	arrayMeta["codecs"] = json::array({
		{ {"name", "bytes"}, {"configuration", {{"endian", "little"}}} }
		});
	arrayMeta["attributes"] = json::object();
	arrayMeta["storage_transformers"] = json::array();

	ofstream arrayJson(path + "/zarr.json", ios::binary);
	arrayJson << arrayMeta.dump(2);
	arrayJson.close();

	return 0;
}

int zarr_writePyramid3D(const string& groupPath, const uint16_t* data, const int64_t dims[3])
{
	system(("mkdir \"" + groupPath + "\" 2>nul").c_str());

	vector<int64_t> levelScaleFactors;//2^level, isotropic downsampling at every level
	int64_t curDims[3] = { dims[0], dims[1], dims[2] };
	const uint16_t* curData = data;
	vector<uint16_t> ownedBuf;//owns the current level's data once past level 0

	levelScaleFactors.push_back(1);
	int err = writeShardedLevel(groupPath + "/0", curData, curDims);
	if (err > 0)
		return err;

	int level = 0;
	//keep halving while the level about to be downsampled still has every dimension
	//>= 100 -- matches the depth BigStitcher's own pyramids reach in practice
	while (min(curDims[0], min(curDims[1], curDims[2])) >= 100)
	{
		int64_t nextDims[3] = { curDims[0] / 2, curDims[1] / 2, curDims[2] / 2 };
		vector<uint16_t> nextBuf((size_t)nextDims[0] * nextDims[1] * nextDims[2]);
		downsample2x(curData, curDims, nextBuf.data(), nextDims, 2);

		level++;
		err = writeShardedLevel(groupPath + "/" + to_string(level), nextBuf.data(), nextDims);
		if (err > 0)
			return err;

		levelScaleFactors.push_back((int64_t)1 << level);
		ownedBuf = move(nextBuf);
		curData = ownedBuf.data();
		curDims[0] = nextDims[0]; curDims[1] = nextDims[1]; curDims[2] = nextDims[2];
	}

	//OME-Zarr (NGFF 0.5) group wrapper describing the multiscale pyramid. Uses plain
	//(z,y,x) axes rather than BigStitcher's (t,c,z,y,x) -- our output is a single
	//volume, not a multi-timepoint/channel series, and NGFF allows the axes list to
	//match the array's actual rank.
	json groupMeta;
	groupMeta["zarr_format"] = 3;
	groupMeta["node_type"] = "group";
	json axes = json::array({
		{ {"name", "z"}, {"type", "space"} },
		{ {"name", "y"}, {"type", "space"} },
		{ {"name", "x"}, {"type", "space"} }
		});
	json datasets = json::array();
	for (size_t ii = 0; ii < levelScaleFactors.size(); ii++)
	{
		double s = (double)levelScaleFactors[ii];
		datasets.push_back({
			{"path", to_string(ii)},
			{"coordinateTransformations", json::array({ {{"type", "scale"}, {"scale", {s, s, s}}} })}
			});
	}
	groupMeta["attributes"]["ome"]["version"] = "0.5";
	groupMeta["attributes"]["ome"]["multiscales"] = json::array({
		{ {"version", "0.5"}, {"name", ""}, {"axes", axes}, {"datasets", datasets} }
		});

	ofstream groupJson(groupPath + "/zarr.json", ios::binary);
	groupJson << groupMeta.dump(2);
	groupJson.close();

	return 0;
}
