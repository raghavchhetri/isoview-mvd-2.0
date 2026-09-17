#include "zarrIO.h"
#include <iostream>
#include <vector>
#include <cstdint>

using namespace std;

int main(int argc, char** argv)
{
	string path = argv[1];

	zarrArrayInfo info;
	int err = zarr_readHeader(path, info);
	if (err > 0)
	{
		cout << "readHeader failed: " << err << endl;
		return 1;
	}
	cout << "shape:";
	for (auto v : info.shape) cout << " " << v;
	cout << " dtype=" << info.dtype << endl;

	size_t ndims = info.shape.size();
	int64_t nz = info.shape[ndims - 3], ny = info.shape[ndims - 2], nx = info.shape[ndims - 1];
	size_t numVoxels = (size_t)nz * ny * nx;
	vector<uint16_t> buf(numVoxels, 0);

	err = zarr_readArray(path, buf.data(), buf.size() * sizeof(uint16_t), info);
	if (err > 0)
	{
		cout << "readArray failed: " << err << endl;
		return 1;
	}

	for (int64_t z = 0; z < nz; z += (nz / 16))
	{
		size_t nonzero = 0;
		uint16_t maxv = 0;
		for (int64_t y = 0; y < ny; y++)
			for (int64_t x = 0; x < nx; x++)
			{
				uint16_t v = buf[(size_t)x + nx * ((size_t)y + ny * z)];
				if (v != 0) nonzero++;
				if (v > maxv) maxv = v;
			}
		cout << "z=" << z << " nonzero_frac=" << (double)nonzero / (nx * ny) << " max=" << maxv << endl;
	}

	return 0;
}
