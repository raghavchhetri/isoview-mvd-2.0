//standalone dev tool: converts a KLB (uint16) file to a Zarr v3 array, for testing
//the Zarr v3 read path against known-good KLB-based deconvolution results.
#include "multiviewImage.h"
#include "zarrIO.h"
#include <iostream>

using namespace std;

int main(int argc, char** argv)
{
	if (argc < 3)
	{
		cout << "usage: tool_klb2zarr <input.klb> <output.zarr>" << endl;
		return 1;
	}

	multiviewImage<uint16_t> img;
	int err = img.readImage(string(argv[1]), -1);
	if (err > 0)
	{
		cout << "readImage failed: " << err << endl;
		return 1;
	}

	int64_t dims[3] = { img.dimsImgVec[0].dims[0], img.dimsImgVec[0].dims[1], img.dimsImgVec[0].dims[2] };
	cout << "read " << argv[1] << " dims=" << dims[0] << "x" << dims[1] << "x" << dims[2] << endl;

	err = zarr_writeArray3D(string(argv[2]), img.getPointer_CPU(0), dims, "uint16");
	if (err > 0)
	{
		cout << "zarr_writeArray3D failed: " << err << endl;
		return 1;
	}

	cout << "wrote " << argv[2] << endl;
	return 0;
}
