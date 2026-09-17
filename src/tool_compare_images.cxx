//standalone dev tool: reads two images (KLB or Zarr v3, auto-detected) and reports
//difference statistics, for validating the Zarr v3 pipeline against known-good KLB output.
#include "multiviewImage.h"
#include <iostream>
#include <cmath>

using namespace std;

int main(int argc, char** argv)
{
	if (argc < 3)
	{
		cout << "usage: tool_compare_images <fileA> <fileB>" << endl;
		return 1;
	}

	multiviewImage<float> a, b;
	int err = a.readImage(string(argv[1]), -1);
	if (err > 0) { cout << "readImage A failed: " << err << endl; return 1; }
	err = b.readImage(string(argv[2]), -1);
	if (err > 0) { cout << "readImage B failed: " << err << endl; return 1; }

	cout << "A dims: " << a.dimsImgVec[0].dims[0] << "x" << a.dimsImgVec[0].dims[1] << "x" << a.dimsImgVec[0].dims[2] << endl;
	cout << "B dims: " << b.dimsImgVec[0].dims[0] << "x" << b.dimsImgVec[0].dims[1] << "x" << b.dimsImgVec[0].dims[2] << endl;

	if (a.numElements(0) != b.numElements(0))
	{
		cout << "MISMATCH: different number of elements (" << a.numElements(0) << " vs " << b.numElements(0) << ")" << endl;
		return 2;
	}

	int64_t N = a.numElements(0);
	const float* pa = a.getPointer_CPU(0);
	const float* pb = b.getPointer_CPU(0);

	double sumAbsDiff = 0, maxAbsDiff = 0, sumA = 0, sumB = 0, sumAB = 0, sumA2 = 0, sumB2 = 0;
	int64_t numExactEqual = 0;
	for (int64_t ii = 0; ii < N; ii++)
	{
		double diff = fabs((double)pa[ii] - (double)pb[ii]);
		sumAbsDiff += diff;
		if (diff > maxAbsDiff) maxAbsDiff = diff;
		if (pa[ii] == pb[ii]) numExactEqual++;
		sumA += pa[ii]; sumB += pb[ii];
		sumAB += (double)pa[ii] * pb[ii];
		sumA2 += (double)pa[ii] * pa[ii];
		sumB2 += (double)pb[ii] * pb[ii];
	}

	double meanA = sumA / N, meanB = sumB / N;
	double cov = sumAB / N - meanA * meanB;
	double varA = sumA2 / N - meanA * meanA;
	double varB = sumB2 / N - meanB * meanB;
	double corr = cov / (sqrt(varA) * sqrt(varB));

	cout << "N=" << N << endl;
	cout << "meanA=" << meanA << " meanB=" << meanB << endl;
	cout << "meanAbsDiff=" << (sumAbsDiff / N) << " maxAbsDiff=" << maxAbsDiff << endl;
	cout << "fractionExactlyEqual=" << ((double)numExactEqual / N) << endl;
	cout << "pearsonCorrelation=" << corr << endl;

	return 0;
}
