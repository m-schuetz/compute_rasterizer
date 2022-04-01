
#include <iostream>
#include <functional>

#include "unsuck.hpp"

using namespace std;


struct Point {
	double x = 0.0;
	double y = 0.0;
	double z = 0.0;
	double value = 0.0;
	uint64_t value_u64 = 0;
	uint8_t data[34];
};

struct Vector3 {
	double x = 0.0;
	double y = 0.0;
	double z = 0.0;
};

struct Header {
	int versionMajor = 0;
	int versionMinor = 0;
	uint64_t offsetToPointData = 0;
	int formatID = 0;
	int recordLength = 0;
	int64_t numPoints = 0;

	Vector3 min;
	Vector3 max;
	Vector3 scale;
	Vector3 offset;

};

Header parseHeader(string file) {

	auto source = readBinaryFile(file, 0, 375);

	Header header;

	header.versionMajor = read<uint8_t>(source, 24);
	header.versionMinor = read<uint8_t>(source, 25);
	header.offsetToPointData = read<uint32_t>(source, 96);
	header.formatID = read<uint8_t>(source, 104);
	header.recordLength = read<uint16_t>(source, 105);
	header.numPoints = read<uint32_t>(source, 107);
	if (header.versionMinor >= 4) {
		header.numPoints = read<uint64_t>(source, 247);
	}
	//header.numPoints = min(header.numPoints, 500'000'000);


	header.scale.x = read<double>(source, 131);
	header.scale.y = read<double>(source, 139);
	header.scale.z = read<double>(source, 147);

	header.offset.x = read<double>(source, 155);
	header.offset.y = read<double>(source, 163);
	header.offset.z = read<double>(source, 171);

	header.min.x = read<double>(source, 187);
	header.min.y = read<double>(source, 203);
	header.min.z = read<double>(source, 219);

	header.max.x = read<double>(source, 179);
	header.max.y = read<double>(source, 195);
	header.max.z = read<double>(source, 211);

	return header;
}

void readPoints(string file, int64_t pointsPerBatch, function<void(vector<uint8_t>&, int64_t, int64_t)> callback) {

	Header header = parseHeader(file);

	int64_t remaining = header.numPoints;
	int64_t fileOffset = header.offsetToPointData;
	int64_t processed = 0;

	while (remaining > 0) {

		int64_t currentBatchNumPoints = std::min(remaining, pointsPerBatch);
		int64_t currentBatchSize = currentBatchNumPoints * header.recordLength;

		auto buffer = readBinaryFile(file, fileOffset, currentBatchSize);

		callback(buffer, processed, currentBatchNumPoints);

		fileOffset += currentBatchSize;
		remaining = remaining - currentBatchNumPoints;
		processed += currentBatchNumPoints;
	}


}

int workgroupSize = 128;
int pointsPerThread = 100;
int batchSize = workgroupSize * pointsPerThread;
int batchBufferStride = 64;



//#define ORDER_RANDOM
#define ORDER_MORTON

#define PATTERN_8B_4B_SOA


vector<int> createRandomIndices(int numIndices){
	
	vector<int> indices(numIndices);
	for(int i = 0; i < numIndices; i++){
		indices[i] = i;
	}

	auto rng = std::default_random_engine {};
	std::shuffle(indices.begin(), indices.end(), rng);

	return indices;
}

int main(){

	string flags = "";

#if defined(ORDER_RANDOM)
	flags = flags + "_random";
#endif

#if defined(PATTERN_8B_4B_SOA)
	flags = flags + "_8b_4b_soa";
#endif

	
	string name = "lifeboat";
	//string name = "retz";
	//string file = "E:/dev/pointclouds/benchmark/retz/morton.las";
	//string file = "E:/dev/pointclouds/benchmark/endeavor/morton.las";
	string file = "E:/dev/pointclouds/benchmark/lifeboat/morton.las";
	//string file = "F:/pointclouds/benchmark/retz/morton.las";
	string outPath = "E:/temp/" + name + flags + ".bin";

	int pointsPerBatch = workgroupSize * pointsPerThread;

	auto header = parseHeader(file);

	int64_t processed = 0;
	int64_t batchesProcessed = 0;
	int64_t bitSize = 0;
	int64_t compressedBitSize = 0;
	int64_t numJumps = 0;
	int64_t diffCompressedBitSize = 0;

	auto outBuffer = make_shared<Buffer>(12 * header.numPoints);
	int numBatches = ceil(float(header.numPoints) / float(batchSize));
	auto batchBuffer = make_shared<Buffer>(batchBufferStride * numBatches);
	memset(batchBuffer->data, 0, batchBuffer->size);

	readPoints(file, pointsPerBatch, [&header, &processed, &batchesProcessed, &bitSize, &compressedBitSize, &outBuffer, &batchBuffer, numBatches, &numJumps, &diffCompressedBitSize](vector<uint8_t>& buffer, int64_t batch_startIndex, int64_t batch_numPoints) {

		//if (batchesProcessed != 0) {
		//	return;
		//}

		Vector3 min = { Infinity, Infinity, Infinity };
		Vector3 max = { -Infinity, -Infinity, -Infinity };

		Vector3 bbSize = {
			header.max.x - header.min.x,
			header.max.y - header.min.y,
			header.max.z - header.min.z
		};

		int offset_rgb = 0;
		if (header.formatID == 2) {
			offset_rgb = 20;
		} else if (header.formatID == 3) {
			offset_rgb = 28;
		}

		int64_t sum_r = 0;
		int64_t sum_g = 0;
		int64_t sum_b = 0;

#if defined(ORDER_RANDOM)
		auto indices = createRandomIndices(batch_numPoints);
#endif		

		double prev_x = 0.0;
		double prev_y = 0.0;
		double prev_z = 0.0;
		int64_t diff_bits = 0;

		for (int64_t i = 0; i < batch_numPoints; i++) {

#if defined(ORDER_RANDOM)
			int index = indices[i];
#else
			int index = i;
#endif		

			int64_t pointOffset = index * header.recordLength;
			int32_t X = read<int32_t>(buffer, pointOffset + 0);
			int32_t Y = read<int32_t>(buffer, pointOffset + 4);
			int32_t Z = read<int32_t>(buffer, pointOffset + 8);

			double x = double(X) * header.scale.x + header.offset.x;
			double y = double(Y) * header.scale.y + header.offset.y;
			double z = double(Z) * header.scale.z + header.offset.z;

			min.x = std::min(min.x, x);
			min.y = std::min(min.y, y);
			min.z = std::min(min.z, z);

			max.x = std::max(max.x, x);
			max.y = std::max(max.y, y);
			max.z = std::max(max.z, z);

			int32_t R = read<uint16_t>(buffer, pointOffset + offset_rgb + 0);
			int32_t G = read<uint16_t>(buffer, pointOffset + offset_rgb + 2);
			int32_t B = read<uint16_t>(buffer, pointOffset + offset_rgb + 4);

			int r = R > 255 ? R / 256 : R;
			int g = G > 255 ? G / 256 : G;
			int b = B > 255 ? B / 256 : B;

			sum_r += r;
			sum_g += g;
			sum_b += b;


			//if (processed < 50)
			{

				int diff_x = 1000.0 * (x - prev_x);
				int diff_y = 1000.0 * (y - prev_y);
				int diff_z = 1000.0 * (z - prev_z);

				// note: adding one more for sign
				int diff_bits_x = diff_x == 0 ? 1 : ceil(std::log2(abs(diff_x))) + 1; 
				int diff_bits_y = diff_y == 0 ? 1 : ceil(std::log2(abs(diff_y))) + 1;
				int diff_bits_z = diff_z == 0 ? 1 : ceil(std::log2(abs(diff_z))) + 1;

				diff_bits += diff_bits_x + diff_bits_y + diff_bits_z;


				if (diff_bits_x + diff_bits_y + diff_bits_z > 30 || processed < 10) {
					//cout << "diff bits: " << diff_bits_x << ", " << diff_bits_y << ", " << diff_bits_z << endl;
					numJumps++;
				}


			}


			prev_x = x;
			prev_y = y;
			prev_z = z;

			//{ // 13 bit
			//	double factor = 8192.0;
			//	uint64_t qx = uint32_t(factor * (x - header.min.x) / bbSize.x);
			//	uint64_t qy = uint32_t(factor * (y - header.min.y) / bbSize.y);
			//	uint64_t qz = uint32_t(factor * (z - header.min.z) / bbSize.z);

			//	int r = R > 255 ? R / 256 : R;
			//	int g = G > 255 ? G / 256 : G;
			//	int b = B > 255 ? B / 256 : B;

			//	uint64_t color = r | (g << 8) | (b << 16);

			//	uint64_t encoded = color | (qx << 24) | (qy << 37) | (qz << 50);

			//	outBuffer->set<uint64_t>(encoded, 8 * i);
			//}

			//{ // 10 bit, colors, struct of arrays
			//	double factor = 1024.0;
			//	uint64_t qx = uint32_t(factor * (x - header.min.x) / bbSize.x);
			//	uint64_t qy = uint32_t(factor * (y - header.min.y) / bbSize.y);
			//	uint64_t qz = uint32_t(factor * (z - header.min.z) / bbSize.z);

			//	int r = R > 255 ? R / 256 : R;
			//	int g = G > 255 ? G / 256 : G;
			//	int b = B > 255 ? B / 256 : B;

			//	uint32_t pos = (qx << 0) | (qy << 10) | (qz << 20);
			//	uint32_t color = r | (g << 8) | (b << 16);

			//	outBuffer->set<uint32_t>(pos, 4 * processed + 0);
			//	outBuffer->set<uint32_t>(color, 4 * processed + 4 * header.numPoints);
			//}

#if defined(PATTERN_8B_4B_SOA)
			{ // 16 bit, colors, struct of arrays
				double factor = 65536.0;
				uint64_t qx = uint32_t(factor * (x - header.min.x) / bbSize.x);
				uint64_t qy = uint32_t(factor * (y - header.min.y) / bbSize.y);
				uint64_t qz = uint32_t(factor * (z - header.min.z) / bbSize.z);

				uint64_t pos = (qx << 0) | (qy << 16) | (qz << 32);
				uint32_t color = r | (g << 8) | (b << 16);

				outBuffer->set<uint64_t>(pos, 8 * processed + 0);
				outBuffer->set<uint32_t>(color, 4 * processed + 8 * header.numPoints);
			}
#endif

			//{ // 32 byte
			//	outBuffer->set<float>(x, 12 * i + 0);
			//	outBuffer->set<float>(y, 12 * i + 4);
			//	outBuffer->set<float>(z, 12 * i + 8);
			//}


			//{ // 13 bit in 32 bit
			//	double factor = 8192.0;
			//	uint32_t qx = uint32_t(factor * (x - header.min.x) / bbSize.x);
			//	uint32_t qy = uint32_t(factor * (y - header.min.y) / bbSize.y);
			//	uint32_t qz = uint32_t(factor * (z - header.min.z) / bbSize.z);

			//	outBuffer->set<uint32_t>(qx, 12 * i + 0);
			//	outBuffer->set<uint32_t>(qy, 12 * i + 4);
			//	outBuffer->set<uint32_t>(qz, 12 * i + 8);

			//	//int r = R > 255 ? R / 256 : R;
			//	//int g = G > 255 ? G / 256 : G;
			//	//int b = B > 255 ? B / 256 : B;

			//	//uint64_t color = r | (g << 8) | (b << 16);

			//	//uint64_t encoded = color | (qx << 24) | (qy << 37) | (qz << 50);

			//	//outBuffer->set<uint64_t>(encoded, 8 * i);


			//}

			processed++;
			


		}

		
		batchBuffer->set<float>(min.x, batchesProcessed * batchBufferStride + 0);
		batchBuffer->set<float>(min.y, batchesProcessed * batchBufferStride + 4);
		batchBuffer->set<float>(min.z, batchesProcessed * batchBufferStride + 8);
		batchBuffer->set<float>(max.x, batchesProcessed * batchBufferStride + 12);
		batchBuffer->set<float>(max.y, batchesProcessed * batchBufferStride + 16);
		batchBuffer->set<float>(max.z, batchesProcessed * batchBufferStride + 20);
		
		batchBuffer->set<uint8_t>((sum_r / batch_numPoints), batchesProcessed * batchBufferStride + 24);
		batchBuffer->set<uint8_t>((sum_g / batch_numPoints), batchesProcessed * batchBufferStride + 25);
		batchBuffer->set<uint8_t>((sum_b / batch_numPoints), batchesProcessed * batchBufferStride + 26);
		batchBuffer->set<uint8_t>(255, batchesProcessed * batchBufferStride + 27);
		
		// #points in batch
		batchBuffer->set<int32_t>(batch_numPoints, batchesProcessed * batchBufferStride + 28);
		// offset (#points(
		batchBuffer->set<int32_t>(processed - batch_numPoints, batchesProcessed * batchBufferStride + 32);

		

		batchesProcessed++;

		Vector3 size = {
			max.x - min.x,
			max.y - min.y,
			max.z - min.z
		};

		int bitsX = std::ceil(std::log2(size.x * 1000.0));
		int bitsY = std::ceil(std::log2(size.y * 1000.0));
		int bitsZ = std::ceil(std::log2(size.z * 1000.0));
		int bitsXYZ = bitsX + bitsY + bitsZ;

		bitSize += batch_numPoints * 12 * 8;
		compressedBitSize += batch_numPoints * bitsXYZ;

		if((batchesProcessed % 100) == 0){
			cout << "progress: " << batchesProcessed << " / " << numBatches << endl;
		}

		int64_t full_bits = batch_numPoints * 12 * 8;
		double rate = double(diff_bits) / double(full_bits);

		diffCompressedBitSize += diff_bits;

		cout << "comression: " << formatNumber(100.0 * rate, 1) << "%" << endl;


	});

	writeBinaryFile(outPath, outBuffer);
	writeBinaryFile(outPath + ".batches", batchBuffer);

	//auto fout = ofstream(outPath, ios::binary);
	//fout.write(outBuffer->data_char, outBuffer->size);
	//fout.close();

	cout << "bitSize: " << formatNumber(bitSize) << endl;
	cout << "compressedBitSize: " << formatNumber(compressedBitSize) << endl;

	cout << "byteSize: " << formatNumber(bitSize / 8) << endl;
	cout << "compressedByteSize: " << formatNumber(compressedBitSize / 8) << endl;
	cout << "diffCompressedByteSize: " << formatNumber(diffCompressedBitSize / 8) << endl;

	cout << "#jumps: " << numJumps << endl;

	cout << "done" << endl;

	return 0;
}