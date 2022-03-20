
#include <iostream>
#include <execution>
#include <algorithm>
#include <functional>

#include "unsuck.hpp"


using namespace std;

template<class T>
struct SortPair{
	uint32_t index = 0;
	T value = 0;
};

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
	int numPoints = 0;

	Vector3 min;
	Vector3 max;
	Vector3 scale;
	Vector3 offset;

};

template<typename T>
T read(shared_ptr<Buffer> buffer, uint64_t offset) {
	T value;

	memcpy(&value, buffer->data_u8 + offset, sizeof(T));

	return value;
}

// from https://www.forceflow.be/2013/10/07/morton-encodingdecoding-through-bit-interleaving-implementations/
// method to seperate bits from a given integer 3 positions apart
inline uint64_t splitBy3(uint32_t a) {
	uint64_t x = a & 0x1fffff; // we only look at the first 21 bits
	x = (x | x << 32) & 0x1f00000000ffff; // shift left 32 bits, OR with self, and 00011111000000000000000000000000000000001111111111111111
	x = (x | x << 16) & 0x1f0000ff0000ff; // shift left 32 bits, OR with self, and 00011111000000000000000011111111000000000000000011111111
	x = (x | x << 8) & 0x100f00f00f00f00f; // shift left 32 bits, OR with self, and 0001000000001111000000001111000000001111000000001111000000000000
	x = (x | x << 4) & 0x10c30c30c30c30c3; // shift left 32 bits, OR with self, and 0001000011000011000011000011000011000011000011000011000100000000
	x = (x | x << 2) & 0x1249249249249249;

	return x;
}

// from https://www.forceflow.be/2013/10/07/morton-encodingdecoding-through-bit-interleaving-implementations/
inline uint64_t mortonEncode_magicbits(uint32_t x, uint32_t y, uint32_t z) {
	uint64_t answer = 0;
	answer |= splitBy3(x) | splitBy3(y) << 1 | splitBy3(z) << 2;

	return answer;
}

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
	header.numPoints = min(header.numPoints, 1'000'000'000);


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

vector<SortPair<int64_t>> sort_original(string file) {

	auto header = parseHeader(file);

	vector<SortPair<int64_t>> ordering;
	ordering.reserve(header.numPoints);
	printMemoryReport();

	for (uint32_t i = 0; i < header.numPoints; i++) {
		ordering[i] = {i, i};
	}

	return ordering;
}

vector<SortPair<int64_t>> sort_x(string file) {

	auto header = parseHeader(file);

	vector<SortPair<int64_t>> ordering;
	ordering.reserve(header.numPoints);
	printMemoryReport();

	int64_t processed = 0;

	readPoints(file, 10'000'000, [&ordering, &header, &processed](vector<uint8_t>& buffer, int64_t batch_startIndex, int64_t batch_numPoints) {

		for (int64_t i = 0; i < batch_numPoints; i++) {

			int64_t pointOffset = i * header.recordLength;
			int32_t X = read<int32_t>(buffer, pointOffset + 0);
			int32_t Y = read<int32_t>(buffer, pointOffset + 4);
			int32_t Z = read<int32_t>(buffer, pointOffset + 8);

			double x = double(X) * header.scale.x + header.offset.x;
			double y = double(Y) * header.scale.y + header.offset.y;
			double z = double(Z) * header.scale.z + header.offset.z;

			SortPair<int64_t> pair;
			pair.index = processed;
			pair.value = X;

			ordering.push_back(pair);

			processed++;
		}

		cout << "points processed: " << formatNumber(processed) << endl;
	});

	return ordering;
}

vector<SortPair<double>> sort_shuffle(string file) {

	auto header = parseHeader(file);

	vector<SortPair<double>> ordering;
	ordering.reserve(header.numPoints);
	printMemoryReport();

	int64_t processed = 0;

	readPoints(file, 10'000'000, [&ordering, &header, &processed](vector<uint8_t>& buffer, int64_t batch_startIndex, int64_t batch_numPoints) {

		auto randomNumbers = random(0.0, 100'000'000.0, int(batch_numPoints));

		for (int64_t i = 0; i < batch_numPoints; i++) {

			int64_t pointOffset = i * header.recordLength;
			int32_t X = read<int32_t>(buffer, pointOffset + 0);
			int32_t Y = read<int32_t>(buffer, pointOffset + 4);
			int32_t Z = read<int32_t>(buffer, pointOffset + 8);

			double x = double(X) * header.scale.x + header.offset.x;
			double y = double(Y) * header.scale.y + header.offset.y;
			double z = double(Z) * header.scale.z + header.offset.z;

			SortPair<double> pair;
			pair.index = processed;
			pair.value = randomNumbers[i];

			ordering.push_back(pair);

			processed++;
		}

		cout << "points processed: " << formatNumber(processed) << endl;
	});

	return ordering;
}


vector<SortPair<uint64_t>> sort_morton(string file) {

	auto header = parseHeader(file);

	vector<SortPair<uint64_t>> ordering;
	ordering.reserve(header.numPoints);
	printMemoryReport();

	Vector3 min = header.min;
	Vector3 size = {
		header.max.x - header.min.x,
		header.max.y - header.min.y,
		header.max.z - header.min.z
	};
	double cubeSize = std::max(std::max(size.x, size.y), size.z);
	Vector3 max = {
		min.x + cubeSize,
		min.y + cubeSize,
		min.z + cubeSize,
	};

	double factor = pow(2.0, 21.0);

	int64_t processed = 0;

	readPoints(file, 10'000'000, [&ordering, &header, &processed, min, cubeSize, factor](vector<uint8_t>& buffer, int64_t batch_startIndex, int64_t batch_numPoints) {

		for (int64_t i = 0; i < batch_numPoints; i++) {

			int64_t pointOffset = i * header.recordLength;
			int32_t X = read<int32_t>(buffer, pointOffset + 0);
			int32_t Y = read<int32_t>(buffer, pointOffset + 4);
			int32_t Z = read<int32_t>(buffer, pointOffset + 8);

			double x = double(X) * header.scale.x + header.offset.x;
			double y = double(Y) * header.scale.y + header.offset.y;
			double z = double(Z) * header.scale.z + header.offset.z;

			double nx = (x - min.x) / cubeSize;
			double ny = (y - min.y) / cubeSize;
			double nz = (z - min.z) / cubeSize;

			uint32_t mX = uint32_t(nx * factor);
			uint32_t mY = uint32_t(ny * factor);
			uint32_t mZ = uint32_t(nz * factor);

			uint64_t mortonCode = mortonEncode_magicbits(mX, mY, mZ);

			SortPair<uint64_t> pair;
			pair.index = processed;
			pair.value = mortonCode;

			ordering.push_back(pair);

			processed++;
		}

		cout << "points processed: " << formatNumber(processed) << endl;
		});

	return ordering;
}

vector<SortPair<uint64_t>> sort_morton_shuffled(string file) {
	
	auto mortonOrdering = sort_morton(file);

	auto parallel = std::execution::par_unseq;
	sort(parallel, mortonOrdering.begin(), mortonOrdering.end(), [](auto& a, auto& b) {
		return a.value < b.value;
	});

	int granularity = 128;
	int n = mortonOrdering.size();

	vector<SortPair<uint64_t>> ordering;
	ordering.reserve(n);
	for (int i = 0; i < n; i += granularity) {

		int64_t randomValue = random(0.0, 100'000'000.0);

		for (int j = 0; j < granularity && (i + j) < n; j++) {

			SortPair<uint64_t> pair;
			pair.index = mortonOrdering[i + j].index;
			pair.value = randomValue;

			ordering.push_back(pair);
		}

	}

	return ordering;
}

int main() {

	cout << "start" << endl;

	string file = "F:/temp/wgtest/banyunibo_laserscans/merged.las";
	string targetDir = "F:/temp/wgtest/banyunibo_laserscans/";
	//string targetDir = "D:/temp/sorted";
	string targetPath = targetDir + "/morton.las";

	// READ AND PARSE LAS FILE
	cout << "pass 1: read file, generate sort keys" << endl;

	auto header = parseHeader(file);

	
	{
		printMemoryReport();

		//auto ordering = sort_original(file);
		//auto ordering = sort_x(file);
		//auto ordering = sort_shuffle(file);
		auto ordering = sort_morton(file);
		//auto ordering = sort_morton_shuffled(file);

		printMemoryReport();

		cout << "sort source<->target mapping" << endl;
		auto parallel = std::execution::par_unseq;
		sort(parallel, ordering.begin(), ordering.end(), [](auto& a, auto& b) {
			return a.value < b.value;
		});



		{
			fs::create_directories(targetDir);
			auto of = fstream(targetPath, ios::out | ios::binary);

			auto headerBuffer = readBinaryFile(file, 0, header.offsetToPointData);

			if (header.versionMinor >= 4) {
				reinterpret_cast<uint64_t*>(headerBuffer.data() + 247)[0] = header.numPoints;
				//header.numPoints = read<uint64_t>(source, 247);
			}else{
				reinterpret_cast<uint32_t*>(headerBuffer.data() + 107)[0] = header.numPoints;
				//header.numPoints = read<uint32_t>(source, 107);		
			}

			of.write(reinterpret_cast<char*>(headerBuffer.data()), headerBuffer.size());

			int64_t pointsPerBatch = 100'000'000;
			int64_t remaining = header.numPoints;
			int64_t processed = 0;
			Buffer target(pointsPerBatch * header.recordLength);

			while (remaining > 0) {
				int64_t currentBatchNumPoints = std::min(remaining, pointsPerBatch);
				int64_t currentBatchSize = currentBatchNumPoints * header.recordLength;

				cout << "process batch, " << currentBatchNumPoints << " points" << endl;

				// iterate through chunks of input file, and transfer relevant points from chunks to current batch
				readPoints(file, 100'000'000, [&processed, currentBatchNumPoints, &ordering, &target, &header](vector<uint8_t>& source, int64_t batch_firstIndex, int64_t batch_numPoints) {

					int64_t batch_lastIndex = batch_firstIndex + batch_numPoints - 1;

					for (int64_t i = 0; i < currentBatchNumPoints; i++) {
						int64_t target_index = processed + i;
						int64_t source_index = ordering[target_index].index;

						bool sourceBatchHasPoint = (source_index >= batch_firstIndex) && (source_index <= batch_lastIndex);
						if (sourceBatchHasPoint) {
							int64_t sourceBufferIndex = source_index - batch_firstIndex;
							int64_t targetBufferIndex = i;

							memcpy(
								target.data_u8 + targetBufferIndex * header.recordLength,
								source.data() + sourceBufferIndex * header.recordLength, 
								header.recordLength);
						}
					}

				});


				cout << "write: #points: " << formatNumber(currentBatchNumPoints) << ", #bytes: " << formatNumber(currentBatchSize) << endl;
				of.write(target.data_char, currentBatchSize);
				of.flush();

				remaining -= currentBatchNumPoints;
				processed += currentBatchNumPoints;

				cout << "end batch" << endl;
			}




			of.close();
		}

	}
	

	cout << "done" << endl;

	return 0;
}

