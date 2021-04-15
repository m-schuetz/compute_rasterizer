
#include <iostream>

#include "unsuck.hpp"

using namespace std;

struct SortPair {
	uint32_t index = 0;
	int64_t value = 0;
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

//void sort_morton(vector<Point>& points, Vector3 min, Vector3 max) {
//
//
//
//}

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

shared_ptr<Buffer> sort_morton(string targetDir, shared_ptr<Buffer> source, Header header, vector<Point>& points) {
	cout << "sort: morton" << endl;

	// TODO:

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

	for (int64_t i = 0; i < header.numPoints; i++) {
		Point& point = points[i];

		double nx = (point.x - min.x) / cubeSize;
		double ny = (point.y - min.y) / cubeSize;
		double nz = (point.z - min.z) / cubeSize;

		uint32_t X = uint32_t(nx * factor);
		uint32_t Y = uint32_t(ny * factor);
		uint32_t Z = uint32_t(nz * factor);

		uint64_t mortonCode = mortonEncode_magicbits(X, Y, Z);

		point.value_u64 = mortonCode;
	}

	//sort_morton(points, min, max);


	sort(points.begin(), points.end(), [](auto& a, auto& b) {
		return a.value_u64 < b.value_u64;
		});

	shared_ptr<Buffer> target = make_shared<Buffer>(source->size);

	// copy up to exclusive first point
	memcpy(target->data, source->data, header.offsetToPointData);

	// now copy points according to ordering
	for (int64_t i = 0; i < header.numPoints; i++) {
		Point point = points[i];

		uint64_t targetOffset = header.offsetToPointData + i * header.recordLength;

		memcpy(target->data_u8 + targetOffset, point.data, header.recordLength);
	}

	string targetPath = targetDir + "/sort_morton.las";
	cout << "writing file " << targetPath << endl;
	writeBinaryFile(targetPath, *target);

	return target;
}

shared_ptr<Buffer> sort_X(string targetDir, shared_ptr<Buffer> source, Header header, vector<Point>& points) {
	cout << "sort: x-axis" << endl;
	
	sort(points.begin(), points.end(), [](auto &a, auto& b) {
		return a.x < b.x;
	});

	shared_ptr<Buffer> target = make_shared<Buffer>(source->size);

	// copy up to exclusive first point
	memcpy(target->data, source->data, header.offsetToPointData);

	// now copy points according to ordering
	for (int64_t i = 0; i < header.numPoints; i++) {
		Point point = points[i];

		uint64_t targetOffset = header.offsetToPointData + i * header.recordLength;

		memcpy(target->data_u8 + targetOffset, point.data, header.recordLength);
	}

	string targetPath = targetDir + "/sort_x.las";
	cout << "writing file " << targetPath << endl;
	writeBinaryFile(targetPath, *target);

	return target;
}

shared_ptr<Buffer> sort_shuffle(string targetDir, shared_ptr<Buffer> source, Header header, vector<Point>& points) {

	cout << "sort: shuffle" << endl;

	vector<double> randomValues = random(0.0, 100'000'000.0, header.numPoints);
	for (int64_t i = 0; i < header.numPoints; i++) {
		points[i].value = randomValues[i];
	}

	sort(points.begin(), points.end(), [](auto& a, auto& b) {
		return a.value < b.value;
	});

	shared_ptr<Buffer> target = make_shared<Buffer>(source->size);

	// copy up to exclusive first point
	memcpy(target->data, source->data, header.offsetToPointData);

	// now copy points according to ordering
	for (int64_t i = 0; i < header.numPoints; i++) {
		Point point = points[i];

		uint64_t targetOffset = header.offsetToPointData + i * header.recordLength;

		memcpy(target->data_u8 + targetOffset, point.data, header.recordLength);
	}

	string targetPath = targetDir + "/sort_shuffled.las";
	cout << "writing file " << targetPath << endl;
	writeBinaryFile(targetPath, *target);

	return target;
}

Header parseHeader(shared_ptr<Buffer> source) {

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

vector<Point> parsePoints(shared_ptr<Buffer> source, Header header) {

	vector<Point> points;
	points.reserve(header.numPoints);

	for (int64_t i = 0; i < header.numPoints; i++) {
		Point point;

		uint64_t pointOffset = header.offsetToPointData + i * header.recordLength;
		memcpy(point.data, source->data_u8 + pointOffset, header.recordLength);

		int32_t X = read<int32_t>(source, pointOffset + 0);
		int32_t Y = read<int32_t>(source, pointOffset + 4);
		int32_t Z = read<int32_t>(source, pointOffset + 8);

		point.x = double(X) * header.scale.x + header.offset.x;
		point.y = double(Y) * header.scale.y + header.offset.y;
		point.z = double(Z) * header.scale.z + header.offset.z;

		points.push_back(point);
	}

	return points;
}

int main() {

	cout << "start" << endl;

	string file = "D:/dev/pointclouds/riegl/retz.las";
	//string file = "D:/dev/pointclouds/weiss/pos8_lifeboats.las";
	//string file = "D:/dev/pointclouds/archpro/heidentor.las";
	//string file = "D:/dev/pointclouds/lion.las";

	// READ AND PARSE LAS FILE
	cout << "reading file" << endl;
	auto source = readBinaryFile(file);

	auto header = parseHeader(source);
	auto points = parsePoints(source, header);

	string targetDir = "D:/temp/sorted";
	fs::create_directories(targetDir);
	
	sort_shuffle(targetDir, source, header, points);
	sort_morton(targetDir, source, header, points);
	sort_X(targetDir, source, header, points);


	// WRITE FILE
	//string targetPath = "D:/temp/test.las";
	//cout << "writing file " << targetPath << endl;
	//writeBinaryFile(targetPath, *sorted);
	

	cout << "done" << endl;


	return 0;
}

