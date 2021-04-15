
#pragma once

#include <string>
#include <vector>
#include <queue>
#include <fstream>
#include <chrono>
#include <thread>
#include <mutex>
#include <atomic>
#include <iostream>
#include <algorithm>
#include <sstream>
#include <future>
#include <cstdio>
#include <filesystem>
#include <functional>

#include "BArray.h"
#include "modules/CppUtils/CppUtils.h"

// for benchmarking cold start performance
#define DISABLE_FILE_CACHE

#ifdef DISABLE_FILE_CACHE
#include <Windows.h>
#endif

namespace fs = std::filesystem;

namespace LASLoaderThreaded {

	using std::string;
	using std::vector;
	using std::ifstream;
	using std::ios;
	using std::cout;
	using std::endl;
	using std::streamsize;
	using std::thread;
	using std::mutex;
	using std::unique_lock;
	using std::lock_guard;
	using std::atomic;
	using std::min;
	using std::stringstream;
	using std::function;
	using std::queue;

	struct LASHeader {

		uint16_t fileSourceID = 0;
		uint16_t globalEncoding = 0;
		uint32_t project_ID_GUID_data_1 = 0;
		uint16_t project_ID_GUID_data_2 = 0;
		uint16_t project_ID_GUID_data_3 = 0;
		uint16_t project_ID_GUID_data_4 = 0;

		uint8_t versionMajor = 0;
		uint8_t versionMinor = 0;
		string systemIdentifier = "";
		string generatingSoftware = "";
		uint16_t fileCreationDay = 0;
		uint16_t fileCreationYear = 0;

		uint16_t headerSize = 0;
		uint32_t offsetToPointData = 0;
		uint32_t numVLRs = 0;
		uint8_t pointDataFormat = 0;
		uint16_t pointDataRecordLength = 0;
		uint64_t numPoints = 0;
		vector<uint32_t> numPointsPerReturn;

		double scaleX = 0;
		double scaleY = 0;
		double scaleZ = 0;

		double offsetX = 0;
		double offsetY = 0;
		double offsetZ = 0;

		double maxX = 0;
		double minX = 0;

		double maxY = 0;
		double minY = 0;

		double maxZ = 0;
		double minZ = 0;

	};

	struct VariableLengthRecord {
		string userID;
		uint16_t recordID = 0;
		uint16_t recordLengthAfterHeader = 0;
		string description;

		vector<char> buffer;
	};

	struct ExtraBytes {

		uint8_t reserved[2] = {0, 0};
		uint8_t dataType = 0;
		uint8_t options = 0;
		int8_t name[32] = {0};
		uint8_t unused[4] = {0};
		uint8_t noData[24] = {0};
		uint8_t min[24] = {0};
		uint8_t max[24] = {0};
		double scale[3] = {0.0, 0.0, 0.0};
		double offset[3] = {0.0, 0.0, 0.0};
		int8_t description[32] = {0};

		vector<int> dataTypeSizes = { 0, 1, 1, 2, 2, 4, 4, 8, 8, 4, 8, 2, 2, 4, 4, 8, 8, 16, 16, 8, 16, 3, 3, 6, 6, 12, 12, 24, 24, 12, 24 };

		int bytes() {
			return dataTypeSizes[dataType];
		}

		vector<uint64_t> noDataU64() {
			vector<uint64_t> value = {
				reinterpret_cast<uint64_t*>(noData)[0],
				reinterpret_cast<uint64_t*>(noData)[1],
				reinterpret_cast<uint64_t*>(noData)[2]
			};

			return value;
		}

		vector<int64_t> noDataI64() {
			vector<int64_t> value = {
				reinterpret_cast<int64_t*>(noData)[0],
				reinterpret_cast<int64_t*>(noData)[1],
				reinterpret_cast<int64_t*>(noData)[2]
			};

			return value;
		}

		vector<double> noDataDouble() {
			vector<double> value = {
				reinterpret_cast<double*>(noData)[0],
				reinterpret_cast<double*>(noData)[1],
				reinterpret_cast<double*>(noData)[2]
			};

			return value;
		}

		vector<uint64_t> minU64() {
			vector<uint64_t> value = {
				reinterpret_cast<uint64_t*>(min)[0],
				reinterpret_cast<uint64_t*>(min)[1],
				reinterpret_cast<uint64_t*>(min)[2]
			};

			return value;
		}

		vector<int64_t> minI64() {
			vector<int64_t> value = {
				reinterpret_cast<int64_t*>(min)[0],
				reinterpret_cast<int64_t*>(min)[1],
				reinterpret_cast<int64_t*>(min)[2]
			};

			return value;
		}

		vector<double> minDouble() {
			vector<double> value = {
				reinterpret_cast<double*>(min)[0],
				reinterpret_cast<double*>(min)[1],
				reinterpret_cast<double*>(min)[2]
			};

			return value;
		}

		vector<uint64_t> maxU64() {
			vector<uint64_t> value = {
				reinterpret_cast<uint64_t*>(max)[0],
				reinterpret_cast<uint64_t*>(max)[1],
				reinterpret_cast<uint64_t*>(max)[2]
			};

			return value;
		}

		vector<int64_t> maxI64() {
			vector<int64_t> value = {
				reinterpret_cast<int64_t*>(max)[0],
				reinterpret_cast<int64_t*>(max)[1],
				reinterpret_cast<int64_t*>(max)[2]
			};

			return value;
		}

		vector<double> maxDouble() {
			vector<double> value = {
				reinterpret_cast<double*>(max)[0],
				reinterpret_cast<double*>(max)[1],
				reinterpret_cast<double*>(max)[2]
			};

			return value;
		}

	};

	struct XYZRGBA {
		float x;
		float y;
		float z;
		uint8_t r;
		uint8_t g;
		uint8_t b;
		uint8_t a;
	};

	struct DataFormat {
		int size;
	};

	struct Attribute {

		string name = "no name";
		//uint8_t* data;
		BArray* data = nullptr;

		int byteOffset;
		int bytes = 0;

		int elements = 0;
		int elementSize = 0;

		function<void(uint8_t* binaryChunk, int&)> read;

		// min and max can be char, int, double, etc., 
		//uint8_t* min[24];
		//uint8_t* max[24];

		vector<double> scale = { 1.0, 1.0, 1.0 };
		vector<double> offset = { 0.0, 0.0, 0.0 };

		Attribute() {

		}

		//int8_t readInt8(int byteOffset) {
		//	int8_t  v = reinterpret_cast<int8_t*>(data + byteOffset)[0];
		//
		//	v = double(v) * scale[0] + offset[0];
		//
		//	return v;
		//}
		//
		//uint8_t  readUint8(int byteOffset) {
		//	uint8_t  v = reinterpret_cast<uint8_t*>(data + byteOffset)[0];
		//
		//	v = double(v) * scale[0] + offset[0];
		//
		//	return v;
		//}
		//
		//int16_t readInt16(int byteOffset) {
		//	int16_t v = reinterpret_cast<int16_t*>(data + byteOffset)[0];
		//
		//	v = double(v) * scale[0] + offset[0];
		//
		//	return v;
		//}
		//
		//uint16_t readUint16(int byteOffset) {
		//	uint16_t v = reinterpret_cast<uint16_t*>(data + byteOffset)[0];
		//
		//	v = double(v) * scale[0] + offset[0];
		//
		//	return v;
		//}
		//
		//int32_t readInt32(int byteOffset) {
		//	int32_t v = reinterpret_cast<int32_t*>(data + byteOffset)[0];
		//
		//	v = double(v) * scale[0] + offset[0];
		//
		//	return v;
		//}
		//
		//uint32_t readUint32(int byteOffset) {
		//	uint32_t v = reinterpret_cast<uint32_t*>(data + byteOffset)[0];
		//
		//	v = double(v) * scale[0] + offset[0];
		//
		//	return v;
		//}
		//
		//float readFloat(int byteOffset) {
		//	float v = reinterpret_cast<float*>(data + byteOffset)[0];
		//
		//	v = double(v) * scale[0] + offset[0];
		//
		//	return v;
		//}
		//
		//float readDouble(int byteOffset) {
		//	double v = reinterpret_cast<double*>(data + byteOffset)[0];
		//
		//	v = v * scale[0] + offset[0];
		//
		//	return v;
		//}


	};

	struct Points {

		vector<XYZRGBA> xyzrgba;
		vector<Attribute> attributes;

		uint32_t size = 0;

		Points() {

		}

	};

	class LASLoader {

	public:

		string file;

		LASHeader header;
		vector<VariableLengthRecord> variableLengthRecords;

		vector<char> headerBuffer;
		queue<BArray*> binaryChunks;
		vector<Points*> chunks;

		mutex mtx_processing_chunk;
		mutex mtc_access_chunk;
		mutex mtx_binary_chunks;

		atomic<uint64_t> numLoaded = 0;
		atomic<uint64_t> numParsed = 0;

		uint32_t defaultChunkSize = 500'000;

		LASLoader(string file) {
			this->file = file;

			loadHeader();
			loadVariableLengthRecords();

			{ // print extra bytes
				vector<ExtraBytes> ebs = getExtraBytes();

				cout << endl;
				cout << "== EXTRA BYTE ATTRIBUTES == " << endl;
				for (ExtraBytes& eb : ebs) {
					string name = string(&eb.name[0], &eb.name[31]);
					cout << "extra byte, name: " << name << endl;
				}
				cout << "=================";
				cout << endl;
			}

			{ // print attributes
				cout << endl;
				cout << "== ATTRIBUTES == " << endl;
				for (Attribute& a : getAttributes()) {
					cout << a.name << ": " << a.byteOffset << ", " << a.bytes << endl;
				}
				cout << "=================";
				cout << endl;
			}

			createBinaryLoaderThread();
			createBinaryChunkParserThreadDynamicAttributeArray();
			createBinaryChunkParserThreadDynamicAttributeArray();
			createBinaryChunkParserThreadDynamicAttributeArray();
			createBinaryChunkParserThreadDynamicAttributeArray();
		}

		~LASLoader() {

		}

		void waitUntilFullyParsed() {

			while (!fullyParsed()) {
				std::this_thread::sleep_for(std::chrono::milliseconds(100));
			}

		}

		bool fullyLoaded() {
			bool result = numLoaded == header.numPoints;

			return result;
		}

		bool fullyParsed() {
			bool result = numParsed == header.numPoints;

			return result;
		}

		bool hasChunkAvailable() {
			lock_guard<mutex> lock(mtc_access_chunk);
			bool result = chunks.size() > 0;

			return result;
		}

		bool allChunksServed() {
			lock_guard<mutex> lock(mtc_access_chunk);
			bool result = fullyParsed() && chunks.size() == 0;

			return result;
		}

		Points* getNextChunk() {
			lock_guard<mutex> lock(mtc_access_chunk);

			Points* chunk = nullptr;

			if (chunks.size() > 0) {
				chunk = chunks.back();
				chunks.pop_back();
			}

			return chunk;
		}

	//private:



		void loadHeader() {
			ifstream fhandle(file, ios::binary | ios::ate);
			streamsize size = fhandle.tellg();
			fhandle.seekg(0, std::ios::beg);

			int headerSize = 227;
			headerBuffer.resize(headerSize);

			if (fhandle.read(headerBuffer.data(), headerSize)) {
				cout << "header buffer loaded" << endl;
			}
			else {
				cout << "header buffer not loaded :x" << endl;
			}

			header.versionMajor = reinterpret_cast<uint8_t*>(headerBuffer.data() + 24)[0];
			header.versionMinor = reinterpret_cast<uint8_t*>(headerBuffer.data() + 25)[0];

			if (header.versionMajor >= 1 && header.versionMinor >= 4) {
				fhandle.seekg(0, std::ios::beg);

				headerSize = 375;
				headerBuffer.resize(headerSize);

				if (fhandle.read(headerBuffer.data(), headerSize)) {
					cout << "extended header buffer loaded" << endl;
				}
				else {
					cout << "extended header buffer not loaded :(" << endl;
				}
			}

			// TODO probably should use that instead of hardcoding 227 and 375?
			header.headerSize = reinterpret_cast<uint16_t*>(headerBuffer.data() + 94)[0];

			header.offsetToPointData = reinterpret_cast<uint32_t*>(headerBuffer.data() + 96)[0];
			header.numVLRs = reinterpret_cast<uint32_t*>(headerBuffer.data() + 100)[0];
			header.pointDataFormat = reinterpret_cast<uint8_t*>(headerBuffer.data() + 104)[0];
			header.pointDataRecordLength = reinterpret_cast<uint16_t*>(headerBuffer.data() + 105)[0];
			header.numPoints = reinterpret_cast<uint32_t*>(headerBuffer.data() + 107)[0];
			
			header.scaleX = reinterpret_cast<double*>(headerBuffer.data() + 131)[0];
			header.scaleY = reinterpret_cast<double*>(headerBuffer.data() + 139)[0];
			header.scaleZ = reinterpret_cast<double*>(headerBuffer.data() + 147)[0];

			header.offsetX = reinterpret_cast<double*>(headerBuffer.data() + 155)[0];
			header.offsetY = reinterpret_cast<double*>(headerBuffer.data() + 163)[0];
			header.offsetZ = reinterpret_cast<double*>(headerBuffer.data() + 171)[0];

			header.minX = reinterpret_cast<double*>(headerBuffer.data() + 187)[0];
			header.minY = reinterpret_cast<double*>(headerBuffer.data() + 203)[0];
			header.minZ = reinterpret_cast<double*>(headerBuffer.data() + 219)[0];

			header.maxX = reinterpret_cast<double*>(headerBuffer.data() + 179)[0];
			header.maxY = reinterpret_cast<double*>(headerBuffer.data() + 195)[0];
			header.maxZ = reinterpret_cast<double*>(headerBuffer.data() + 211)[0];

			if (header.versionMajor >= 1 && header.versionMinor >= 4) {
				header.numPoints = reinterpret_cast<uint64_t*>(headerBuffer.data() + 247)[0];
			}

			//int maxPoints = 3 * 134'000'000;
			//int maxPoints = 500'000'000;
			int maxPoints = 1'000'000'000;
			//int maxPoints = 400'000'000;
			if (header.numPoints > maxPoints) {
				cout << "#points limited to " << maxPoints << ", was " << header.numPoints << endl;
				header.numPoints = maxPoints;
			}

			cout << "header.headerSize: " << header.headerSize << endl;
			cout << "header.offsetToPointData: " << header.offsetToPointData << endl;
			cout << "header.pointDataFormat: " << header.pointDataFormat << endl;
			cout << "header.pointDataRecordLength: " << header.pointDataRecordLength << endl;
			cout << "header.numPoints: " << header.numPoints << endl;

			fhandle.close();
		}

		void loadVariableLengthRecords() {
			ifstream fhandle(file, ios::binary | ios::ate);
			streamsize size = fhandle.tellg();

			//auto setReadPos = [&fhandle](int offset){fhandle.seekg(offset, std::ios::beg);};

			auto readBytes = [&fhandle](int offset, int length) {
				fhandle.seekg(offset, std::ios::beg);
				vector<char> data(length);

				fhandle.read(data.data(), length);

				return data;
			};

			int vlrHeaderSize = 54;
			int offset = header.headerSize;

			for (uint32_t i = 0; i < header.numVLRs; i++) {
				//fhandle.seekg(offset, std::ios::beg);
				//setReadPos(offset);

				VariableLengthRecord vlr;

				vector<char> buffer = readBytes(offset, vlrHeaderSize);

				vlr.userID = string(buffer.begin() + 2, buffer.begin() + 2 + 16);
				vlr.recordID = reinterpret_cast<uint16_t*>(buffer.data() + 18)[0];
				vlr.recordLengthAfterHeader = reinterpret_cast<uint16_t*>(buffer.data() + 20)[0];
				vlr.description = string(buffer.begin() + 22, buffer.begin() + 22 + 32);

				vlr.buffer = readBytes(offset + vlrHeaderSize, vlr.recordLengthAfterHeader);

				variableLengthRecords.emplace_back(vlr);

				offset += 54 + vlr.recordLengthAfterHeader;
			}

			for (VariableLengthRecord& vlr : variableLengthRecords) {
				cout << "==== VLR start ===" << endl;

				cout << "description: " << vlr.description << endl;
				cout << "length: " << vlr.recordLengthAfterHeader << endl;
				cout << "recordID: " << vlr.recordID << endl;


				cout << "==== VLR end ===" << endl;
			}
		}

		vector<ExtraBytes> getExtraBytes() {

			vector<ExtraBytes> extraBytes;

			auto vlrExtraBytesIT = std::find_if(variableLengthRecords.begin(), variableLengthRecords.end(), [](const VariableLengthRecord& vlr) {
				return vlr.recordID == 4;
			});

			if (vlrExtraBytesIT != variableLengthRecords.end()) {
				VariableLengthRecord vlr = *vlrExtraBytesIT;

				int numAttributes = vlr.recordLengthAfterHeader / 192;

				for (int i = 0; i < numAttributes; i++) {

					ExtraBytes attribute;

					memcpy(reinterpret_cast<char*>(&attribute), vlr.buffer.data() + i * 192, 192);

					extraBytes.push_back(attribute);

				}
			}

			return extraBytes;

		}

		int getOffsetIntensity() {
			return 12;
		}

		int getOffsetReturnNumber() {
			return 14;
		}

		int getOffsetClassification() {
			int format = header.pointDataFormat;

			if (format <= 5) {
				return 15;
			}
			else if (format == 6 || format == 7) {
				return 16;
			}

			return 0;
		}

		int getOffsetScanAngleRank() {
			int format = header.pointDataFormat;

			if (format != 5) {
				return 16;
			}
			else {
				return 0;
			}
		}

		int getOffsetUserData() {
			return 17;
		}

		int getOffsetPointSourceID() {
			int format = header.pointDataFormat;

			if (format <= 5) {
				return 18;
			}
			else if (format == 6 || format == 7) {
				return 20;
			}

			return 0;
		}



		int getOffsetGpsTime() {
			int format = header.pointDataFormat;

			if (format == 1) {
				return 20;
			}
			else if (format == 3) {
				return 20;
			}
			else if (format == 6) {
				return 22;
			}
			else if (format == 7) {
				return 22;
			}

			return 0;
		}

		int getOffsetRGB() {
			int format = header.pointDataFormat;

			if (format == 2) {
				return 20;
			}
			else if (format == 3) {
				return 28;
			}
			else if (format == 5) {
				return 28;
			}
			else if (format == 7) {
				return 30;
			}

			return 0;
		}

		void transform(uint8_t* source, uint8_t* dest, int size) {
			memcpy(dest, source, size);
		}

		vector<Attribute> getAttributes() {

			vector<Attribute> attributes;

			{ // XYZ
				Attribute a;

				a.name = "XYZ";
				a.bytes = 12;
				a.byteOffset = 0;
				a.elements = 3;
				a.elementSize = 4;

				attributes.push_back(a);
			}

			//{ // INTENSITY
			//	Attribute a;

			//	a.name = "intensity";
			//	a.bytes = 2;
			//	a.byteOffset = getOffsetIntensity();
			//	a.elements = 1;
			//	a.elementSize = 2;

			//	attributes.push_back(a);
			//}

			//{ // RETURN NUMBER
			//	Attribute a;

			//	a.bytes = 1;
			//	a.name = "returnNumber";
			//	a.byteOffset = getOffsetReturnNumber();
			//	a.elements = 1;
			//	a.elementSize = 1;

			//	attributes.push_back(a);
			//}

			//{ // CLASSIFICATION
			//	Attribute a;

			//	a.name = "classification";
			//	a.bytes = 1;
			//	a.byteOffset = getOffsetClassification();
			//	a.elements = 1;
			//	a.elementSize = 1;

			//	attributes.push_back(a);
			//}

			//// SCAN ANGLE RANK
			//if (getOffsetScanAngleRank() > 0) {
			//	Attribute a;

			//	a.name = "Scan Angle Rank";
			//	a.bytes = 1;
			//	a.byteOffset = getOffsetScanAngleRank();
			//	a.elements = 1;
			//	a.elementSize = 1;

			//	attributes.push_back(a);
			//}

			//// USER DATA
			//if (getOffsetUserData() > 0) {
			//	Attribute a;

			//	a.name = "User Data";
			//	a.bytes = 1;
			//	a.byteOffset = getOffsetUserData();
			//	a.elements = 1;
			//	a.elementSize = 1;

			//	attributes.push_back(a);
			//}

			//// POINT SOURCE ID
			//if (getOffsetPointSourceID() > 0) {
			//	Attribute a;

			//	a.name = "SourceID";
			//	a.bytes = 2;
			//	a.byteOffset = getOffsetPointSourceID();
			//	a.elements = 1;
			//	a.elementSize = 2;

			//	attributes.push_back(a);
			//}

			// RGB
			//if (getOffsetRGB() > 0) {
			//	Attribute a;
			//
			//	a.name = "RGB";
			//	a.bytes = 6;
			//	a.byteOffset = getOffsetRGB();
			//	a.elements = 3;
			//	a.elementSize = 2;
			//
			//	attributes.push_back(a);
			//}

			if (getOffsetRGB() > 0) {
				{
					Attribute a;

					a.name = "Red";
					a.bytes = 2;
					a.byteOffset = getOffsetRGB();
					a.elements = 1;
					a.elementSize = 2;

					attributes.push_back(a);
				}

				{
					Attribute a;

					a.name = "Green";
					a.bytes = 2;
					a.byteOffset = getOffsetRGB() + 2;
					a.elements = 1;
					a.elementSize = 2;

					attributes.push_back(a);
				}

				{
					Attribute a;

					a.name = "Blue";
					a.bytes = 2;
					a.byteOffset = getOffsetRGB() + 4;
					a.elements = 1;
					a.elementSize = 2;

					attributes.push_back(a);
				}
			}

			// GPS Time
			if (getOffsetGpsTime() > 0) {
				Attribute a;

				a.name = "GPS Time";
				a.bytes = 8;
				a.byteOffset = getOffsetGpsTime();
				a.elements = 1;
				a.elementSize = 8;

				attributes.push_back(a);
			}

			static vector<DataFormat> dataFormats = {
				DataFormat{20},
				DataFormat{28},
				DataFormat{26},
				DataFormat{34},
				DataFormat{57},
				DataFormat{63},
				DataFormat{30},
				DataFormat{36},
			};

			int currentOffset = dataFormats[header.pointDataFormat].size;
			auto extraBytes = getExtraBytes();

			for (auto& extraAttribute : extraBytes) {

				Attribute a;

				string name = string(&extraAttribute.name[0], &extraAttribute.name[31]);;
				name.erase(std::find(name.begin(), name.end(), '\0'), name.end());


				a.name = name;
				a.bytes = extraAttribute.bytes();
				a.byteOffset = currentOffset;

				// TODO: this is wrong:
				a.elements = 1;
				a.elementSize = a.bytes;

				attributes.push_back(a);

				currentOffset += a.bytes;
			}

			return attributes;
		}

		void createBinaryChunkParserThreadDynamicAttributeArray() {

			thread t([this]() {

				int i = 0;

				bool done = false;
				while (!done) {

					mtx_binary_chunks.lock();
					done = fullyLoaded() && binaryChunks.size() == 0;
					mtx_binary_chunks.unlock();

					if (done) {
						break;
					}

					mtx_binary_chunks.lock();
					if (binaryChunks.size() == 0) {

						mtx_binary_chunks.unlock();

						std::this_thread::sleep_for(std::chrono::milliseconds(5));

						continue;
					}

					auto binaryChunk = binaryChunks.front();
					binaryChunks.pop();
					mtx_binary_chunks.unlock();

					i++;

					{
						auto start = Utils::now();

						auto attributes = getAttributes();

						uint64_t n = binaryChunk->size / uint64_t(header.pointDataRecordLength);
						Points* points = new Points();
						points->size = n;
						points->xyzrgba.reserve(n);

						for (Attribute& attribute : attributes) {
							//attribute.data = reinterpret_cast<uint8_t*>(malloc(n * attribute.bytes));
							attribute.data = new BArray(n * attribute.bytes);
						}

						points->attributes = attributes;

						int positionOffset = 0;

						for (int i = 0; i < n; i++) {

							int byteOffset = i * header.pointDataRecordLength;

							uint8_t* data = binaryChunk->dataU8;

							for (Attribute& attribute : attributes) {
								transform(data + byteOffset + attribute.byteOffset, attribute.data->dataU8 + i * attribute.bytes, attribute.bytes);
							}

						}

						{ // specifically prepare xyz and color data for immediate use
							struct XYZI32 {
								int32_t x;
								int32_t y;
								int32_t z;
							};

							Attribute& aXYZ = attributes[0];
							XYZI32* xyz = reinterpret_cast<XYZI32*>(aXYZ.data->data);

							auto itRed = std::find_if(attributes.begin(), attributes.end(), [](Attribute& a) {
								return a.name == "Red";
							});
							auto itGreen= std::find_if(attributes.begin(), attributes.end(), [](Attribute& a) {
								return a.name == "Green";
							});
							auto itBlue = std::find_if(attributes.begin(), attributes.end(), [](Attribute& a) {
								return a.name == "Blue";
							});

							Attribute& aRed = *itRed;
							Attribute& aGreen = *itGreen;
							Attribute& aBlue = *itBlue;

							for (int i = 0; i < n; i++) {
								XYZRGBA point;
								XYZI32 pos = xyz[i];

								point.x = float(double(pos.x) * header.scaleX + header.offsetX - header.minX);
								point.y = float(double(pos.y) * header.scaleY + header.offsetY - header.minY);
								point.z = float(double(pos.z) * header.scaleZ + header.offsetZ - header.minZ);

								point.r = aRed.data->dataU16[i] <= 255 ? aRed.data->dataU16[i] : aRed.data->dataU16[i] / 256;
								point.g = aGreen.data->dataU16[i] <= 255 ? aGreen.data->dataU16[i] : aGreen.data->dataU16[i] / 256;
								point.b = aBlue.data->dataU16[i] <= 255 ? aBlue.data->dataU16[i] : aBlue.data->dataU16[i] / 256;

								point.a = 0;

								points->xyzrgba.emplace_back(point);
							}
						}



						mtc_access_chunk.lock();
						//cout << "chunk parsed by thread: " << std::this_thread::get_id() << ", numParsed: " << numParsed << endl;
						chunks.emplace_back(points);
						mtc_access_chunk.unlock();

						numParsed += n;

						delete binaryChunk;

						auto end = Utils::now();
						auto duration = end - start;
						//cout << "process duration: " << duration << "s" << endl;
					}

				}

				cout << "done parsing binary chunks" << endl;

			});
			t.detach();
		}


		void createBinaryLoaderThread() {

			thread t([this]() {
				double start = Utils::now();

				uint64_t offset = header.offsetToPointData;
				uint64_t pointsLoaded = 0;
				uint64_t bytes = header.numPoints * header.pointDataRecordLength;

#ifdef DISABLE_FILE_CACHE
				{ // disable windows file cache for benchmarking
					LPCTSTR lfile = file.c_str();

					auto hFile = CreateFile(lfile, GENERIC_READ,
						FILE_SHARE_READ,
						NULL, OPEN_EXISTING,
						FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN
						| FILE_FLAG_NO_BUFFERING, NULL);
				}
#endif

				FILE* in = fopen(file.c_str(), "rb");
				_fseeki64(in, offset, ios::beg);
				auto size = fs::file_size(file);

				bool done = false;
				while (!done) {

					uint32_t chunkSizePoints = (uint32_t)min(uint64_t(defaultChunkSize), header.numPoints - pointsLoaded);
					uint32_t chunkSizeBytes = chunkSizePoints * header.pointDataRecordLength;

					BArray* chunkBuffer = new BArray(chunkSizeBytes);
					auto bytesRead = fread(chunkBuffer->data, 1, chunkSizeBytes, in);

					done = bytesRead == 0;

					mtx_binary_chunks.lock();
					//binaryChunks.emplace_back(chunkBuffer);
					binaryChunks.emplace(chunkBuffer);
					mtx_binary_chunks.unlock();

					offset += chunkSizeBytes;
					pointsLoaded += chunkSizePoints;
					numLoaded = pointsLoaded;

					if (pointsLoaded >= header.numPoints) {
						break;
					}

				}

				cout << pointsLoaded << endl;

				double end = Utils::now();
				double duration = end - start;

				cout << "done loading binary chunks" << endl;
				cout << "duration: " << duration << endl;

			});

			t.detach();

		}



	};

}