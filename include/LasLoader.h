
#pragma once

#include <string>
#include <unordered_map>
#include <map>
#include <thread>
#include <functional>

#include "glm/common.hpp"
#include "glm/matrix.hpp"
#include <glm/gtx/transform.hpp>

#include "unsuck.hpp"

using glm::dvec3;
using glm::ivec3;

using namespace std;

struct LasPoints{
	shared_ptr<Buffer> buffer;
	int64_t numPoints;
};


//mutex mtx_wat;

struct LasLoader{

	LasLoader(){
	
	}

	static LasPoints loadSync(string file, int64_t firstPoint, int64_t wantedPoints){

		//lock_guard<mutex> lock(mtx_wat);

		auto headerBuffer = new Buffer(2 * 1024);
		readBinaryFile(file, 0, headerBuffer ->size, headerBuffer ->data);

		uint64_t offsetToPointData = headerBuffer ->get<uint32_t>(96);
		int format = headerBuffer ->get<uint8_t>(104);
		int recordLength = headerBuffer ->get<uint16_t>(105);

		int versionMajor = headerBuffer ->get<uint8_t>(24);
		int versionMinor = headerBuffer ->get<uint8_t>(25);

		int64_t numPoints = 0;
		if(versionMajor == 1 && versionMinor <= 3){
			numPoints = headerBuffer ->get<uint32_t>(107);
		}else{
			numPoints = headerBuffer ->get<int64_t>(247);
		}

		dvec3 c_scale = {
			headerBuffer->get<double>(131),
			headerBuffer->get<double>(139),
			headerBuffer->get<double>(147)
		};

		dvec3 c_offset = {
			headerBuffer->get<double>(155),
			headerBuffer->get<double>(163),
			headerBuffer->get<double>(171)
		};

		int64_t batchSize_points = std::min(numPoints - firstPoint, wantedPoints);

		int64_t byteOffset = offsetToPointData + recordLength * firstPoint;
		int64_t byteSize = batchSize_points * recordLength;

		auto rawBuffer = make_shared<Buffer>(byteSize);
		readBinaryFile(file, byteOffset, byteSize, rawBuffer->data);

		// transform to XYZRGBA
		auto targetBuffer = make_shared<Buffer>(32 * batchSize_points);

		for(int i = 0; i < batchSize_points; i++){
			int64_t offset = i * recordLength;

			int32_t X = rawBuffer->get<int32_t>(offset + 0);
			int32_t Y = rawBuffer->get<int32_t>(offset + 4);
			int32_t Z = rawBuffer->get<int32_t>(offset + 8);
			uint16_t R = rawBuffer->get<uint16_t>(offset + 28);
			uint16_t G = rawBuffer->get<uint16_t>(offset + 30);
			uint16_t B = rawBuffer->get<uint16_t>(offset + 32);

			double x = double(X) * c_scale.x + c_offset.x;
			double y = double(Y) * c_scale.y + c_offset.y;
			double z = double(Z) * c_scale.z + c_offset.z;

			R = R > 255 ? R / 255 : R;
			G = G > 255 ? G / 255 : G;
			B = B > 255 ? B / 255 : B;
			uint32_t color = (R << 0) | (G << 8) | (B << 16);

			targetBuffer->set<double>(x, 32 * i + 0);
			targetBuffer->set<double>(y, 32 * i + 8);
			targetBuffer->set<double>(z, 32 * i + 16);
			targetBuffer->set<uint32_t>(color, 32 * i + 24);
		}

		LasPoints laspoints;
		laspoints.buffer = targetBuffer;
		laspoints.numPoints = batchSize_points;

		return laspoints;
	}

	static void load(string file, int64_t firstPoint, int64_t wantedPoints, function<void(shared_ptr<Buffer>, int64_t numLoaded)> callback){

		thread t([=](){
			auto laspoints = LasLoader::loadSync(file, firstPoint, wantedPoints);

			callback(laspoints.buffer, laspoints.numPoints);
		});

		t.detach();
	}

}; 
