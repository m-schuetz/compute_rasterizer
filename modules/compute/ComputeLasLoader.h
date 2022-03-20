
#pragma once

#include <string>

#include "glm/common.hpp"
#include "glm/matrix.hpp"
#include "glm/vec3.hpp"
#include <glm/gtx/transform.hpp>
#include "unsuck.hpp"
#include "Shader.h"
#include "Resources.h"

using namespace std;
using glm::vec3;

struct ComputeLasData : public Resource {

	struct LoaderTask{
		shared_ptr<Buffer> buffer = nullptr;
		int64_t pointOffset = 0;
		int64_t numPoints = 0;
	};

	string path = "";
	shared_ptr<LoaderTask> task = nullptr;

	// LAS header data
	bool headerLoaded = false;
	int64_t numPoints = 0;
	int64_t numPointsLoaded = 0;
	uint32_t offsetToPointData = 0;
	int pointFormat = 0;
	uint32_t bytesPerPoint = 0;
	dvec3 scale = {1.0, 1.0, 1.0};
	dvec3 offset = {0.0, 0.0, 0.0};
	dvec3 boxMin;
	dvec3 boxMax;

	// GL Buffers
	GLBuffer ssBatches;
	GLBuffer ssXyz_12b;
	GLBuffer ssXyz_8b;
	GLBuffer ssXyz_4b;
	GLBuffer ssColors;
	GLBuffer ssLoadBuffer;
	//GLBuffer ssLOD;
	//GLBuffer ssLODColor;

	ComputeLasData(){

	}

	void loadHeader(){

		ComputeLasData* data = this;

		auto buffer_header = readBinaryFile(data->path, 0, 375);

		int versionMajor = buffer_header->get<uint8_t>(24);
		int versionMinor = buffer_header->get<uint8_t>(25);

		if(versionMajor == 1 && versionMinor < 4){
			data->numPoints = buffer_header->get<uint32_t>(107);
		}else{
			data->numPoints = buffer_header->get<uint64_t>(247);
		}

		//numPoints = min(numPoints, 50'000'000ll);
		numPoints = min(numPoints, 1'000'000'000ll);
		// 1'073'741'824

		data->offsetToPointData = buffer_header->get<uint32_t>(96);
		data->pointFormat = buffer_header->get<uint8_t>(104);
		data->bytesPerPoint = buffer_header->get<uint16_t>(105);

		data->scale.x = buffer_header->get<double>(131);
		data->scale.y = buffer_header->get<double>(139);
		data->scale.z = buffer_header->get<double>(147);

		data->offset.x = buffer_header->get<double>(155);
		data->offset.y = buffer_header->get<double>(163);
		data->offset.z = buffer_header->get<double>(171);

		data->boxMin.x = buffer_header->get<double>(187);
		data->boxMin.y = buffer_header->get<double>(203);
		data->boxMin.z = buffer_header->get<double>(219);

		data->boxMax.x = buffer_header->get<double>(179);
		data->boxMax.y = buffer_header->get<double>(195);
		data->boxMax.z = buffer_header->get<double>(211);

		data->headerLoaded = true;
	}

	static shared_ptr<ComputeLasData> create(string path){
		auto data = make_shared<ComputeLasData>();
		data->path = path;
		data->loadHeader();

		return data;
	}

	void load(Renderer* renderer);
	void unload(Renderer* renderer);
	void process(Renderer* renderer);

};