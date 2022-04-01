
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


struct LasStandardData : public Resource {

	struct LoaderTask{
		shared_ptr<Buffer> buffer = nullptr;
		int64_t pointOffset = 0;
		int64_t numPoints = 0;
	};

	string path = "";
	shared_ptr<LoaderTask> task = nullptr;
	mutex mtx_state;

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

	GLBuffer ssPoints;

	LasStandardData(){

	}

	void loadHeader(){

		LasStandardData* data = this;

		auto buffer_header = readBinaryFile(data->path, 0, 375);

		int versionMajor = buffer_header->get<uint8_t>(24);
		int versionMinor = buffer_header->get<uint8_t>(25);

		if(versionMajor == 1 && versionMinor < 4){
			data->numPoints = buffer_header->get<uint32_t>(107);
		}else{
			data->numPoints = buffer_header->get<uint64_t>(247);
		}

		data->numPoints = min(data->numPoints, 1'000'000'000ll);

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

	static shared_ptr<LasStandardData> create(string path){
		auto data = make_shared<LasStandardData>();
		data->path = path;
		data->loadHeader();

		return data;
	}

	void load(Renderer* renderer){

		cout << "LasStandardData::load()" << endl;

		{
			lock_guard<mutex> lock(mtx_state);

			if(state != ResourceState::UNLOADED){
				return;
			}else{
				state = ResourceState::LOADING;
			}
		}

		this->ssPoints = renderer->createBuffer(16 * this->numPoints);
		GLuint zero = 0;
		glClearNamedBufferData(this->ssPoints.handle, GL_R32UI, GL_RED, GL_UNSIGNED_INT, &zero);

		LasStandardData *ref = this;
		thread t([ref](){

			int pointsRemaining = ref->numPoints;
			int pointsRead = 0;
			while(pointsRemaining > 0){

				{ // abort loader thread if state is set to unloading
					lock_guard<mutex> lock(ref->mtx_state);

					if(ref->state == ResourceState::UNLOADING){
						cout << "stopping loader thread for " << ref->path << endl;

						ref->state = ResourceState::UNLOADED;

						return;
					}
				}

				if(ref->task){
					this_thread::sleep_for(1ms);
					continue;
				}

				constexpr int POINTS_PER_BATCH = 1'000'000;
				int pointsInBatch = std::min(pointsRemaining, POINTS_PER_BATCH);
				int64_t start = int64_t(ref->offsetToPointData) + int64_t(ref->bytesPerPoint) * int64_t(pointsRead);
				int64_t size = ref->bytesPerPoint * pointsInBatch;
				auto source = readBinaryFile(ref->path, start, size);
				auto target = make_shared<Buffer>(16 * pointsInBatch);

				int rgbOffset = 0;
				if(ref->pointFormat == 2){
					rgbOffset = 20;
				}else if(ref->pointFormat == 3){
					rgbOffset = 28;
				}

				for(int i = 0; i < pointsInBatch; i++){
					int32_t pointOffset = i * ref->bytesPerPoint;

					int32_t X = source->get<int32_t>(pointOffset + 0);
					int32_t Y = source->get<int32_t>(pointOffset + 4);
					int32_t Z = source->get<int32_t>(pointOffset + 8);

					float x = double(X) * ref->scale.x + ref->offset.x - ref->boxMin.x;
					float y = double(Y) * ref->scale.y + ref->offset.y - ref->boxMin.y;
					float z = double(Z) * ref->scale.z + ref->offset.z - ref->boxMin.z;

					int32_t R = source->get<uint16_t>(pointOffset + rgbOffset + 0);
					int32_t G = source->get<uint16_t>(pointOffset + rgbOffset + 2);
					int32_t B = source->get<uint16_t>(pointOffset + rgbOffset + 4);

					uint8_t r = R > 255 ? R / 256 : R;
					uint8_t g = G > 255 ? G / 256 : G;
					uint8_t b = B > 255 ? B / 256 : B;

					target->set<float>(x, 16 * i + 0);
					target->set<float>(y, 16 * i + 4);
					target->set<float>(z, 16 * i + 8);
					target->set<uint8_t>(r, 16 * i + 12);
					target->set<uint8_t>(g, 16 * i + 13);
					target->set<uint8_t>(b, 16 * i + 14);
					target->set<uint8_t>(255, 16 * i + 15);

				}

				// TODO: parse points, transform to XYZRGBA layout

				auto task = make_shared<LoaderTask>();
				task->buffer = target;
				task->pointOffset = pointsRead;
				task->numPoints = pointsInBatch;

				ref->task = task;

				pointsRemaining -= pointsInBatch;
				pointsRead += pointsInBatch;

				Debug::set("numPointsLoaded", formatNumber(pointsRead));

				if((pointsRead % 10'000'000) == 0){
					cout << "loaded " << formatNumber(pointsRead) << endl;
				}

			}

			cout << "finished loading " << formatNumber(pointsRead) << " points" << endl;

			{ // check if resource was marked as unloading in the meantime
				lock_guard<mutex> lock(ref->mtx_state);

				if(ref->state == ResourceState::UNLOADING){
					cout << "stopping loader thread for " << ref->path << endl;

					ref->state = ResourceState::UNLOADED;
				}else if(ref->state == ResourceState::LOADING){
					ref->state = ResourceState::LOADED;
				}
			}

		});
		t.detach();
		
	}

	void unload(Renderer* renderer){

		cout << "LasStandardData::unload()" << endl;

		numPointsLoaded = 0;

		glDeleteBuffers(1, &ssPoints.handle);

		lock_guard<mutex> lock(mtx_state);

		if(state == ResourceState::LOADED){
			state = ResourceState::UNLOADED;
		}else if(state == ResourceState::LOADING){
			// if loader thread is still running, notify thread by marking resource as "unloading"
			state = ResourceState::UNLOADING;
		}
	}

	void process(Renderer* renderer){

		if(this->task){
			glNamedBufferSubData(this->ssPoints.handle, 16 * numPointsLoaded, this->task->buffer->size, this->task->buffer->data);

			this->numPointsLoaded += this->task->numPoints;
			this->task = nullptr;

		}

	}

};