
#pragma once

#include <string>
#include <queue>
#include <vector>
#include <mutex>
#include <thread>

#include "glm/common.hpp"
#include "glm/matrix.hpp"
#include <glm/gtx/transform.hpp>
#include "nlohmann/json.hpp"
#include <glm/gtc/matrix_transform.hpp> 

#include "unsuck.hpp"

#include "ProgressiveFileBuffer.h"
#include "Shader.h"
#include "Box.h"
#include "Debug.h"
#include "Camera.h"
#include "LasLoader.h"
#include "Frustum.h"
#include "Renderer.h"
#include "GLTimerQueries.h"
#include "Method.h"
#include "Runtime.h"
#include "compute/ComputeLasLoader.h"

using namespace std;
using namespace std::chrono_literals;
using nlohmann::json;

using glm::ivec2;

struct ComputeLoopLas : public Method{

	struct UniformData{
		mat4 world;
		mat4 view;
		mat4 proj;
		mat4 transform;
		mat4 transformFrustum;
		int pointsPerThread;
		int enableFrustumCulling;
		int showBoundingBox;
		int numPoints;
		ivec2 imageSize;
	};

	struct DebugData{
		uint32_t value = 0;
		bool enabled = false;
		uint32_t numPointsProcessed = 0;
		uint32_t numNodesProcessed = 0;
		uint32_t numPointsRendered = 0;
		uint32_t numNodesRendered = 0;
		uint32_t numPointsVisible = 0;
		uint32_t numLow = 0;
		uint32_t numMed = 0;
		uint32_t numHig = 0;
	};

	string source = "";
	Shader* csRender = nullptr;
	Shader* csResolve = nullptr;

	GLBuffer ssFramebuffer;
	GLBuffer ssDebug;
	GLBuffer ssBoundingBoxes;
	GLBuffer uniformBuffer;
	UniformData uniformData;

	shared_ptr<ComputeLasData> las = nullptr;

	Renderer* renderer = nullptr;

	ComputeLoopLas(Renderer* renderer, shared_ptr<ComputeLasData> las){

		this->name = "loop_las";
		this->description = R"ER01(
- Each thread renders X points.
- Loads points from LAS file
- encodes point coordinates in 10+10+10 bits
- Workgroup picks 10, 20 or 30 bit precision
  depending on screen size of bounding box
		)ER01";
		this->las = las;
		this->group = "10-10-10 bit encoded";

		csRender = new Shader({ {"./modules/compute_loop_las/render.cs", GL_COMPUTE_SHADER} });
		csResolve = new Shader({ {"./modules/compute_loop_las/resolve.cs", GL_COMPUTE_SHADER} });

		this->renderer = renderer;

		ssFramebuffer = renderer->createBuffer(8 * 2048 * 2048);
		ssDebug = renderer->createBuffer(256);
		ssBoundingBoxes = renderer->createBuffer(48 * 1'000'000);
		uniformBuffer = renderer->createUniformBuffer(512);

		GLuint zero = 0;
		glClearNamedBufferData(ssDebug.handle, GL_R32UI, GL_RED, GL_UNSIGNED_INT, &zero);
		glClearNamedBufferData(ssBoundingBoxes.handle, GL_R32UI, GL_RED, GL_UNSIGNED_INT, &zero);
	}
	
	void update(Renderer* renderer){

		if(Runtime::resource != (Resource*)las.get()){

			if(Runtime::resource != nullptr){
				Runtime::resource->unload(renderer);
			}

			las->load(renderer);

			Runtime::resource = (Resource*)las.get();
		}

	}

	void render(Renderer* renderer) {

		GLTimerQueries::timestamp("compute-loop-start");

		las->process(renderer);

		if(las->numPointsLoaded == 0){
			return;
		}

		auto fbo = renderer->views[0].framebuffer;

		if(Runtime::requestReadBatches){

			int numBatches = ceil(double(las->numPointsLoaded) / double(POINTS_PER_WORKGROUP));
			auto buffer = renderer->readBuffer(las->ssBatches, 0, 64 * numBatches);

			vec3 min = vec3(Infinity, Infinity, Infinity);
			vec3 max = vec3(-Infinity, -Infinity, -Infinity);

			vector<float> diagonals;

			float datasetDiagonal = glm::length(las->boxMax - las->boxMin);

			for(int i = 0; i < numBatches; i++){
				vec3 batchMin = {
					buffer->get<float>(64 * i +  4),
					buffer->get<float>(64 * i +  8),
					buffer->get<float>(64 * i + 12)
				};

				vec3 batchMax = {
					buffer->get<float>(64 * i + 16),
					buffer->get<float>(64 * i + 20),
					buffer->get<float>(64 * i + 24)
				};

				float diagonal = glm::length(batchMax - batchMin);

				diagonals.push_back(diagonal);

				min.x = std::min(min.x, batchMin.x);
				min.y = std::min(min.y, batchMin.y);
				min.z = std::min(min.z, batchMin.z);
				max.x = std::max(max.x, batchMax.x);
				max.y = std::max(max.y, batchMax.y);
				max.z = std::max(max.z, batchMax.z);
			}

			//float numBins = 128.0;
			//vector<int> histogram(int(numBins), 0.0);

			//for(int i = 0; i < numBatches; i++){
			//	vec3 batchMin = {
			//		buffer->get<float>(64 * i +  4),
			//		buffer->get<float>(64 * i +  8),
			//		buffer->get<float>(64 * i + 12)
			//	};

			//	vec3 batchMax = {
			//		buffer->get<float>(64 * i + 16),
			//		buffer->get<float>(64 * i + 20),
			//		buffer->get<float>(64 * i + 24)
			//	};

			//	float diagonal = glm::length(batchMax - batchMin);
			//	float u = diagonal / datasetDiagonal;

			//	int bin = std::min(u * numBins, numBins - 1.0f);

			//	histogram[bin]++;
			//}

			std::sort(diagonals.begin(), diagonals.end());

			float minDiagonal = diagonals[0];
			float lower_quartile = diagonals[int(float(diagonals.size()) * 0.25)];
			float median = diagonals[int(float(diagonals.size()) * 0.50)];
			float upper_quartile = diagonals[int(float(diagonals.size()) * 0.75)];
			float maxDiagonal = diagonals[diagonals.size() - 1];

			stringstream ss;
			ss << "batches: " << numBatches << endl;
			ss << "min: " << min.x << ", " << min.y << ", " << min.z << endl;
			ss << "max: " << max.x << ", " << max.y << ", " << max.z << endl;
			//ss << "histogram bounds: " << minDiagonal << " - " << maxDiagonal << endl;
			ss << "maxDiagonal: " << maxDiagonal << endl;
			ss << "min-lower-median-upper-max: " << minDiagonal << ", " << lower_quartile << ", " << median << ", " << upper_quartile << ", " << maxDiagonal << endl;

			//ss << "histogram: ";
			//for(float i = 0.0; i < numBins; i += 1.0){
			//	ss << histogram[int(i)] << "\t";
			//}
			ss << endl;

			writeFile("./misc.txt", ss.str());

			Runtime::requestReadBatches = false;
		}

		// Update Uniform Buffer
		{
			mat4 world;
			mat4 view = renderer->views[0].view;
			mat4 proj = renderer->views[0].proj;
			mat4 worldView = view * world;
			mat4 worldViewProj = proj * view * world;

			uniformData.world = world;
			uniformData.view = view;
			uniformData.proj = proj;
			uniformData.transform = worldViewProj;
			if(Debug::updateFrustum){
				uniformData.transformFrustum = worldViewProj;
			}
			uniformData.pointsPerThread = POINTS_PER_THREAD;
			uniformData.numPoints = las->numPointsLoaded;
			uniformData.enableFrustumCulling = Debug::frustumCullingEnabled ? 1 : 0;
			uniformData.showBoundingBox = Debug::showBoundingBox ? 1 : 0;
			uniformData.imageSize = {fbo->width, fbo->height};

			glNamedBufferSubData(uniformBuffer.handle, 0, sizeof(UniformData), &uniformData);
		}

		if(Debug::enableShaderDebugValue){
			DebugData data;
			data.enabled = true;

			glNamedBufferSubData(ssDebug.handle, 0, sizeof(DebugData), &data);
		}
		
		// RENDER
		if(csRender->program != -1){
			GLTimerQueries::timestamp("draw-start");

			glUseProgram(csRender->program);

			auto& viewLeft = renderer->views[0];

			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssFramebuffer.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 30, ssDebug.handle);
			glBindBufferBase(GL_UNIFORM_BUFFER, 31, uniformBuffer.handle);

			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 40, las->ssBatches.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 41, las->ssXyz_12b.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 42, las->ssXyz_8b.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 43, las->ssXyz_4b.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 44, las->ssColors.handle);

			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 50, ssBoundingBoxes.handle);

			glBindImageTexture(0, fbo->colorAttachments[0]->handle, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA8UI);

			int numBatches = ceil(double(las->numPointsLoaded) / double(POINTS_PER_WORKGROUP));
			
			glDispatchCompute(numBatches, 1, 1);

			GLTimerQueries::timestamp("draw-end");
		}

		// RESOLVE
		if(csResolve->program != -1){
			GLTimerQueries::timestamp("resolve-start");

			glUseProgram(csResolve->program);
			
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssFramebuffer.handle);
			glBindBufferBase(GL_UNIFORM_BUFFER, 31, uniformBuffer.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 44, las->ssColors.handle);

			glBindImageTexture(0, fbo->colorAttachments[0]->handle, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA8UI);

			int groups_x = ceil(float(fbo->width) / 16.0f);
			int groups_y = ceil(float(fbo->height) / 16.0f);
			glDispatchCompute(groups_x, groups_y, 1);

			GLTimerQueries::timestamp("resolve-end");
		}

		// READ DEBUG VALUES
		if(Debug::enableShaderDebugValue){
			glMemoryBarrier(GL_ALL_BARRIER_BITS);

			DebugData data;
			glGetNamedBufferSubData(ssDebug.handle, 0, sizeof(DebugData), &data);

			auto dbg = Debug::getInstance();

			dbg->pushFrameStat("#nodes processed"        , formatNumber(data.numNodesProcessed));
			dbg->pushFrameStat("#nodes rendered"         , formatNumber(data.numNodesRendered));
			dbg->pushFrameStat("#points processed"       , formatNumber(data.numPointsProcessed));
			dbg->pushFrameStat("#points rendered"        , formatNumber(data.numPointsRendered));
			dbg->pushFrameStat("divider" , "");
			dbg->pushFrameStat("#points visible"         , formatNumber(data.numPointsVisible));
			dbg->pushFrameStat("#low"                    , formatNumber(data.numLow));
			dbg->pushFrameStat("#med"                    , formatNumber(data.numMed));
			dbg->pushFrameStat("#hig"                    , formatNumber(data.numHig));

			glMemoryBarrier(GL_ALL_BARRIER_BITS);
		}

		// BOUNDING BOXES
		if(Debug::showBoundingBox){
			glMemoryBarrier(GL_ALL_BARRIER_BITS);
			glBindFramebuffer(GL_FRAMEBUFFER, fbo->handle);

			auto camera = renderer->camera;
			renderer->drawBoundingBoxes(camera.get(), ssBoundingBoxes);
		}

		{ // CLEAR
			glMemoryBarrier(GL_ALL_BARRIER_BITS);

			GLuint zero = 0;
			float inf = -Infinity;
			GLuint intbits;
			memcpy(&intbits, &inf, 4);

			glClearNamedBufferSubData(ssFramebuffer.handle, GL_R32UI, 0, fbo->width * fbo->height * 8, GL_RED, GL_UNSIGNED_INT, &intbits);
			glClearNamedBufferData(ssDebug.handle, GL_R32UI, GL_RED, GL_UNSIGNED_INT, &zero);
			glClearNamedBufferSubData(ssBoundingBoxes.handle, GL_R32UI, 0, 48, GL_RED, GL_UNSIGNED_INT, &zero);

			glMemoryBarrier(GL_ALL_BARRIER_BITS);
		}
		
		GLTimerQueries::timestamp("compute-loop-end");
	}


};