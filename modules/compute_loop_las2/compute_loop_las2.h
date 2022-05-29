
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
#include "compute/LasLoaderSparse.h"

using namespace std;
using namespace std::chrono_literals;
using nlohmann::json;

struct ComputeLoopLas2 : public Method{

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
		int colorizeChunks;
		int colorizeOverdraw;
	};

	struct DebugData{
		uint32_t value = 0;
		bool enabled = false;
		uint32_t numPointsProcessed = 0;
		uint32_t numNodesProcessed = 0;
		uint32_t numPointsRendered = 0;
		uint32_t numNodesRendered = 0;
		uint32_t numPointsVisible = 0;
	};

	string source = "";
	Shader* csRender = nullptr;
	Shader* csResolve = nullptr;

	GLBuffer ssFramebuffer;
	GLBuffer ssDebug;
	GLBuffer ssBoundingBoxes;
	GLBuffer ssFiles;
	GLBuffer uniformBuffer;
	UniformData uniformData;
	shared_ptr<Buffer> ssFilesBuffer;

	shared_ptr<LasLoaderSparse> las = nullptr;

	Renderer* renderer = nullptr;

	ComputeLoopLas2(Renderer* renderer, shared_ptr<LasLoaderSparse> las){

		this->name = "loop_las_prefetch";
		this->description = R"ER01(
- Each thread renders X points.
- Loads points from LAS file
- While loading, each workgroup 
  - Computes the bounding box of assigned points.
  - creates 4, 8 and 12 byte vertex buffers.
- Workgroup picks 4, 8, or 12 byte precision
  depending on screen size of bounding box
		)ER01";
		this->las = las;
		this->group = "10-10-10 bit encoded";

		csRender = new Shader({ {"./modules/compute_loop_las2/render.cs", GL_COMPUTE_SHADER} });
		csResolve = new Shader({ {"./modules/compute_loop_las2/resolve.cs", GL_COMPUTE_SHADER} });

		ssFramebuffer = renderer->createBuffer(8 * 2048 * 2048);

		this->renderer = renderer;

		ssFilesBuffer = make_shared<Buffer>(10'000 * 128);

		ssDebug = renderer->createBuffer(256);
		ssBoundingBoxes = renderer->createBuffer(48 * 1'000'000);
		ssFiles = renderer->createBuffer(ssFilesBuffer->size);
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
			
		}

	}

	void render(Renderer* renderer) {

		GLTimerQueries::timestamp("compute-loop-start");

		las->process();

		if(las->numPointsLoaded == 0){
			return;
		}

		auto fbo = renderer->views[0].framebuffer;

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
			uniformData.colorizeChunks = Debug::colorizeChunks;
			uniformData.colorizeOverdraw = Debug::colorizeOverdraw;

			glNamedBufferSubData(uniformBuffer.handle, 0, sizeof(UniformData), &uniformData);
		}

		{ // update file buffer

			for(int i = 0; i < las->files.size(); i++){
				auto lasfile = las->files[i];

				dmat4 world = glm::translate(dmat4(), lasfile->boxMin);
				dmat4 view = renderer->views[0].view;
				dmat4 proj = renderer->views[0].proj;
				dmat4 worldView = view * world;
				dmat4 worldViewProj = proj * view * world;

				mat4 transform = worldViewProj;
				mat4 fWorld = world;

				memcpy(
					ssFilesBuffer->data_u8 + 256 * i + 0, 
					glm::value_ptr(transform), 
					64);

				if(Debug::updateFrustum){
					memcpy(
						ssFilesBuffer->data_u8 + 256 * i + 64, 
						glm::value_ptr(transform), 
						64);
				}

				memcpy(
					ssFilesBuffer->data_u8 + 256 * i + 128, 
					glm::value_ptr(fWorld), 
					64);

			}

			glNamedBufferSubData(ssFiles.handle, 0, 256 * las->files.size(), ssFilesBuffer->data);
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
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 41, las->ssXyzHig.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 42, las->ssXyzMed.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 43, las->ssXyzLow.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 44, las->ssColors.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 45, ssFiles.handle);

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
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 30, ssDebug.handle);
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