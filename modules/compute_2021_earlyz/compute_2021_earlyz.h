
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
#include "compute/LasLoaderStandard.h"

using namespace std;
using namespace std::chrono_literals;
using nlohmann::json;



struct ComputeEarlyZ : public Method{

	string source = "";
	Shader* csRender = nullptr;
	Shader* csResolve = nullptr;

	GLBuffer ssFramebuffer;
	GLBuffer ssDebug;

	shared_ptr<LasStandardData> las = nullptr;

	Renderer* renderer = nullptr;

	ComputeEarlyZ(Renderer* renderer, shared_ptr<LasStandardData> las){

		this->name = "(2021) early-z";
		this->description = R"ER01(
		)ER01";
		this->las = las;
		this->group = "2021 method; standard 16 byte per point";

		csRender = new Shader({ {"./modules/compute_2021_earlyz/render.cs", GL_COMPUTE_SHADER} });
		csResolve = new Shader({ {"./modules/compute_2021_earlyz/resolve.cs", GL_COMPUTE_SHADER} });

		this->renderer = renderer;

		ssFramebuffer = renderer->createBuffer(8 * 2048 * 2048);
		ssDebug = renderer->createBuffer(256);

		GLuint zero = 0;
		glClearNamedBufferData(ssDebug.handle, GL_R32UI, GL_RED, GL_UNSIGNED_INT, &zero);
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

		GLTimerQueries::timestamp("compute-earlyz-start");

		las->process(renderer);

		auto fbo = renderer->views[0].framebuffer;
		auto camera = renderer->camera;

		if(las->numPointsLoaded == 0){
			return;
		}

		// RENDER
		if(csRender->program != -1)
		{ 

			GLTimerQueries::timestamp("draw-start");

			glUseProgram(csRender->program);

			auto& viewLeft = renderer->views[0];

			mat4 world;
			mat4 view = viewLeft.view;
			mat4 proj = viewLeft.proj;
			mat4 worldViewProj = proj * view * world;

			glUniformMatrix4fv(0, 1, GL_FALSE, &worldViewProj[0][0]);
			glUniform2i(1, fbo->width, fbo->height);
			
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssFramebuffer.handle);

			int64_t pointsRemaining = las->numPointsLoaded;
			int64_t pointsRendered = 0;
			int64_t pointsPerBatch = 268'000'000;
			while(pointsRemaining > 0){
				
				int pointsInBatch = min(pointsRemaining, pointsPerBatch);

				int64_t start = 16ul * pointsRendered;
				int64_t size = 16ul * pointsInBatch;
				glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 5, las->ssPoints.handle, start, size);

				int numBatches = ceil(double(pointsInBatch) / 128.0);
				
				glDispatchCompute(numBatches, 1, 1);

				pointsRendered += pointsInBatch;
				pointsRemaining -= pointsInBatch;
			}

			GLTimerQueries::timestamp("draw-end");
		}

		// RESOLVE
		if(csResolve->program != -1)
		{ 

			GLTimerQueries::timestamp("resolve-start");

			glUseProgram(csResolve->program);
			
			
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssFramebuffer.handle);

			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, las->ssPoints.handle);

			// constexpr int64_t BYTES_BATCH = 16 * 100'000'000;
			// glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 5, las->ssPoints.handle, 
			// 	0, min(BYTES_BATCH, 16 * las->numPointsLoaded));

			// glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 6, las->ssPoints.handle, 
			// 	BYTES_BATCH, min(16 * las->numPointsLoaded, 2 * BYTES_BATCH));

			glBindImageTexture(0, fbo->colorAttachments[0]->handle, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA8UI);

			int groups_x = fbo->width / 16;
			int groups_y = fbo->height / 16;
			glDispatchCompute(groups_x, groups_y, 1);
			

			glMemoryBarrier(GL_ALL_BARRIER_BITS);

			GLTimerQueries::timestamp("resolve-end");
		}


		// { // CLEAR
		// 	glMemoryBarrier(GL_ALL_BARRIER_BITS);

		// 	GLuint zero = 0;
		// 	float inf = -Infinity;
		// 	GLuint intbits;
		// 	memcpy(&intbits, &inf, 4);

		// 	glClearNamedBufferData(ssFramebuffer.handle, GL_R32UI, GL_RED, GL_UNSIGNED_INT, &intbits);

		// 	glMemoryBarrier(GL_ALL_BARRIER_BITS);
		// }
		
		GLTimerQueries::timestamp("compute-earlyz-end");
	}


};