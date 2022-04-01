
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



struct Compute2021HQS : public Method{

	string source = "";
	Shader* csDepth = nullptr;
	Shader* csColor = nullptr;
	Shader* csResolve = nullptr;

	GLBuffer ssDepth;
	GLBuffer ssColor;

	shared_ptr<LasStandardData> las = nullptr;

	Renderer* renderer = nullptr;

	Compute2021HQS(Renderer* renderer, shared_ptr<LasStandardData> las){

		this->name = "(2021) hqs";
		this->description = R"ER01(
		)ER01";
		this->las = las;
		this->group = "2021 method; standard 16 byte per point";

		csDepth = new Shader({ {"./modules/compute_2021_hqs/render_depth.cs", GL_COMPUTE_SHADER} });
		csColor = new Shader({ {"./modules/compute_2021_hqs/render_attribute.cs", GL_COMPUTE_SHADER} });
		csResolve = new Shader({ {"./modules/compute_2021_hqs/resolve.cs", GL_COMPUTE_SHADER} });

		this->renderer = renderer;

		ssDepth = renderer->createBuffer(8 * 2048 * 2048);
		ssColor = renderer->createBuffer(16 * 2048 * 2048);

		GLuint zero = 0;
		glClearNamedBufferData(ssDepth.handle, GL_R32UI, GL_RED, GL_UNSIGNED_INT, &zero);
		glClearNamedBufferData(ssColor.handle, GL_R32UI, GL_RED, GL_UNSIGNED_INT, &zero);
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

		GLTimerQueries::timestamp("compute-hqs-start");

		las->process(renderer);

		auto fbo = renderer->views[0].framebuffer;
		auto camera = renderer->camera;

		if(las->numPointsLoaded == 0){
			return;
		}

		// RENDER DEPTH
		if(csDepth->program != -1){ 

			GLTimerQueries::timestamp("depth-start");

			glUseProgram(csDepth->program);

			auto& viewLeft = renderer->views[0];

			mat4 world;
			mat4 view = viewLeft.view;
			mat4 proj = viewLeft.proj;
			mat4 worldViewProj = proj * view * world;

			glUniformMatrix4fv(0, 1, GL_FALSE, &worldViewProj[0][0]);
			glUniform2i(1, fbo->width, fbo->height);
			
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, ssDepth.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, ssColor.handle);

			int64_t pointsRemaining = las->numPointsLoaded;
			int64_t pointsRendered = 0;
			int64_t pointsPerBatch = 268'000'000;
			while(pointsRemaining > 0){
				
				int pointsInBatch = min(pointsRemaining, pointsPerBatch);
				
				int64_t start = 16ul * pointsRendered;
				int64_t size = 16ul * pointsInBatch;
				glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 0, las->ssPoints.handle, start, size);

				int numBatches = ceil(double(pointsInBatch) / 128.0);
				
				glDispatchCompute(numBatches, 1, 1);

				pointsRendered += pointsInBatch;
				pointsRemaining -= pointsInBatch;
			}

			GLTimerQueries::timestamp("depth-end");
		}

		// RENDER COLOR
		if(csColor->program != -1)
		{ 

			GLTimerQueries::timestamp("color-start");

			glUseProgram(csColor->program);

			auto& viewLeft = renderer->views[0];

			mat4 world;
			mat4 view = viewLeft.view;
			mat4 proj = viewLeft.proj;
			mat4 worldViewProj = proj * view * world;

			glUniformMatrix4fv(0, 1, GL_FALSE, &worldViewProj[0][0]);
			glUniform2i(1, fbo->width, fbo->height);
			
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, ssDepth.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, ssColor.handle);

			int64_t pointsRemaining = las->numPointsLoaded;
			int64_t pointsRendered = 0;
			int64_t pointsPerBatch = 268'000'000;
			while(pointsRemaining > 0){
				
				int pointsInBatch = min(pointsRemaining, pointsPerBatch);
				
				int64_t start = 16ul * pointsRendered;
				int64_t size = 16ul * pointsInBatch;
				glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 0, las->ssPoints.handle, start, size);

				int numBatches = ceil(double(pointsInBatch) / 128.0);
				
				glDispatchCompute(numBatches, 1, 1);

				pointsRendered += pointsInBatch;
				pointsRemaining -= pointsInBatch;
			}

			GLTimerQueries::timestamp("color-end");
		}

		// RESOLVE
		if(csResolve->program != -1)
		{ 

			GLTimerQueries::timestamp("resolve-start");

			glUseProgram(csResolve->program);
			
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, las->ssPoints.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, ssDepth.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, ssColor.handle);

			glBindImageTexture(0, fbo->colorAttachments[0]->handle, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA8UI);

			int groups_x = fbo->width / 16;
			int groups_y = fbo->height / 16;
			glDispatchCompute(groups_x, groups_y, 1);
			

			glMemoryBarrier(GL_ALL_BARRIER_BITS);

			GLTimerQueries::timestamp("resolve-end");
		}

		
		GLTimerQueries::timestamp("compute-hqs-end");
	}


};