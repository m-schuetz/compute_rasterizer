
#pragma once

#include <string>
#include <queue>
#include <vector>
#include <mutex>
#include <thread>

#include "glm/common.hpp"
#include "glm/matrix.hpp"
#include <glm/gtx/transform.hpp>

#include "unsuck.hpp"

#include "Box.h"
#include "Debug.h"
#include "Camera.h"
#include "LasLoader.h"
#include "Frustum.h"
#include "Renderer.h"
#include "GLTimerQueries.h"

using namespace std;
using namespace std::chrono_literals;

struct ComputeBasic{

	string path = "";
	string source = "";
	Shader* csTiles = nullptr;
	Shader* csRender = nullptr;
	GLBuffer ssTileMetadata;
	GLBuffer ssTiles;
	GLBuffer ssDepth;
	GLBuffer ss_D_RGBA;


	ComputeBasic(Renderer* renderer){
		csTiles = new Shader({ {"./shaders/compute_tiles.cs", GL_COMPUTE_SHADER} });
		// csRender = new Shader({ {"./shaders/compute_basic.cs", GL_COMPUTE_SHADER} });
		csRender = new Shader({ {"./shaders/compute_tiles_draw.cs", GL_COMPUTE_SHADER} });

		ssTileMetadata = renderer->createBuffer(256);
		ssTiles = renderer->createBuffer(16 * 512 * 512);
		ssDepth = renderer->createBuffer(4 * 2048 * 2048);
		ss_D_RGBA = renderer->createBuffer(8 * 2048 * 2048);
	}


	void update(Renderer* renderer){

		auto camera = renderer->camera;
		auto viewProj = camera->proj * camera->view;
		Frustum frustum;
		frustum.set(viewProj);

	}

	void render(Renderer* renderer) {

		//this->camera = renderer->camera;

		// { // debug
		// 	Box box;
		// 	box.min = {-1.0, -1.0, -1.0};
		// 	box.max = { 1.0,  1.0,  1.0};
		// 	ivec3 color = { 255, 0, 0 };
		// 	renderer->drawBoundingBox(box.center(), box.size(), color);
		// }

		auto fbo = renderer->views[0].framebuffer;
		auto camera = renderer->camera;

		mat4 view = camera->view;
		mat4 proj = camera->proj;
		dmat4 viewProj = camera->proj * camera->view;

		int numTiles = 0;
		if(csTiles->program != -1)
		{ // COMPUTE TILES
			// parametric function (u, v) -> (x, y, z)
			// split into, e.g., 128x128 tiles
			// each thread computes batch bounds, and emmits tasksaccordingly
			// - bounds outside frustum		no work
			// - bounds smaller 2x2			draw directly
			// - bounds smaller 16x16		emmit 1 tile
			// - bounds smaller 32x32		emmit 2x2 tiles
			// - bounds smaller 64x64		emmit 4x4 tiles, and so on

			GLTimerQueries::timestamp("compute-tiles-start");

			glUseProgram(csTiles->program);

			glUniformMatrix4dv(0, 1, GL_FALSE, &viewProj[0][0]);
			glUniform2i(1, fbo->width, fbo->height);

			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssTileMetadata.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, ssTiles.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, ssDepth.handle);

			glBindImageTexture(0, fbo->colorAttachments[0]->handle, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA8UI);

			glDispatchCompute(32, 32, 1);

			GLTimerQueries::timestamp("compute-tiles-end");

			glMemoryBarrier(GL_ALL_BARRIER_BITS);

			auto buffer = renderer->readBuffer(ssTileMetadata, 0, 256);
			numTiles = buffer->get<uint32_t>(0);
			 //cout << formatNumber(numTiles) << endl;
		}

		if(csRender->program != -1)
		{ // DRAW TILES

			GLTimerQueries::timestamp("draw-tiles-start");

			glUseProgram(csRender->program);

			glUniformMatrix4dv(0, 1, GL_FALSE, &viewProj[0][0]);
			glUniform2i(1, fbo->width, fbo->height);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssTileMetadata.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, ssTiles.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, ssDepth.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, ss_D_RGBA.handle);

			glBindImageTexture(0, fbo->colorAttachments[0]->handle, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA8UI);

			glDispatchCompute(numTiles, 1, 1);

			GLTimerQueries::timestamp("draw-tiles-end");
		}

		// { // DRAW TILES
		// 	glUseProgram(csRender->program);

		// 	glUniformMatrix4dv(0, 1, GL_FALSE, &viewProj[0][0]);
		// 	glUniform2i(1, fbo->width, fbo->height);

		// 	glBindImageTexture(0, fbo->colorAttachments[0]->handle, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA8UI);

		// 	glDispatchCompute(32, 32, 1);
		// }

		{
			glMemoryBarrier(GL_ALL_BARRIER_BITS);

			GLuint zero = 0;
			glClearNamedBufferData(ssTileMetadata.handle, GL_R32UI, GL_RED, GL_UNSIGNED_INT, &zero);

			float inf = -Infinity;
			GLuint intbits;
			memcpy(&intbits, &inf, 4);
			glClearNamedBufferData(ssDepth.handle, GL_R32UI, GL_RED, GL_UNSIGNED_INT, &intbits);
			glClearNamedBufferData(ss_D_RGBA.handle, GL_R32UI, GL_RED, GL_UNSIGNED_INT, &intbits);

			glMemoryBarrier(GL_ALL_BARRIER_BITS);
		}
		

	}


};