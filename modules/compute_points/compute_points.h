
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

struct ComputePoints{

	string path = "";
	string source = "";
	Shader* csRender = nullptr;
	Shader* csResolve = nullptr;
	GLBuffer ssDepth;
	GLBuffer ss_D_RGBA;


	ComputePoints(Renderer* renderer){
		csRender = new Shader({ {"./modules/compute_points/compute_points_draw.cs", GL_COMPUTE_SHADER} });
		csResolve = new Shader({ {"./modules/compute_points/compute_points_resolve.cs", GL_COMPUTE_SHADER} });

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

		GLTimerQueries::timestamp("compute-points-start");

		//this->camera = renderer->camera;

		 //{ // debug
		 //	Box box;
		 //	box.min = {-1.0, -1.0, -1.0};
		 //	box.max = { 1.0,  1.0,  1.0};
		 //	ivec3 color = { 255, 0, 0 };
		 //	renderer->drawBoundingBox(box.center(), box.size(), color);
		 //}

		auto fbo = renderer->views[0].framebuffer;
		auto camera = renderer->camera;

		mat4 view = camera->view;
		mat4 proj = camera->proj;
		dmat4 viewProj = camera->proj * camera->view;


		// RENDER
		if(csRender->program != -1)
		{ 

			GLTimerQueries::timestamp("draw-start");

			glUseProgram(csRender->program);

			glUniformMatrix4dv(0, 1, GL_FALSE, &viewProj[0][0]);
			glUniform2i(1, fbo->width, fbo->height);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, ssDepth.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, ss_D_RGBA.handle);

			glBindImageTexture(0, fbo->colorAttachments[0]->handle, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA8UI);

			glDispatchCompute(5, 1, 1);

			GLTimerQueries::timestamp("draw-end");
		}


		// RESOLVE
		if(csResolve->program != -1)
		{ 

			GLTimerQueries::timestamp("resolve-start");

			glUseProgram(csResolve->program);

			glUniformMatrix4dv(0, 1, GL_FALSE, &viewProj[0][0]);
			glUniform2i(1, fbo->width, fbo->height);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, ssDepth.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, ss_D_RGBA.handle);

			glBindImageTexture(0, fbo->colorAttachments[0]->handle, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA8UI);

			int groups_x = fbo->width / 16;
			int groups_y = fbo->height / 16;
			glDispatchCompute(groups_x, groups_y, 1);

			GLTimerQueries::timestamp("resolve-end");
		}

		{ // CLEAR
			glMemoryBarrier(GL_ALL_BARRIER_BITS);

			GLuint zero = 0;
			float inf = -Infinity;
			GLuint intbits;
			memcpy(&intbits, &inf, 4);

			glClearNamedBufferData(ssDepth.handle, GL_R32UI, GL_RED, GL_UNSIGNED_INT, &intbits);
			glClearNamedBufferData(ss_D_RGBA.handle, GL_R32UI, GL_RED, GL_UNSIGNED_INT, &intbits);

			glMemoryBarrier(GL_ALL_BARRIER_BITS);
		}
		
		GLTimerQueries::timestamp("compute-points-end");
	}


};