
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

using namespace std;
using namespace std::chrono_literals;
using nlohmann::json;



struct ComputeParametric : public Method{

	string path = "";
	string source = "";
	Shader* csRender = nullptr;
	Shader* csResolve = nullptr;

	GLBuffer ssFramebuffer;
	GLBuffer ssDebug;

	Renderer* renderer = nullptr;

	ComputeParametric(Renderer* renderer){

		this->name = "parametric";
		this->description = R"ER01(

		)ER01";
		this->path = path;
		this->group = "misc";

		csRender = new Shader({ {"./modules/compute_parametric/render.cs", GL_COMPUTE_SHADER} });
		csResolve = new Shader({ {"./modules/compute_parametric/resolve.cs", GL_COMPUTE_SHADER} });

		this->renderer = renderer;

		ssFramebuffer = renderer->createBuffer(8 * 2048 * 2048);
		ssDebug = renderer->createBuffer(256);

		GLuint zero = 0;
		glClearNamedBufferData(ssDebug.handle, GL_R32UI, GL_RED, GL_UNSIGNED_INT, &zero);
	}
	
	void update(Renderer* renderer){

		// if(Runtime::resource != (Resource*)las.get()){

		// 	if(Runtime::resource != nullptr){
		// 		Runtime::resource->unload(renderer);
		// 	}

		// 	las->load(renderer);

		// 	Runtime::resource = (Resource*)las.get();
		// }

	}

	void render(Renderer* renderer) {

		GLTimerQueries::timestamp("compute-loop-start");

		auto fbo = renderer->views[0].framebuffer;
		auto camera = renderer->camera;

		// RENDER
		if(csRender->program != -1)
		{ 

			GLTimerQueries::timestamp("draw-start");

			glUseProgram(csRender->program);

			auto& viewLeft = renderer->views[0];

			mat4 world;
			mat4 view = viewLeft.view;
			mat4 proj = viewLeft.proj;
			mat4 worldView = view * world;
			mat4 viewProj = mat4(proj) * view;
			mat4 worldViewProj = proj * view * world;

			glUniformMatrix4fv(0, 1, GL_FALSE, &worldViewProj[0][0]);
			glUniformMatrix4fv(1, 1, GL_FALSE, &worldViewProj[0][0]);
			glUniformMatrix4fv(2, 1, GL_FALSE, &worldView[0][0]);
			glUniformMatrix4fv(3, 1, GL_FALSE, &proj[0][0]);

			glUniform3f(9, camera->position.x, camera->position.y, camera->position.z);
			glUniform2i(10, fbo->width, fbo->height);
			
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssFramebuffer.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 30, ssDebug.handle);

			glBindImageTexture(0, fbo->colorAttachments[0]->handle, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA8UI);

			int numCells = 100 * 100;
			
			glDispatchCompute(numCells, 1, 1);

			GLTimerQueries::timestamp("draw-end");
		}

		// DEBUG
		if(Debug::enableShaderDebugValue)
		{ 
			glMemoryBarrier(GL_ALL_BARRIER_BITS);

			auto dbg = renderer->readBuffer(ssDebug, 0, 4);

			Debug::getInstance()->values["debug value"] = formatNumber(dbg->data_i32[0]);

			glMemoryBarrier(GL_ALL_BARRIER_BITS);

		}


		// RESOLVE
		if(csResolve->program != -1)
		{ 

			GLTimerQueries::timestamp("resolve-start");

			glUseProgram(csResolve->program);
			
			{ // view 0
				glUniform2i(1, fbo->width, fbo->height);
				glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssFramebuffer.handle);

				glBindImageTexture(0, fbo->colorAttachments[0]->handle, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA8UI);

				int groups_x = fbo->width / 16;
				int groups_y = fbo->height / 16;
				glDispatchCompute(groups_x, groups_y, 1);
			}

			glMemoryBarrier(GL_ALL_BARRIER_BITS);

			GLTimerQueries::timestamp("resolve-end");
		}


		{ // CLEAR
			glMemoryBarrier(GL_ALL_BARRIER_BITS);

			GLuint zero = 0;
			float inf = -Infinity;
			GLuint intbits;
			memcpy(&intbits, &inf, 4);

			glClearNamedBufferData(ssFramebuffer.handle, GL_R32UI, GL_RED, GL_UNSIGNED_INT, &intbits);
			glClearNamedBufferData(ssDebug.handle, GL_R32UI, GL_RED, GL_UNSIGNED_INT, &zero);

			glMemoryBarrier(GL_ALL_BARRIER_BITS);
		}
		
		GLTimerQueries::timestamp("compute-loop-end");
	}


};