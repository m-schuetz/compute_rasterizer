
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



struct ComputeGL : public Method{

	struct UniformData{
		mat4 transform;
		mat4 world;
		mat4 view;
		mat4 proj;

		float time;
		glm::vec2 screenSize;
	};

	string source = "";
	Shader* shader = nullptr;

	GLBuffer uniformBuffer;

	shared_ptr<LasStandardData> las = nullptr;

	Renderer* renderer = nullptr;

	ComputeGL(Renderer* renderer, shared_ptr<LasStandardData> las){

		this->name = "(2021) GL_POINTS";
		this->description = R"ER01(
		)ER01";
		this->las = las;
		this->group = "2021 method; standard 16 byte per point";

		shader = new Shader( 
			"./modules/compute_2021_gl/points.vs",
			"./modules/compute_2021_gl/points.fs"
		);

		uniformBuffer = renderer->createUniformBuffer(512);

		this->renderer = renderer;
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

		GLTimerQueries::timestamp("compute-gl-start");

		las->process(renderer);

		if(las->numPointsLoaded == 0){
			return;
		}

		auto fbo = renderer->views[0].framebuffer;
		auto camera = renderer->camera;


		static bool initialized = false;
		static GLuint vao;
		static GLuint vbo = las->ssPoints.handle;
		
		if(!initialized){

			glCreateVertexArrays(1, &vao);

			glVertexArrayVertexBuffer(vao, 0, vbo, 0, 16);

			glEnableVertexArrayAttrib(vao, 0);
			glEnableVertexArrayAttrib(vao, 1);

			glVertexArrayAttribFormat(vao, 0, 3, GL_FLOAT, GL_FALSE, 0);
			glVertexArrayAttribFormat(vao, 1, 4, GL_UNSIGNED_BYTE, GL_TRUE, 12);

			glVertexArrayAttribBinding(vao, 0, 0);
			glVertexArrayAttribBinding(vao, 1, 0);

			initialized = true;
		}

		// Update Uniform Buffer
		{
			mat4 world;
			mat4 view = renderer->views[0].view;
			mat4 proj = renderer->views[0].proj;
			mat4 worldView = view * world;
			mat4 worldViewProj = proj * view * world;

			UniformData uniformData;
			uniformData.world = world;
			uniformData.view = view;
			uniformData.proj = proj;
			uniformData.transform = worldViewProj;
			uniformData.time = now();
			uniformData.screenSize = {fbo->width, fbo->height};

			glNamedBufferSubData(uniformBuffer.handle, 0, sizeof(UniformData), &uniformData);
		}

		glBindFramebuffer(GL_FRAMEBUFFER, fbo->handle);

		glUseProgram(shader->program);
		
		glBindBufferBase(GL_UNIFORM_BUFFER, 4, uniformBuffer.handle);

		glBindVertexArray(vao);

		glDrawArrays(GL_POINTS, 0, las->numPointsLoaded);

		glBindVertexArray(0);
		
		GLTimerQueries::timestamp("compute-gl-end");
	}


};