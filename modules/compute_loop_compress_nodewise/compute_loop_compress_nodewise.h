
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

using namespace std;
using namespace std::chrono_literals;
using nlohmann::json;



struct ComputeLoopCompressNodewise : public Method{

	string path = "";
	string source = "";
	Shader* csRender = nullptr;
	Shader* csResolve = nullptr;

	GLBuffer ssFramebuffer_0;
	GLBuffer ssFramebuffer_1;
	GLBuffer ssDebug;
	GLBuffer ssStats;

	int64_t numPoints = 0;
	dvec3 boxMin;
	dvec3 boxMax;

	shared_ptr<ProgressiveFileBuffer> buffer_batches = nullptr;
	shared_ptr<ProgressiveFileBuffer> buffer_position = nullptr;
	shared_ptr<ProgressiveFileBuffer> buffer_color = nullptr;


	ComputeLoopCompressNodewise(Renderer* renderer, string path, 
		shared_ptr<ProgressiveFileBuffer> buffer_batches,
		shared_ptr<ProgressiveFileBuffer> buffer_position,
		shared_ptr<ProgressiveFileBuffer> buffer_color){

		this->name = "loop_nodes_compressed";
		this->description = R"ER01(
- One workgroup per octree node
  (Variable loop sizes)
- Coordinates compressed with respect 
  to bounding box.
		)ER01";
		this->path = path;
		this->buffer_batches = buffer_batches;
		this->buffer_position = buffer_position;
		this->buffer_color = buffer_color;

		csRender = new Shader({ {"./modules/compute_loop_compress_nodewise/render.cs", GL_COMPUTE_SHADER} });
		csResolve = new Shader({ {"./modules/compute_loop_compress_nodewise/resolve.cs", GL_COMPUTE_SHADER} });

		ssFramebuffer_0 = renderer->createBuffer(8 * 2048 * 2048);
		ssFramebuffer_1 = renderer->createBuffer(8 * 2048 * 2048);

		ssDebug = renderer->createBuffer(256);
		ssStats = renderer->createBuffer(256);

		load();
	}

	void load(){
		auto strMetadata = readTextFile(path + "/compressed_nodewise/metadata.json");
		auto jsMetadata = json::parse(strMetadata);

		boxMin.x = jsMetadata["boundingBox"]["min"][0];
		boxMin.y = jsMetadata["boundingBox"]["min"][1];
		boxMin.z = jsMetadata["boundingBox"]["min"][2];
		boxMax.x = jsMetadata["boundingBox"]["max"][0];
		boxMax.y = jsMetadata["boundingBox"]["max"][1];
		boxMax.z = jsMetadata["boundingBox"]["max"][2];

		numPoints = jsMetadata["points"];

	}


	void update(Renderer* renderer){

		// auto camera = renderer->camera;
		// auto viewProj = camera->proj * camera->view;

		// Frustum frustum;
		// frustum.set(viewProj);

	}

	void render(Renderer* renderer) {

		GLTimerQueries::timestamp("compute-loop-start");

		buffer_batches->process();
		buffer_position->process();
		buffer_color->process();

		auto fbo = renderer->views[0].framebuffer;
		auto camera = renderer->camera;

		// RENDER
		if(csRender->program != -1)
		{ 

			GLTimerQueries::timestamp("draw-start");

			glUseProgram(csRender->program);

			auto& viewLeft = renderer->views[0];
			auto& viewRight = renderer->views[1];

			//mat4 translate = glm::translate(glm::vec3(-5.0f, -5.0f, 0.0f));
			//mat4 scale = glm::scale(glm::vec3(0.01f, 0.01f, 0.01f));
			//mat4 world = translate * scale;
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

			mat4 worldViewProjLeft = mat4(viewLeft.proj) * mat4(viewLeft.view) * world;
			mat4 worldViewProjRight = mat4(viewRight.proj) * mat4(viewRight.view) * world;
			glUniformMatrix4fv(20, 1, GL_FALSE, &worldViewProjLeft[0][0]);
			glUniformMatrix4fv(21, 1, GL_FALSE, &worldViewProjRight[0][0]);

			glUniform3f(5, boxMin.x, boxMin.y, boxMin.z);
			glUniform3f(6, boxMax.x, boxMax.y, boxMax.z);

			int numLoadedPoints = buffer_position->size / 8;
			glUniform1i(7, numPoints);
			glUniform1i(8, renderer->vrEnabled ? 1 : 0);
			glUniform1f(30, Debug::LOD);
			glUniform1i(31, Debug::lodEnabled ? 1 : 0);
			glUniform1i(32, Debug::frustumCullingEnabled ? 1 : 0);

			glUniform3f(9, camera->position.x, camera->position.y, camera->position.z);
			glUniform2i(10, fbo->width, fbo->height);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssFramebuffer_0.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, ssFramebuffer_1.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 11, buffer_batches->glBuffers[0]);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 30, ssDebug.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 31, ssStats.handle);

			for(int i = 0; i < buffer_position->glBuffers.size(); i++){
				glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 20 + i, buffer_position->glBuffers[i]);	
			}

			glBindImageTexture(0, fbo->colorAttachments[0]->handle, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA8UI);

			int numBatches = buffer_batches->size / 64;
			glDispatchCompute(numBatches, 1, 1);

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

		//{ // STATS
		//	//glMemoryBarrier(GL_ALL_BARRIER_BITS);

			//auto dbg = renderer->readBuffer(ssStats, 0, 8);

		//	cout << formatNumber(dbg->data_i32[0]) << ", " << formatNumber(dbg->data_i32[1]) << endl;

		//	//glMemoryBarrier(GL_ALL_BARRIER_BITS);

		//}


		// RESOLVE
		if(csResolve->program != -1)
		{ 

			GLTimerQueries::timestamp("resolve-start");

			glUseProgram(csResolve->program);
			
			{ // view 0
				glUniform2i(1, fbo->width, fbo->height);
				glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssFramebuffer_0.handle);

				for(int i = 0; i < buffer_color->glBuffers.size(); i++){
					glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 50 + i, buffer_color->glBuffers[i]);	
				}

				glBindImageTexture(0, fbo->colorAttachments[0]->handle, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA8UI);

				int groups_x = fbo->width / 16;
				int groups_y = fbo->height / 16;
				glDispatchCompute(groups_x, groups_y, 1);
			}

			glMemoryBarrier(GL_ALL_BARRIER_BITS);

			if(renderer->vrEnabled)
			{ // view 1
				auto view1 = renderer->views[1];
				auto fbo = view1.framebuffer;

				glUniform2i(1, fbo->width, fbo->height);
				glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssFramebuffer_1.handle);

				for(int i = 0; i < buffer_color->glBuffers.size(); i++){
					glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 50 + i, buffer_color->glBuffers[i]);	
				}

				glBindImageTexture(0, fbo->colorAttachments[0]->handle, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA8UI);

				int groups_x = fbo->width / 16;
				int groups_y = fbo->height / 16;
				glDispatchCompute(groups_x, groups_y, 1);
			}

			GLTimerQueries::timestamp("resolve-end");
		}

		{ // CLEAR
			glMemoryBarrier(GL_ALL_BARRIER_BITS);

			GLuint zero = 0;
			float inf = -Infinity;
			GLuint intbits;
			memcpy(&intbits, &inf, 4);

			glClearNamedBufferData(ssStats.handle, GL_R32UI, GL_RED, GL_UNSIGNED_INT, &zero);
			glClearNamedBufferData(ssDebug.handle, GL_R32UI, GL_RED, GL_UNSIGNED_INT, &zero);

			glClearNamedBufferData(ssFramebuffer_0.handle, GL_R32UI, GL_RED, GL_UNSIGNED_INT, &intbits);
			
			if(renderer->vrEnabled){
				glClearNamedBufferData(ssFramebuffer_1.handle, GL_R32UI, GL_RED, GL_UNSIGNED_INT, &intbits);
			}

			glMemoryBarrier(GL_ALL_BARRIER_BITS);
		}
		
		GLTimerQueries::timestamp("compute-loop-end");
	}


};