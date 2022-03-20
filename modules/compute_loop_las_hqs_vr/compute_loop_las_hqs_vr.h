
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
#include "compute/ComputeLasLoader.h"

using namespace std;
using namespace std::chrono_literals;
using nlohmann::json;

struct ComputeLoopLasHqsVR : public Method{

	string source = "";
	Shader* csDepth = nullptr;
	Shader* csColor = nullptr;
	Shader* csResolve = nullptr;

	GLBuffer ssLeft_depth;
	GLBuffer ssLeft_rgba;
	GLBuffer ssRight_depth;
	GLBuffer ssRight_rgba;

	GLBuffer ssDebug;
	GLBuffer ssBoundingBoxes;
	GLBuffer ssSelection;

	shared_ptr<ComputeLasData> las = nullptr;

	Renderer* renderer = nullptr;

	mat4 world_vr;

	vec3 brushPos;
	float brushSize = 0.0;
	bool brushIsActive = false;

	ComputeLoopLasHqsVR(Renderer* renderer, shared_ptr<ComputeLasData> las){

		this->name = "loop_las_hqs_vr";
		this->description = R"ER01(
Like compute las, but also 
averages overlapping points
		)ER01";
		this->las = las;
		this->group = "10-10-10 bit encoded";

		csDepth = new Shader({ {"./modules/compute_loop_las_hqs_vr/depth.cs", GL_COMPUTE_SHADER} });
		csColor = new Shader({ {"./modules/compute_loop_las_hqs_vr/color.cs", GL_COMPUTE_SHADER} });
		csResolve = new Shader({ {"./modules/compute_loop_las_hqs_vr/resolve.cs", GL_COMPUTE_SHADER} });

		ssLeft_depth = renderer->createBuffer(4 * 2048 * 2048 * 2);
		ssLeft_rgba = renderer->createBuffer(16 * 2048 * 2048 * 2);
		ssRight_depth = renderer->createBuffer(4 * 2048 * 2048 * 2);
		ssRight_rgba = renderer->createBuffer(16 * 2048 * 2048 * 2);

		ssDebug = renderer->createBuffer(256);
		ssBoundingBoxes = renderer->createBuffer(48 * 1'000'000);
		ssSelection = renderer->createBuffer(4 * las->numPoints);

		GLuint zero = 0;
		glClearNamedBufferData(ssDebug.handle, GL_R32UI, GL_RED, GL_UNSIGNED_INT, &zero);
		glClearNamedBufferData(ssBoundingBoxes.handle, GL_R32UI, GL_RED, GL_UNSIGNED_INT, &zero);
		glClearNamedBufferData(ssSelection.handle, GL_R32UI, GL_RED, GL_UNSIGNED_INT, &zero);

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

		GLTimerQueries::timestamp("compute-loop-start");

		las->process(renderer);

		auto fboLeft = renderer->views[0].framebuffer;
		auto fboRight = renderer->views[1].framebuffer;
		auto camera = renderer->camera;

		if(las->numPointsLoaded == 0){
			return;
		}

		// DEPTH
		if(csDepth->program != -1)
		{ 
			GLTimerQueries::timestamp("depth-start");

			glUseProgram(csDepth->program);

			{ // LEFT
				auto& viewParams = renderer->views[0];

				mat4 world = world_vr;
				mat4 view = viewParams.view;
				mat4 proj = viewParams.proj;
				mat4 worldView = view * world;
				mat4 viewProj = mat4(proj) * view;
				mat4 worldViewProj = proj * view * world;

				glUniformMatrix4fv(0, 1, GL_FALSE, &worldViewProj[0][0]);
				glUniformMatrix4fv(1, 1, GL_FALSE, &worldViewProj[0][0]);
				glUniformMatrix4fv(2, 1, GL_FALSE, &worldView[0][0]);
				glUniformMatrix4fv(3, 1, GL_FALSE, &proj[0][0]);
			}

			{ // RIGHT
				auto& viewParams = renderer->views[1];
				
				mat4 world = world_vr;
				mat4 view = viewParams.view;
				mat4 proj = viewParams.proj;
				mat4 worldView = view * world;
				mat4 viewProj = mat4(proj) * view;
				mat4 worldViewProj = proj * view * world;

				glUniformMatrix4fv(4, 1, GL_FALSE, &worldViewProj[0][0]);
				glUniformMatrix4fv(5, 1, GL_FALSE, &worldViewProj[0][0]);
				glUniformMatrix4fv(6, 1, GL_FALSE, &worldView[0][0]);
				glUniformMatrix4fv(7, 1, GL_FALSE, &proj[0][0]);
			}

			glUniform3f(9, camera->position.x, camera->position.y, camera->position.z);
			glUniform2i(10, fboLeft->width, fboLeft->height);
			glUniform1i(11, POINTS_PER_THREAD);
			glUniform1i(12, Debug::frustumCullingEnabled ? 1 : 0);
			glUniform1i(13, Debug::showBoundingBox ? 1 : 0);

			auto boxMin = las->boxMin;
			auto boxMax = las->boxMax;
			auto scale = las->scale;
			auto offset = las->offset;
			glUniform3f(20, boxMin.x, boxMin.y, boxMin.z);
			glUniform3f(21, boxMax.x, boxMax.y, boxMax.z);
			glUniform1i(22, las->numPointsLoaded);
			glUniform1i64ARB(23, las->offsetToPointData);
			glUniform1i(24, las->pointFormat);
			glUniform1i64ARB(25, las->bytesPerPoint);
			glUniform3f(26, scale.x, scale.y, scale.z);
			glUniform3d(27, offset.x, offset.y, offset.z);

			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssLeft_depth.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, ssLeft_rgba.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, ssRight_depth.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, ssRight_rgba.handle);

			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 30, ssDebug.handle);

			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 40, las->ssBatches.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 41, las->ssXyz_12b.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 42, las->ssXyz_8b.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 43, las->ssXyz_4b.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 44, las->ssColors.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 50, ssBoundingBoxes.handle);

			// glBindImageTexture(0, fboLeft->colorAttachments[0]->handle, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA8UI);

			int numBatches = ceil(double(las->numPointsLoaded) / double(POINTS_PER_WORKGROUP));
			
			glDispatchCompute(numBatches, 1, 1);

			GLTimerQueries::timestamp("depth-end");
		}

		// COLORS
		if(csColor->program != -1)
		{ 

			GLTimerQueries::timestamp("color-start");

			glUseProgram(csColor->program);

			{ // LEFT
				auto& viewParams = renderer->views[0];

				mat4 world = world_vr;
				mat4 view = viewParams.view;
				mat4 proj = viewParams.proj;
				mat4 worldView = view * world;
				mat4 viewProj = mat4(proj) * view;
				mat4 worldViewProj = proj * view * world;

				glUniformMatrix4fv(0, 1, GL_FALSE, &worldViewProj[0][0]);
				glUniformMatrix4fv(1, 1, GL_FALSE, &worldViewProj[0][0]);
				glUniformMatrix4fv(2, 1, GL_FALSE, &worldView[0][0]);
				glUniformMatrix4fv(3, 1, GL_FALSE, &proj[0][0]);
			}

			{ // RIGHT
				auto& viewParams = renderer->views[1];

				mat4 world = world_vr;
				mat4 view = viewParams.view;
				mat4 proj = viewParams.proj;
				mat4 worldView = view * world;
				mat4 viewProj = mat4(proj) * view;
				mat4 worldViewProj = proj * view * world;

				glUniformMatrix4fv(4, 1, GL_FALSE, &worldViewProj[0][0]);
				glUniformMatrix4fv(5, 1, GL_FALSE, &worldViewProj[0][0]);
				glUniformMatrix4fv(6, 1, GL_FALSE, &worldView[0][0]);
				glUniformMatrix4fv(7, 1, GL_FALSE, &proj[0][0]);
			}

			glUniform3f(9, camera->position.x, camera->position.y, camera->position.z);
			glUniform2i(10, fboLeft->width, fboLeft->height);
			glUniform1i(11, POINTS_PER_THREAD);
			glUniform1i(12, Debug::frustumCullingEnabled ? 1 : 0);
			glUniform1i(13, Debug::showBoundingBox ? 1 : 0);

			auto boxMin = las->boxMin;
			auto boxMax = las->boxMax;
			auto scale = las->scale;
			auto offset = las->offset;
			glUniform3f(20, boxMin.x, boxMin.y, boxMin.z);
			glUniform3f(21, boxMax.x, boxMax.y, boxMax.z);
			glUniform1i(22, las->numPointsLoaded);
			glUniform1i64ARB(23, las->offsetToPointData);
			glUniform1i(24, las->pointFormat);
			glUniform1i64ARB(25, las->bytesPerPoint);
			glUniform3f(26, scale.x, scale.y, scale.z);
			glUniform3d(27, offset.x, offset.y, offset.z);
			glUniform3f(30, brushPos.x, brushPos.y, brushPos.z);
			glUniform1f(31, brushSize);
			glUniform1i(32, brushIsActive ? 1 : 0);

			
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssLeft_depth.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, ssLeft_rgba.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, ssRight_depth.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, ssRight_rgba.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 30, ssDebug.handle);

			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 40, las->ssBatches.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 41, las->ssXyz_12b.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 42, las->ssXyz_8b.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 43, las->ssXyz_4b.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 44, las->ssColors.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 50, ssBoundingBoxes.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 51, ssSelection.handle);
			

			// glBindImageTexture(0, fboLeft->colorAttachments[0]->handle, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA8UI);

			int numBatches = ceil(double(las->numPointsLoaded) / double(POINTS_PER_WORKGROUP));
			
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


		// RESOLVE
		if(csResolve->program != -1)
		{ 

			GLTimerQueries::timestamp("resolve-start");

			glUseProgram(csResolve->program);
			
			{ // view 0
				glUniform2i(1, fboLeft->width, fboLeft->height);
				glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssLeft_depth.handle);
				glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, ssLeft_rgba.handle);
				glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, ssRight_depth.handle);
				glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, ssRight_rgba.handle);

				glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 44, las->ssColors.handle);

				glBindImageTexture(0, fboLeft->colorAttachments[0]->handle, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA8UI);
				glBindImageTexture(1, fboRight->colorAttachments[0]->handle, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA8UI);

				int groups_x = fboLeft->width / 16;
				int groups_y = fboLeft->height / 16;
				glDispatchCompute(groups_x, groups_y, 1);
			}

			glMemoryBarrier(GL_ALL_BARRIER_BITS);

			GLTimerQueries::timestamp("resolve-end");
		}

		auto ovr = OpenVRHelper::instance();
		Pose pose_left = ovr->getLeftControllerPose();
		Pose pose_right = ovr->getRightControllerPose();
		auto state_left = ovr->getLeftControllerState();
		auto state_right = ovr->getRightControllerState();

		auto flip = glm::dmat4(
			1.0, 0.0, 0.0, 0.0,
			0.0, 0.0, 1.0, 0.0,
			0.0, -1.0, 0.0, 0.0,
			0.0, 0.0, 0.0, 1.0
		);

		auto print = [](vec3 v){
			cout << v.x << ", " << v.y << ", " << v.z << endl;
		};

		static bool left_was_triggered = false;
		static bool right_was_triggered = false;
		static vec3 left_drag_start;
		static vec3 right_drag_start;
		static mat4 world_vr_start;

		bool left_is_triggered = false;
		bool right_is_triggered = false;
		bool was_transforming = left_was_triggered && right_was_triggered;

		if(state_left.ulButtonPressed != 0){
			auto mask_trigger = ButtonMaskFromId(vr::EVRButtonId::k_EButton_SteamVR_Trigger);
			left_is_triggered = (mask_trigger & state_left.ulButtonPressed) != 0ul;
		}

		if(state_right.ulButtonPressed != 0){
			auto mask_trigger = ButtonMaskFromId(vr::EVRButtonId::k_EButton_SteamVR_Trigger);
			right_is_triggered = (mask_trigger & state_right.ulButtonPressed) != 0ul;
		}

		bool is_transforming = left_is_triggered && right_is_triggered;
		bool is_dragging = left_is_triggered && !right_is_triggered;

		if(pose_right.valid){
			vec3 posRight = flip * pose_right.transform * dvec4{0.0, 0.0, 0.0, 1.0};

			vec3 p0 = glm::inverse(world_vr) * glm::vec4(0.0, 0.0, 0.0, 1.0);
			vec3 p1 = glm::inverse(world_vr) * glm::vec4(0.1, 0.0, 0.0, 1.0);

			brushPos = glm::inverse(world_vr) * glm::vec4(posRight, 1.0);
			brushSize = glm::distance(p0, p1);
		}

		brushIsActive = right_is_triggered && !left_is_triggered;


		if(is_transforming){

			vec3 posLeft = flip * pose_left.transform * dvec4{0.0, 0.0, 0.0, 1.0};
			vec3 posRight = flip * pose_right.transform * dvec4{0.0, 0.0, 0.0, 1.0};
			vec3 posCenter = (posLeft + posRight) / 2.0f;

			if(!left_was_triggered || !right_was_triggered){
				// start transforming
				left_drag_start = posLeft;
				right_drag_start = posRight;
				world_vr_start = world_vr;
			}

			vec3 diff_left = posLeft - left_drag_start;
			vec3 diff_right = posRight - right_drag_start;
			vec3 diff_center = posCenter - (left_drag_start + right_drag_start) / 2.0f;

			float length_before = glm::distance(left_drag_start, right_drag_start);
			float length_after = glm::distance(posLeft, posRight);

			float scale = length_after / length_before;

			auto angleOf = [](vec3 vec){
				glm::vec2 v2 = {vec.x, vec.y};

				return std::atan2(v2.y, v2.x);
			};

			float a1 = angleOf(posRight - posLeft);
			float a2 = angleOf(right_drag_start - left_drag_start);
			float angle = a1 - a2;

			mat4 mToOrigin = glm::translate(glm::mat4(), -posCenter);
			mat4 mScale = glm::scale(glm::mat4(), {scale, scale, scale});
			mat4 mToScene = glm::translate(glm::mat4(), posCenter);
			mat4 mRotY = glm::rotate(angle, vec3{0.0f, 0.0f, 1.0f});
			mat4 mTranslate = glm::translate(glm::mat4(), diff_center);


			world_vr = mTranslate * mToScene * mRotY * mScale * mToOrigin * world_vr_start;

		}
		
		if(was_transforming && !is_transforming){
			if(left_is_triggered){
				vec3 posLeft = flip * pose_left.transform * dvec4{0.0, 0.0, 0.0, 1.0};

				// start dragging
				left_drag_start = posLeft;
				world_vr_start = world_vr;
			}
		}
		
		if(is_dragging){

			vec3 posLeft = flip * pose_left.transform * dvec4{0.0, 0.0, 0.0, 1.0};

			if(!left_was_triggered && left_is_triggered){
				// start dragging!
				left_drag_start = posLeft;
				world_vr_start = world_vr;
			}else if(left_was_triggered && !left_is_triggered){
				// stop dragging!
				
			}

			vec3 diff = posLeft - left_drag_start;
			mat4 translate = glm::translate(glm::mat4(), diff);

			world_vr = translate * world_vr_start;
		}

		left_was_triggered = left_is_triggered;
		right_was_triggered = right_is_triggered;


		

		if(pose_left.valid){
			dvec4 posLeft = flip * pose_left.transform * dvec4{0.0, 0.0, 0.0, 1.0};
			renderer->drawBoundingBox(dvec3(posLeft), {0.1, 0.1, 0.1}, {200, 0, 0});
		}

		if(pose_right.valid){
			dvec4 posRight = flip * pose_right.transform * dvec4{0.0, 0.0, 0.0, 1.0};
			renderer->drawBoundingBox(dvec3(posRight), {0.1, 0.1, 0.1}, {0, 200, 0});
		}

		for(int i : {0, 1})
		{ 
			auto& viewParams = renderer->views[i];
			glBindFramebuffer(GL_FRAMEBUFFER, viewParams.framebuffer->handle);

			glViewport(0, 0, viewParams.framebuffer->width, viewParams.framebuffer->height);

			auto camera_vr_world = make_shared<Camera>();
			camera_vr_world->view = viewParams.view;
			camera_vr_world->proj = viewParams.proj;

			renderer->drawBoundingBoxes(camera_vr_world.get(), ssBoundingBoxes);
		}


		{ // CLEAR

			GLTimerQueries::timestamp("clear-start");

			glMemoryBarrier(GL_ALL_BARRIER_BITS);

			GLuint zero = 0;
			float inf = -Infinity;
			GLuint intbits;
			memcpy(&intbits, &inf, 4);

			glClearNamedBufferData(ssLeft_depth.handle, GL_R32UI, GL_RED, GL_UNSIGNED_INT, &intbits);
			glClearNamedBufferData(ssLeft_rgba.handle, GL_R32UI, GL_RED, GL_UNSIGNED_INT, &zero);
			glClearNamedBufferData(ssRight_depth.handle, GL_R32UI, GL_RED, GL_UNSIGNED_INT, &intbits);
			glClearNamedBufferData(ssRight_rgba.handle, GL_R32UI, GL_RED, GL_UNSIGNED_INT, &zero);
			
			glClearNamedBufferData(ssDebug.handle, GL_R32UI, GL_RED, GL_UNSIGNED_INT, &zero);
			glClearNamedBufferSubData(ssBoundingBoxes.handle, GL_R32UI, 0, 64, GL_RED, GL_UNSIGNED_INT, &zero);

			glMemoryBarrier(GL_ALL_BARRIER_BITS);

			GLTimerQueries::timestamp("clear-end");
		}
		
		GLTimerQueries::timestamp("compute-loop-end");
	}


};