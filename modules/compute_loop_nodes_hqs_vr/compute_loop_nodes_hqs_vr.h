
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
#include "compute/PotreeData.h"
#include "Runtime.h"

using namespace std;
using namespace std::chrono_literals;
using nlohmann::json;



struct ComputeLoopNodesHqsVr : public Method{

	struct UniformData{
		mat4 left_world;
		mat4 left_view;
		mat4 left_proj;
		mat4 left_transform;
		mat4 left_transformFrustum;
		mat4 right_world;
		mat4 right_view;
		mat4 right_proj;
		mat4 right_transform;
		mat4 right_transformFrustum;

		int pointsPerThread;
		int enableFrustumCulling;
		int showBoundingBox;
		int numPoints;
		ivec2 imageSize;
	};

	struct DebugData{
		uint32_t value = 0;
		bool enabled = false;
		uint32_t depth_numPointsProcessed = 0;
		uint32_t depth_numNodesProcessed = 0;
		uint32_t depth_numPointsRendered = 0;
		uint32_t depth_numNodesRendered = 0;
		uint32_t color_numPointsProcessed = 0;
		uint32_t color_numNodesProcessed = 0;
		uint32_t color_numPointsRendered = 0;
		uint32_t color_numNodesRendered = 0;
		uint32_t numPointsVisible = 0;
	};

	string path = "";
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
	GLBuffer uniformBuffer;
	UniformData uniformData;

	shared_ptr<PotreeData> potreeData;

	Renderer* renderer = nullptr;

	mat4 world_vr;

	vec3 brushPos;
	float brushSize = 0.0;
	bool brushIsActive = false;

	ComputeLoopNodesHqsVr(Renderer* renderer, shared_ptr<PotreeData> potreeData){

		this->name = "loop_nodes_hqs_vr";
		this->description = R"ER01(
- One workgroup per octree node
  (Variable loop sizes)
- 8 byte encoded coordinates
		)ER01";
		this->potreeData = potreeData;
		this->group = "render Potree nodes";

		csDepth = new Shader({ {"./modules/compute_loop_nodes_hqs_vr/depth.cs", GL_COMPUTE_SHADER} });
		csColor = new Shader({ {"./modules/compute_loop_nodes_hqs_vr/color.cs", GL_COMPUTE_SHADER} });
		csResolve = new Shader({ {"./modules/compute_loop_nodes_hqs_vr/resolve.cs", GL_COMPUTE_SHADER} });

		ssLeft_depth = renderer->createBuffer(4 * 2048 * 2048 * 2);
		ssLeft_rgba = renderer->createBuffer(16 * 2048 * 2048 * 2);
		ssRight_depth = renderer->createBuffer(4 * 2048 * 2048 * 2);
		ssRight_rgba = renderer->createBuffer(16 * 2048 * 2048 * 2);
		ssDebug = renderer->createBuffer(256);
		ssBoundingBoxes = renderer->createBuffer(48 * 1'000'000);
		uniformBuffer = renderer->createUniformBuffer(1024);

		GLuint zero = 0;
		glClearNamedBufferData(ssDebug.handle, GL_R32UI, GL_RED, GL_UNSIGNED_INT, &zero);
		glClearNamedBufferData(ssBoundingBoxes.handle, GL_R32UI, GL_RED, GL_UNSIGNED_INT, &zero);

		this->renderer = renderer;
	}

	void update(Renderer* renderer){

		if(Runtime::resource != (Resource*)potreeData.get()){

			if(Runtime::resource != nullptr){
				Runtime::resource->unload(renderer);
			}

			potreeData->load(renderer);

			Runtime::resource = (Resource*)potreeData.get();
		}

	}

	void render(Renderer* renderer) {

		GLTimerQueries::timestamp("compute-loop-start");

		potreeData->process(renderer);

		if (potreeData->numPointsLoaded == 0) {
			return;
		}

		auto fboLeft = renderer->views[0].framebuffer;
		auto fboRight = renderer->views[1].framebuffer;
		auto camera = renderer->camera;

		// Update Uniform Buffer
		{

			if(Debug::requestCopyVrMatrices){

				auto tostring = [](mat4 matrix, string name){
					stringstream ss;
					ss << std::setprecision(4) << std::fixed;

					ss << name << "[0] = {" << matrix[0].x << ", " << matrix[0].y << ", " << matrix[0].z << ", " << matrix[0].w << "};" << endl;
					ss << name << "[1] = {" << matrix[1].x << ", " << matrix[1].y << ", " << matrix[1].z << ", " << matrix[1].w << "};" << endl;
					ss << name << "[2] = {" << matrix[2].x << ", " << matrix[2].y << ", " << matrix[2].z << ", " << matrix[2].w << "};" << endl;
					ss << name << "[3] = {" << matrix[3].x << ", " << matrix[3].y << ", " << matrix[3].z << ", " << matrix[3].w << "};" << endl;

					return ss.str(); 
				};

				stringstream ss;
				ss << std::setprecision(4) << std::fixed;
				ss << tostring(world_vr, "world_vr") << endl;
				ss << tostring(renderer->views[0].view, "renderer->views[0].view") << endl;
				ss << tostring(renderer->views[0].proj, "renderer->views[0].proj") << endl;
				ss << tostring(renderer->views[1].view, "renderer->views[1].view") << endl;
				ss << tostring(renderer->views[1].proj, "renderer->views[1].proj") << endl;

				string str = ss.str();

				toClipboard(str);

				Debug::requestCopyVrMatrices = false;
			}

			// MORRO BAY
			if(Debug::dummyVR){
				world_vr[0] = {0.0007, -0.0005, 0.0000, 0.0000};
				world_vr[1] = {0.0005, 0.0007, 0.0000, 0.0000};
				world_vr[2] = {0.0000, 0.0000, 0.0009, 0.0000};
				world_vr[3] = {-1.8058, -0.0005, 0.5107, 1.0000};

				renderer->views[0].view[0] = {-0.0340, 0.1428, -0.9892, -0.0000};
				renderer->views[0].view[1] = {-0.9994, -0.0016, 0.0341, 0.0000};
				renderer->views[0].view[2] = {0.0033, 0.9898, 0.1428, -0.0000};
				renderer->views[0].view[3] = {-0.2707, -0.6063, -0.3805, 1.0000};

				renderer->views[0].proj[0] = {0.7709, 0.0000, 0.0000, 0.0000};
				renderer->views[0].proj[1] = {0.0000, 0.7087, 0.0000, 0.0000};
				renderer->views[0].proj[2] = {-0.1880, -0.0019, 0.0000, -1.0000};
				renderer->views[0].proj[3] = {0.0000, 0.0000, 0.0100, 0.0000};

				renderer->views[1].view[0] = {-0.0340, 0.1428, -0.9892, -0.0000};
				renderer->views[1].view[1] = {-0.9994, -0.0016, 0.0341, 0.0000};
				renderer->views[1].view[2] = {0.0033, 0.9898, 0.1428, -0.0000};
				renderer->views[1].view[3] = {-0.3337, -0.6063, -0.3805, 1.0000};

				renderer->views[1].proj[0] = {0.7721, 0.0000, 0.0000, 0.0000};
				renderer->views[1].proj[1] = {0.0000, 0.7092, 0.0000, 0.0000};
				renderer->views[1].proj[2] = {0.1855, 0.0034, 0.0000, -1.0000};
				renderer->views[1].proj[3] = {0.0000, 0.0000, 0.0100, 0.0000};
			}

			// NIEDERWEIDEN
			// if(Debug::dummyVR){
			// 	world_vr[0] = {-0.0092, 0.0029, 0.0000, 0.0000};
			// 	world_vr[1] = {-0.0029, -0.0092, 0.0000, 0.0000};
			// 	world_vr[2] = {0.0000, 0.0000, 0.0096, 0.0000};
			// 	world_vr[3] = {1.0645, 0.5180, 0.7516, 1.0000};

			// 	renderer->views[0].view[0] = {-0.5680, -0.0032, -0.8231, 0.0000};
			// 	renderer->views[0].view[1] = {-0.8230, 0.0164, 0.5678, 0.0000};
			// 	renderer->views[0].view[2] = {0.0117, 0.9999, -0.0119, -0.0000};
			// 	renderer->views[0].view[3] = {0.0125, -0.8532, -0.0563, 1.0000};

			// 	renderer->views[0].proj[0] = {0.7709, 0.0000, 0.0000, 0.0000};
			// 	renderer->views[0].proj[1] = {0.0000, 0.7087, 0.0000, 0.0000};
			// 	renderer->views[0].proj[2] = {-0.1880, -0.0019, 0.0000, -1.0000};
			// 	renderer->views[0].proj[3] = {0.0000, 0.0000, 0.0100, 0.0000};

			// 	renderer->views[1].view[0] = {-0.5680, -0.0032, -0.8231, 0.0000};
			// 	renderer->views[1].view[1] = {-0.8230, 0.0164, 0.5678, 0.0000};
			// 	renderer->views[1].view[2] = {0.0117, 0.9999, -0.0119, -0.0000};
			// 	renderer->views[1].view[3] = {-0.0505, -0.8532, -0.0563, 1.0000};

			// 	renderer->views[1].proj[0] = {0.7721, 0.0000, 0.0000, 0.0000};
			// 	renderer->views[1].proj[1] = {0.0000, 0.7092, 0.0000, 0.0000};
			// 	renderer->views[1].proj[2] = {0.1855, 0.0034, 0.0000, -1.0000};
			// 	renderer->views[1].proj[3] = {0.0000, 0.0000, 0.0100, 0.0000};
			// }

			// world_vr[0] = {0.0410, -0.0576, 0.0000, 0.0000};
			// world_vr[1] = {0.0576, 0.0410, 0.0000, 0.0000};
			// world_vr[2] = {0.0000, 0.0000, 0.0706, 0.0000};
			// world_vr[3] = {-3.1293, 0.3342, 0.6933, 1.0000};

			// Banyunibo
			// if(Debug::dummyVR){
			// 	world_vr[0] = {0.0410, -0.0576, 0.0000, 0.0000};
			// 	world_vr[1] = {0.0576, 0.0410, 0.0000, 0.0000};
			// 	world_vr[2] = {0.0000, 0.0000, 0.0706, 0.0000};
			// 	world_vr[3] = {-3.1293, 0.3342, 0.6933, 1.0000};

			// 	renderer->views[0].view[0] = {-0.9831, 0.1072, -0.1484, -0.0000};
			// 	renderer->views[0].view[1] = {-0.1579, -0.0857, 0.9837, 0.0000};
			// 	renderer->views[0].view[2] = {0.0927, 0.9905, 0.1012, -0.0000};
			// 	renderer->views[0].view[3] = {-0.1039, -1.0851, -0.0625, 1.0000};

			// 	renderer->views[0].proj[0] = {0.7709, 0.0000, 0.0000, 0.0000};
			// 	renderer->views[0].proj[1] = {0.0000, 0.7087, 0.0000, 0.0000};
			// 	renderer->views[0].proj[2] = {-0.1880, -0.0019, 0.0000, -1.0000};
			// 	renderer->views[0].proj[3] = {0.0000, 0.0000, 0.0100, 0.0000};

			// 	renderer->views[1].view[0] = {-0.9831, 0.1072, -0.1484, -0.0000};
			// 	renderer->views[1].view[1] = {-0.1579, -0.0857, 0.9837, 0.0000};
			// 	renderer->views[1].view[2] = {0.0927, 0.9905, 0.1012, -0.0000};
			// 	renderer->views[1].view[3] = {-0.1650, -1.0851, -0.0625, 1.0000};

			// 	renderer->views[1].proj[0] = {0.7721, 0.0000, 0.0000, 0.0000};
			// 	renderer->views[1].proj[1] = {0.0000, 0.7092, 0.0000, 0.0000};
			// 	renderer->views[1].proj[2] = {0.1855, 0.0034, 0.0000, -1.0000};
			// 	renderer->views[1].proj[3] = {0.0000, 0.0000, 0.0100, 0.0000};
			// }

			{ // LEFT
				auto& viewParams = renderer->views[0];

				mat4 world = world_vr;
				mat4 view = viewParams.view;
				mat4 proj = viewParams.proj;
				mat4 worldViewProj = proj * view * world;

				uniformData.left_world = world;
				uniformData.left_view = view;
				uniformData.left_proj = proj;
				uniformData.left_transform = worldViewProj;

				if(Debug::updateFrustum){
					uniformData.left_transformFrustum = worldViewProj;
				}
			}

			{ // RIGHT
				auto& viewParams = renderer->views[1];

				mat4 world = world_vr;
				mat4 view = viewParams.view;
				mat4 proj = viewParams.proj;
				mat4 worldViewProj = proj * view * world;

				uniformData.right_world = world;
				uniformData.right_view = view;
				uniformData.right_proj = proj;
				uniformData.right_transform = worldViewProj;

				if(Debug::updateFrustum){
					uniformData.right_transformFrustum = worldViewProj;
				}
			}

			uniformData.pointsPerThread = POINTS_PER_THREAD;
			uniformData.numPoints = potreeData->numPointsLoaded;
			uniformData.enableFrustumCulling = Debug::frustumCullingEnabled ? 1 : 0;
			uniformData.showBoundingBox = Debug::showBoundingBox ? 1 : 0;
			uniformData.imageSize = {fboLeft->width, fboLeft->height};

			Debug::set("fbo size", formatNumber(fboLeft->width) + " x " + formatNumber(fboLeft->height));

			glNamedBufferSubData(uniformBuffer.handle, 0, sizeof(UniformData), &uniformData);
		}

		if(Debug::enableShaderDebugValue){
			DebugData data;
			data.enabled = true;

			glNamedBufferSubData(ssDebug.handle, 0, sizeof(DebugData), &data);
		}

		// DEPTH
		if(csDepth->program != -1){ 
			GLTimerQueries::timestamp("depth-start");

			glUseProgram(csDepth->program);

			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssLeft_depth.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, ssLeft_rgba.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, ssRight_depth.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, ssRight_rgba.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 30, ssDebug.handle);
			glBindBufferBase(GL_UNIFORM_BUFFER, 31, uniformBuffer.handle);
			
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 11, potreeData->ssBatches.handle);

			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 41, potreeData->ssXyz_12b.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 42, potreeData->ssXyz_8b.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 43, potreeData->ssXyz_4b.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 44, potreeData->ssColors.handle);

			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 50, ssBoundingBoxes.handle);

			glBindImageTexture(0, fboLeft->colorAttachments[0]->handle, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA8UI);

			int numBatches = potreeData->nodes.size();
			glDispatchCompute(numBatches, 1, 1);

			GLTimerQueries::timestamp("depth-end");
		}

		// COLOR
		if(csColor->program != -1){ 
			GLTimerQueries::timestamp("color-start");

			glUseProgram(csColor->program);

			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssLeft_depth.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, ssLeft_rgba.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, ssRight_depth.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, ssRight_rgba.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 30, ssDebug.handle);
			glBindBufferBase(GL_UNIFORM_BUFFER, 31, uniformBuffer.handle);
			
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 11, potreeData->ssBatches.handle);

			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 41, potreeData->ssXyz_12b.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 42, potreeData->ssXyz_8b.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 43, potreeData->ssXyz_4b.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 44, potreeData->ssColors.handle);

			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 50, ssBoundingBoxes.handle);

			glBindImageTexture(0, fboLeft->colorAttachments[0]->handle, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA8UI);

			int numBatches = potreeData->nodes.size();
			glDispatchCompute(numBatches, 1, 1);

			GLTimerQueries::timestamp("color-end");
		}


		// RESOLVE
		if(csResolve->program != -1){ 
			GLTimerQueries::timestamp("resolve-start");

			glUseProgram(csResolve->program);
			
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssLeft_depth.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, ssLeft_rgba.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, ssRight_depth.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, ssRight_rgba.handle);
			glBindBufferBase(GL_UNIFORM_BUFFER, 31, uniformBuffer.handle);

			glBindImageTexture(0, fboLeft->colorAttachments[0]->handle, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA8UI);
			glBindImageTexture(1, fboRight->colorAttachments[0]->handle, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA8UI);

			int groups_x = fboLeft->width / 16;
			int groups_y = fboLeft->height / 16;
			glDispatchCompute(groups_x, groups_y, 1);
		
			glMemoryBarrier(GL_ALL_BARRIER_BITS);

			GLTimerQueries::timestamp("resolve-end");
		}

		// READ DEBUG VALUES
		if(Debug::enableShaderDebugValue){
			glMemoryBarrier(GL_ALL_BARRIER_BITS);

			DebugData data;
			glGetNamedBufferSubData(ssDebug.handle, 0, sizeof(DebugData), &data);

			// Debug::getInstance()->values["debug value"] = formatNumber(data.value);

			auto dbg = Debug::getInstance();

			dbg->pushFrameStat("[depth] #nodes processed" , formatNumber(data.depth_numNodesProcessed));
			dbg->pushFrameStat("[depth] #nodes rendered"  , formatNumber(data.depth_numNodesRendered));
			dbg->pushFrameStat("[depth] #points processed", formatNumber(data.depth_numPointsProcessed));
			dbg->pushFrameStat("[depth] #points rendered" , formatNumber(data.depth_numPointsRendered));
			dbg->pushFrameStat("divider" , "");
			dbg->pushFrameStat("[color] #nodes processed" , formatNumber(data.color_numNodesProcessed));
			dbg->pushFrameStat("[color] #nodes rendered"  , formatNumber(data.color_numNodesRendered));
			dbg->pushFrameStat("[color] #points processed", formatNumber(data.color_numPointsProcessed));
			dbg->pushFrameStat("[color] #points rendered" , formatNumber(data.color_numPointsRendered));
			
			dbg->pushFrameStat("divider" , "");

			dbg->pushFrameStat("#points visible"          , formatNumber(data.numPointsVisible));

			glMemoryBarrier(GL_ALL_BARRIER_BITS);
		}

		// INTERACTION
		if(!Debug::dummyVR)
		{ 
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
		}

		for(int i : {0, 1})
		{ 
			auto& viewParams = renderer->views[i];
			glBindFramebuffer(GL_FRAMEBUFFER, viewParams.framebuffer->handle);

			glViewport(0, 0, viewParams.framebuffer->width, viewParams.framebuffer->height);

			auto camera_vr_world = make_shared<Camera>();
			camera_vr_world->view = viewParams.view * dmat4(world_vr);
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

			glClearNamedBufferSubData(ssLeft_depth.handle , GL_R32UI, 0, fboLeft->width * fboLeft->height * 4    , GL_RED, GL_UNSIGNED_INT, &intbits);
			glClearNamedBufferSubData(ssRight_depth.handle, GL_R32UI, 0, fboRight->width * fboRight->height * 4  , GL_RED, GL_UNSIGNED_INT, &intbits);
			glClearNamedBufferSubData(ssLeft_rgba.handle  , GL_R32UI, 0, fboLeft->width * fboLeft->height * 8    , GL_RED, GL_UNSIGNED_INT, &zero);
			glClearNamedBufferSubData(ssRight_rgba.handle , GL_R32UI, 0, fboRight->width * fboRight->height * 8  , GL_RED, GL_UNSIGNED_INT, &zero);

			glClearNamedBufferData(ssDebug.handle, GL_R32UI, GL_RED, GL_UNSIGNED_INT, &zero);
			glClearNamedBufferSubData(ssBoundingBoxes.handle, GL_R32UI, 0, 48, GL_RED, GL_UNSIGNED_INT, &zero);

			glMemoryBarrier(GL_ALL_BARRIER_BITS);
			GLTimerQueries::timestamp("clear-end");
		}
		
		GLTimerQueries::timestamp("compute-loop-end");
	}


};