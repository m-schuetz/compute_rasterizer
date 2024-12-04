
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

#include "builtin_types.h"
#include "CudaProgram.h"
#include "cudaGL.h"
#include "compute_loop_las_cuda/kernel_data.h"

using namespace std;
using namespace std::chrono_literals;
using nlohmann::json;

struct ComputeLoopLasCUDA : public Method {

	bool registered = false;

	CudaProgram* resolveProg = nullptr;
	CudaProgram* renderProg = nullptr;

	CUdeviceptr fb;
	CUgraphicsResource output, rgba, batches, xyz12, xyz8, xyz4;
	CUevent start, end;

	shared_ptr<ComputeLasData> las = nullptr;
	Renderer* renderer = nullptr;

	ComputeLoopLasCUDA(Renderer* renderer, shared_ptr<ComputeLasData> las){

		this->name = "loop_las_cuda";
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

		cuMemAlloc(&fb, 8 * 2048 * 2048);
		
		this->renderer = renderer;

		resolveProg = new CudaProgram("./modules/compute_loop_las_cuda/resolve.cu");
		renderProg = new CudaProgram("./modules/compute_loop_las_cuda/render.cu");
		cuEventCreate(&start, CU_EVENT_DEFAULT);
		cuEventCreate(&end, CU_EVENT_DEFAULT);
	}

	~ComputeLoopLasCUDA() {
		if (registered)
		{
			std::vector< CUgraphicsResource> persistent_resources = { batches, xyz12, xyz8, xyz4, rgba };
			cuGraphicsUnmapResources(persistent_resources.size(), persistent_resources.data(), ((CUstream)CU_STREAM_DEFAULT));
			cuGraphicsUnregisterResource(rgba);
			cuGraphicsUnregisterResource(batches);
			cuGraphicsUnregisterResource(xyz12);
			cuGraphicsUnregisterResource(xyz8);
			cuGraphicsUnregisterResource(xyz4);
			cuGraphicsUnregisterResource(output);
		}
	}
	   
	void update(Renderer* renderer){
	}

	void render(Renderer* renderer) {

		GLTimerQueries::timestamp("compute-loop-start");

		las->process(renderer);

		auto fbo = renderer->views[0].framebuffer;
		auto camera = renderer->camera;

		if (renderProg->kernel == nullptr || resolveProg->kernel == nullptr)
			return;

		if(las->numPointsLoaded == 0){
			return;
		}

		if (!registered) {
			cuGraphicsGLRegisterBuffer(&rgba, las->ssColors.handle, CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY);
			cuGraphicsGLRegisterBuffer(&batches, las->ssBatches.handle, CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY);
			cuGraphicsGLRegisterBuffer(&xyz12, las->ssXyz_12b.handle, CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY);
			cuGraphicsGLRegisterBuffer(&xyz8, las->ssXyz_8b.handle, CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY);
			cuGraphicsGLRegisterBuffer(&xyz4, las->ssXyz_4b.handle, CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY);
			cuGraphicsGLRegisterImage(&output, renderer->views[0].framebuffer->colorAttachments[0]->handle, GL_TEXTURE_2D, CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD);

			std::vector< CUgraphicsResource> persistent_resources = { batches, xyz12, xyz8, xyz4, rgba };
			cuGraphicsMapResources(persistent_resources.size(), persistent_resources.data(), ((CUstream)CU_STREAM_DEFAULT));

			registered = true;
		}

		static CUdeviceptr batches_ptr, xyz12_ptr, xyz8_ptr, xyz4_ptr, rgba_ptr, bbs_ptr;
		size_t size;
		cuGraphicsResourceGetMappedPointer(&batches_ptr, &size, batches);
		cuGraphicsResourceGetMappedPointer(&xyz12_ptr, &size, xyz12);
		cuGraphicsResourceGetMappedPointer(&xyz8_ptr, &size, xyz8);
		cuGraphicsResourceGetMappedPointer(&xyz4_ptr, &size, xyz4);
		cuGraphicsResourceGetMappedPointer(&rgba_ptr, &size, rgba);

		// RENDER
		{
			//GLTimerQueries::timestamp("draw-start");

			auto& viewLeft = renderer->views[0];
			mat4 world;
			mat4 view = viewLeft.view;
			mat4 proj = viewLeft.proj;
			mat4 worldView = view * world;
			mat4 viewProj = mat4(proj) * view;
			mat4 worldViewProj = proj * view * world;

			ChangingRenderData cdata;
			*((glm::mat4*)&cdata.uTransform) = glm::transpose(worldViewProj);
			*((glm::mat4*)&cdata.uWorldView) = glm::transpose(worldView);
			*((glm::mat4*)&cdata.uProj) = glm::transpose(proj);
			cdata.uCamPos = float3{ (float)camera->position.x, (float)camera->position.y, (float)camera->position.z };
			cdata.uImageSize = int2{ fbo->width, fbo->height };
			cdata.uPointsPerThread = POINTS_PER_THREAD;
			cdata.uBoxMin = float3{ (float)las->boxMin.x, (float)las->boxMin.y, (float)las->boxMin.z };
			cdata.uBoxMax = float3{ (float)las->boxMax.x, (float)las->boxMax.y, (float)las->boxMax.z };
			cdata.uNumPoints = las->numPointsLoaded;
			cdata.uOffsetPointToData = las->offsetToPointData;
			cdata.uPointFormat = las->pointFormat;
			cdata.uBytesPerPoint = las->bytesPerPoint;
			cdata.uScale = float3{ (float)las->scale.x, (float)las->scale.y, (float)las->scale.z };
			cdata.uEnableFrustumCulling = Debug::frustumCullingEnabled;

			// don't execute a workgroup until all points inside are loaded
			// workgroup only iterates over the source buffer once, 
			// and generates a derivative!
			int numBatches = 0;
			numBatches = ceil(double(las->numPointsLoaded) / double(POINTS_PER_WORKGROUP));

			//cuEventRecord(start, (CUstream)CU_STREAM_DEFAULT);
			void* args[] = { &cdata, &fb, &batches_ptr, &xyz12_ptr, &xyz8_ptr, &xyz4_ptr, &rgba_ptr, &bbs_ptr };
			cuLaunchKernel(renderProg->kernel,
				numBatches, 1, 1,
				WORKGROUP_SIZE, 1, 1,
				0, 0, args, 0);
			//cuEventRecord(end, (CUstream)CU_STREAM_DEFAULT);
			//cuEventSynchronize(end);
			//float ms;
			//cuEventElapsedTime(&ms, start, end);
			//std::cout << ms << "ms" << std::endl;
		}

		// RESOLVE
		{ // view 0
			std::vector<CUgraphicsResource> dynamic_resources = { output };
			cuGraphicsMapResources(dynamic_resources.size(), dynamic_resources.data(), ((CUstream)CU_STREAM_DEFAULT));

			CUDA_RESOURCE_DESC res_desc = {};
			res_desc.resType = CUresourcetype::CU_RESOURCE_TYPE_ARRAY;
			cuGraphicsSubResourceGetMappedArray(&res_desc.res.array.hArray, output, 0, 0);
			CUsurfObject output_surf;
			cuSurfObjectCreate(&output_surf, &res_desc);

			int groups_x = fbo->width / 16;
			int groups_y = fbo->height / 16;
			void* args[] = { &fbo->width, &fbo->height, &output_surf, &fb, &rgba_ptr };

			cuLaunchKernel(resolveProg->kernel,
				groups_x, groups_y, 1,
				16, 16, 1,
				0, 0, args, 0);

			cuSurfObjectDestroy(output_surf);
			cuGraphicsUnmapResources(dynamic_resources.size(), dynamic_resources.data(), ((CUstream)CU_STREAM_DEFAULT));
		}

		//cuGraphicsUnmapResources(persistent_resources.size(), persistent_resources.data(), ((CUstream)CU_STREAM_DEFAULT));

		//GLTimerQueries::timestamp("draw-end");

		{ // CLEAR
			GLuint zero = 0;
			cuMemsetD8(fb, 0xFF, 8 * 2048 * 2048);
		}
		
		GLTimerQueries::timestamp("compute-loop-end");
	}


};