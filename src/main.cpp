

#include <iostream>
#include <filesystem>

#include "GLBuffer.h"
#include "Renderer.h"
#include "Shader.h"
#include "compute_basic.h"
#include "ProgressiveFileBuffer.h"
#include "compute_points/compute_points.h"

#include "compute_loop_compress_nodewise/compute_loop_compress_nodewise.h"
#include "compute_loop_las/compute_loop_las.h"
#include "compute_loop_las2/compute_loop_las2.h"
#include "compute_loop_las_cuda/compute_loop_las_cuda.h"
#include "compute_loop_las_hqs/compute_loop_las_hqs.h"
#include "compute_loop_las_hqs_vr/compute_loop_las_hqs_vr.h"
#include "compute_loop_nodes/compute_loop_nodes.h"
#include "compute_loop_nodes_hqs/compute_loop_nodes_hqs.h"
#include "compute_loop_nodes_hqs_vr/compute_loop_nodes_hqs_vr.h"

#include "compute_2021_earlyz/compute_2021_earlyz.h"
#include "compute_2021_earlyz_reduce/compute_2021_earlyz_reduce.h"
#include "compute_2021_dedup/compute_2021_dedup.h"
#include "compute_2021_hqs/compute_2021_hqs.h"
#include "compute_2021_gl/compute_2021_gl.h"

#include "compute_parametric/compute_parametric.h"

//#include "VrRuntime.h"
#include "Runtime.h"
#include "Method.h"
#include "compute/ComputeLasLoader.h"
#include "compute/LasLoaderStandard.h"
#include "compute/PotreeData.h"

#include "compute/LasLoaderSparse.h"



using namespace std;

int numPoints = 1'000'000;

int main(){

	cout << std::setprecision(2) << std::fixed;

	auto renderer = make_shared<Renderer>();
	//renderer->init();

	// Creating a CUDA context
	cuInit(0);
	CUdevice cuDevice;
	CUcontext context;
	cuDeviceGet(&cuDevice, 0);
	cuCtxCreate(&context, 0, cuDevice);

	auto tStart = now();

	// struct Setting{
	// 	string path_potree = "";
	// 	string path_las = "";
	// 	float yaw = 0.0;
	// 	float pitch = 0.0;
	// 	float radius = 0.0;
	// 	dvec3 target;
	// };
	
	// unordered_map<string, Setting> settings;
	
	// { // RETZ
	// 	Setting setting;
	// 	setting.path_potree = "F:/temp/wgtest/retz/potree"; 
	// 	setting.path_las = "F:/temp/wgtest/retz/pointcloud.las"; 
	// 	setting.yaw = 6.91;
	// 	setting.pitch = -0.78;
	// 	setting.radius = 569.49;
	// 	setting.target = {569.57, 867.56, 29.91};

	// 	settings["retz"] = setting;
	// }
	
	// { // Eclepens
	// 	Setting setting;
	// 	setting.path_potree = "F:/temp/wgtest/eclepens/potree";
	// 	setting.path_las = "F:/temp/wgtest/eclepens/eclepens.las";
	// 	setting.yaw = 6.65;
	// 	setting.pitch = -0.71;
	// 	setting.radius = 1109.77;
	// 	setting.target = {514.05, 475.84, -156.43};

	// 	settings["eclepens"] = setting;
	// }

	// { // Eclepens - morton
	// 	Setting setting;
	// 	setting.path_potree = "F:/temp/wgtest/eclepens/potree";
	// 	setting.path_las = "F:/temp/wgtest/eclepens/eclepens_morton.las";
	// 	setting.yaw = 6.65;
	// 	setting.pitch = -0.71;
	// 	setting.radius = 1109.77;
	// 	setting.target = {514.05, 475.84, -156.43};

	// 	settings["eclepens_morton"] = setting;
	// }
	
	// { // Eclepens at home
	// 	Setting setting;
	// 	setting.path_potree = "E:/dev/pointclouds/paper_testdata/eclepens/potree";
	// 	setting.path_las = "E:/dev/pointclouds/paper_testdata/eclepens/eclepens_morton.las";
	// 	setting.yaw = 6.65;
	// 	setting.pitch = -0.71;
	// 	setting.radius = 1109.77;
	// 	setting.target = {514.05, 475.84, -156.43};

	// 	settings["eclepens_morton_home"] = setting;
	// }


	// {// MORRO BAY
	// 	Setting setting;
	// 	setting.path_potree = "F:/temp/wgtest/morro_bay/potree_morro_bay_278M_morton"; 
	// 	setting.path_las = "F:/temp/wgtest/morro_bay/morro_bay_278M_morton.las"; 
	// 	setting.yaw = -0.15;
	// 	setting.pitch = -0.57;
	// 	setting.radius = 3166.32;
	// 	setting.target = {2239.05, 1713.63, -202.02};

	// 	settings["morrobay_morton"] = setting;
	// }

	// { // ENDEAVOR
	// 	Setting setting;
	// 	setting.path_potree = "F:/temp/wgtest/endeavor/potree"; 
	// 	setting.path_las = "F:/temp/wgtest/endeavor/morton.las"; 
	// 	setting.yaw = -12.08;
	// 	setting.pitch = -0.60;
	// 	setting.radius = 149.96;
	// 	setting.target = {609.85, 610.98, 510.22};

	// 	settings["endeavor"] = setting;
	// }


	// { // Candi Banyunibo outside
	// 	Setting setting;
	// 	setting.path_potree = "F:/temp/wgtest/candi_banyunibo/potree"; 
	// 	setting.path_las = "F:/temp/wgtest/candi_banyunibo/morton.las"; 
	// 	setting.yaw = -13.63;
	// 	setting.pitch = 0.07;
	// 	setting.radius = 20.26;
	// 	setting.target = {38.64, 29.22, 5.23};

	// 	settings["banyunibo_outside_morton"] = setting;
	// }

	// { // Candi Banyunibo inside
	// 	Setting setting;
	// 	setting.path_potree = "F:/temp/wgtest/candi_banyunibo/potree"; 
	// 	setting.path_las = "F:/temp/wgtest/candi_banyunibo/morton.las"; 
	// 	setting.yaw = -18.35;
	// 	setting.pitch = 0.32;
	// 	setting.radius = 3.01;
	// 	setting.target = {41.48, 31.31, 4.96};

	// 	settings["banyunibo_inside_morton"] = setting;
	// }


	// { // Niederweiden
	// 	Setting setting;
	// 	setting.path_potree = "F:/temp/wgtest/niederweiden/potree"; 
	// 	setting.path_las = "F:/temp/wgtest/niederweiden/morton.las"; 
	// 	setting.yaw = 14.51;
	// 	setting.pitch = -0.59;
	// 	setting.radius = 76.96;
	// 	setting.target = {62.38, 85.53, 3.14};

	// 	settings["niederweiden_morton"] = setting;
	// }

	// { // Test
	// 	Setting setting;
	// 	setting.path_potree = "D:/dev/pointclouds/tuwien_baugeschichte/candi Banyunibo/Scan10 - POLYDATA - Candi_Banyunibo010.las_converted";
	// 	//setting.path_las = "D:/dev/pointclouds/tuwien_baugeschichte/candi Banyunibo/morton/10.las";
	// 	// setting.path_las = "D:/dev/pointclouds/tuwien_baugeschichte/candi Banyunibo/candi_banyunibo.las";
	// 	setting.path_las = "F:/temp/wgtest/banyunibo_laserscans/merged.las";

	// 	// outside
	// 	setting.yaw = 4.39;
	// 	setting.pitch = -0.18;
	// 	setting.radius = 8.60;
	// 	setting.target = {25.09, 36.09, 2.77};

	// 	// Inside
	// 	setting.yaw = 6.69;
	// 	setting.pitch = -0.00;
	// 	setting.radius = 1.87;
	// 	setting.target = {41.44, 31.27, 4.31};

	// 	settings["arbegen"] = setting;
	// }

	// parametric functions
	// renderer->controls->yaw = 0.41;
	// renderer->controls->pitch = -0.51;
	// renderer->controls->radius = 13.45;
	// renderer->controls->target = {0.00, 0.00, 0.00};

	// Setting setting = settings["eclepens_morton"];
	// //Setting setting = settings["niederweiden_morton"];
	
	// renderer->controls->yaw = setting.yaw;
	// renderer->controls->pitch = setting.pitch;
	// renderer->controls->radius = setting.radius;
	// renderer->controls->target = setting.target;

	renderer->controls->yaw = 0.53;
	renderer->controls->pitch = -0.68;
	renderer->controls->radius = 2310.47;
	renderer->controls->target = {576.91, 886.62, 10.35};


	// renderer->controls->yaw = -3.03;
	// renderer->controls->pitch = -0.41;
	// renderer->controls->radius = 1.16;
	// renderer->controls->target = {1.41, 8.09, 3.07};



	// auto potreedata = PotreeData::create(setting.path_potree);
	// auto las_encode_444 = ComputeLasData::create(setting.path_las);
	// auto las_standard = LasStandardData::create(setting.path_las);

	auto lasLoaderSparse = make_shared<LasLoaderSparse>(renderer);

	Runtime::lasLoaderSparse = lasLoaderSparse;

	renderer->onFileDrop([lasLoaderSparse, renderer](vector<string> files){

		vector<string> lasfiles;

		for(auto file : files){
			if(iEndsWith(file, "las")){
				lasfiles.push_back(file);
			}
		}

		lasLoaderSparse->add(lasfiles, [renderer](vector<shared_ptr<LasFile>> lasfiles){

			dvec3 boxMin = {Infinity, Infinity, Infinity};
			dvec3 boxMax = {-Infinity, -Infinity, -Infinity};

			for(auto lasfile : lasfiles){
				boxMin = glm::min(boxMin, lasfile->boxMin);
				boxMax = glm::max(boxMax, lasfile->boxMax);
			}

			// zoom to point cloud
			auto size = boxMax - boxMin;
			auto position = (boxMax + boxMin) / 2.0;
			auto radius = glm::length(size) / 1.5;

			renderer->controls->yaw = 0.53;
			renderer->controls->pitch = -0.68;
			renderer->controls->radius = radius;
			renderer->controls->target = position;
		});

		glfwFocusWindow(renderer->window);
	});

	{ // 4-4-4 byte format
		auto computeLoopLas       = new ComputeLoopLas(renderer.get(), lasLoaderSparse);
		auto computeLoopLas2      = new ComputeLoopLas2(renderer.get(), lasLoaderSparse);
		auto computeLoopLasHqs    = new ComputeLoopLasHqs(renderer.get(), lasLoaderSparse);
		// auto computeLoopLasHqsVR  = new ComputeLoopLasHqsVR(renderer.get(), las_encode_444);
		// auto computeCUDALas       = new ComputeLoopLasCUDA(renderer.get(), las_encode_444);
		Runtime::addMethod((Method*)computeLoopLas);
		Runtime::addMethod((Method*)computeLoopLas2);
		Runtime::addMethod((Method*)computeLoopLasHqs);
		// Runtime::addMethod((Method*)computeLoopLasHqsVR);
		// Runtime::addMethod((Method*)computeCUDALas);
	}

	//{ // POTREE FORMAT
	//	auto computeLoopNodes = new ComputeLoopNodes(renderer.get(), potreedata);
	//	auto computeLoopNodesHqs = new ComputeLoopNodesHqs(renderer.get(), potreedata);
	//	auto computeLoopNodesHqsVr = new ComputeLoopNodesHqsVr(renderer.get(), potreedata);
	//	Runtime::addMethod((Method*)computeLoopNodes);
	//	Runtime::addMethod((Method*)computeLoopNodesHqs);
	//	Runtime::addMethod((Method*)computeLoopNodesHqsVr);
	//}

	//{ // OLD METHODS / 16 byte format
	//	//auto computeEarlyZ = new ComputeEarlyZ(renderer.get(), las_standard);
	//	//auto computeEarlyZReduce = new ComputeEarlyZReduce(renderer.get(), las_standard);
	//	auto computeDedup = new ComputeDedup(renderer.get(), las_standard);
	//	auto compute2021Hqs = new Compute2021HQS(renderer.get(), las_standard);
	//	auto compute2021GL = new ComputeGL(renderer.get(), las_standard);
	//	//Runtime::addMethod((Method*)computeEarlyZ);
	//	//Runtime::addMethod((Method*)computeEarlyZReduce);
	//	Runtime::addMethod((Method*)computeDedup);
	//	Runtime::addMethod((Method*)compute2021Hqs);
	//	Runtime::addMethod((Method*)compute2021GL);
	//}

	// { // PARAMETRIC
	// auto computeParametric = new ComputeParametric(renderer.get());
	// Runtime::addMethod((Method*)computeParametric);
	// }


	auto update = [&](){

		//if(Debug::requestResetView){
		//	renderer->controls->yaw = setting.yaw;
		//	renderer->controls->pitch = setting.pitch;
		//	renderer->controls->radius = setting.radius;
		//	renderer->controls->target = setting.target;

		//	Debug::requestResetView = false;
		//}

		lasLoaderSparse->process();

		auto selected = Runtime::getSelectedMethod();
		if(selected){

			bool needsVR = false;
			needsVR = needsVR || selected->name == "loop_las_hqs_vr";
			needsVR = needsVR || selected->name == "loop_nodes_hqs_vr";
			if(needsVR){
				renderer->setVR(true);
			}else{
				renderer->setVR(false);
			}
			

			selected->update(renderer.get());
		}
		
		if(Runtime::resource){

			string state = "";
			if(Runtime::resource->state == ResourceState::LOADED){
				state = "LOADED";
			}else if(Runtime::resource->state == ResourceState::LOADING){
				state = "LOADING";
			}else if(Runtime::resource->state == ResourceState::UNLOADED){
				state = "UNLOADED";
			}else if(Runtime::resource->state == ResourceState::UNLOADING){
				state = "UNLOADING";
			}
			

			Debug::set("state", state);
		}

		for(auto lasfile : Runtime::lasLoaderSparse->files){
			if(lasfile->isDoubleClicked){

				auto size = lasfile->boxMax - lasfile->boxMin;
				auto position = (lasfile->boxMax + lasfile->boxMin) / 2.0;
				auto radius = glm::length(size) / 1.5;

				renderer->controls->yaw = 0.53;
				renderer->controls->pitch = -0.68;
				renderer->controls->radius = radius;
				renderer->controls->target = position;
			}
		}

		// renderer->drawBoundingBox({0.0, 0.0, 0.0}, {200.0, 200.0, 200.0}, {200, 0, 0});

	};

	auto render = [&](){
		
		glBindFramebuffer(GL_FRAMEBUFFER, 0);

		glClearColor(0.0, 0.0, 0.0, 1.0);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


		if(renderer->vrEnabled){

			auto ovr = OpenVRHelper::instance();


			auto flip = glm::dmat4(
				1.0, 0.0, 0.0, 0.0,
				0.0, 0.0, 1.0, 0.0,
				0.0, -1.0, 0.0, 0.0,
				0.0, 0.0, 0.0, 1.0
			);

			auto& viewLeft = renderer->views[0];
			auto& viewRight = renderer->views[1];

			


			if(!Debug::dummyVR){
				auto size = ovr->getRecommmendedRenderTargetSize();
				viewLeft.framebuffer->setSize(size[0], size[1]);
				viewRight.framebuffer->setSize(size[0], size[1]);

				auto poseHMD = ovr->hmdPose;
				auto poseLeft = ovr->getEyePose(vr::Hmd_Eye::Eye_Left);
				auto poseRight = ovr->getEyePose(vr::Hmd_Eye::Eye_Right);

				viewLeft.view = glm::inverse(flip * poseHMD * poseLeft);
				viewLeft.proj = ovr->getProjection(vr::Hmd_Eye::Eye_Left, 0.01, 10'000.0);

				viewRight.view = glm::inverse(flip * poseHMD * poseRight);
				viewRight.proj = ovr->getProjection(vr::Hmd_Eye::Eye_Right, 0.01, 10'000.0);
			}else{
				ivec2 size = {2468, 2740};
				viewLeft.framebuffer->setSize(size[0], size[1]);
				viewRight.framebuffer->setSize(size[0], size[1]);
			}

			//viewLeft.framebuffer->setSize(1440, 1600);
			glBindFramebuffer(GL_FRAMEBUFFER, viewLeft.framebuffer->handle);
			glClearColor(0.8, 0.2, 0.3, 1.0);
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

			//viewRight.framebuffer->setSize(1440, 1600);
			glBindFramebuffer(GL_FRAMEBUFFER, viewRight.framebuffer->handle);
			glClearColor(0.0, 0.8, 0.3, 1.0);
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		}else{

			auto& view = renderer->views[0];

			view.view = renderer->camera->view;
			view.proj = renderer->camera->proj;

			renderer->views[0].framebuffer->setSize(renderer->width, renderer->height);

			glBindFramebuffer(GL_FRAMEBUFFER, view.framebuffer->handle);
			glClearColor(0.0, 0.2, 0.3, 1.0);
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		}

		{
			glBindFramebuffer(GL_FRAMEBUFFER, 0);

			auto selected = Runtime::getSelectedMethod();
			if(selected){
				selected->render(renderer.get());
			}
		}

		
		for(auto lasfile : Runtime::lasLoaderSparse->files){

			dvec3 size = lasfile->boxMax - lasfile->boxMin;
			dvec3 position = (lasfile->boxMax + lasfile->boxMin) / 2.0;

			if(lasfile->isHovered){
				renderer->drawBoundingBox(position, size, {255, 255, 0});
			}else if(lasfile->numPointsLoaded < lasfile->numPoints){
				renderer->drawBoundingBox(position, size, {255, 0, 0});
			}
			
		}
		

	};

	renderer->loop(update, render);

	return 0;
}

