
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

using namespace std;
using namespace std::chrono_literals;

struct PointSource;

mutex mtx_tree;

#define MAX_LOADED_POINTS 50'000'000
#define NODE_CAPACITY 500'000
#define MAX_POINTS_PER_BATCH 1'000'000

struct GlBuffer{
	GLuint vao = 0;
	GLuint vbo = 0;
	int size = 0;
};

struct Batch {
	uint64_t firstPoint = 0;
	uint64_t numPoints = 0;
	shared_ptr<Buffer> buffer = nullptr;
	shared_ptr<PointSource> source;

	GLuint vao = 0;
	GLuint vbo = 0;

	GLuint vao_coarse = 0;
	GLuint vbo_coarse = 0;
};

struct PointSource{
	Box boundingBox;
	string path = "";
	int64_t numPoints = 0;

	deque<shared_ptr<Batch>> batchQueue;

	float priority = 0.0;
};



struct Point {
	double x;
	double y;
	double z;
	uint8_t r;
	uint8_t g;
	uint8_t b;
	uint8_t a;
	uint32_t padding;
};

struct SampleGrid{
	Box box;
	dvec3 size;
	int gridSize = 128;
	int fGridSize = 128;
	int* grid = nullptr;
	
	SampleGrid(Box box){
		this->box = box;
		this->size = box.size();

		grid = reinterpret_cast<int*>(malloc(4 * gridSize * gridSize * gridSize));
		memset(grid, 0, 4 * gridSize * gridSize * gridSize);
	}

	bool add(Point point){
		
		int X = clamp(fGridSize * (point.x - box.min.x) / size.x, 0.0, fGridSize - 1.0);
		int Y = clamp(fGridSize * (point.y - box.min.y) / size.y, 0.0, fGridSize - 1.0);
		int Z = clamp(fGridSize * (point.z - box.min.z) / size.z, 0.0, fGridSize - 1.0);

		int index = X + gridSize * Y + gridSize * gridSize * Z;

		grid[index]++;

		if(grid[index] == 1){
			return true;
		}else{
			return false;
		}


	}

};

struct Node {

	string name = "";
	uint64_t numPoints = 0;
	int index = 0;
	Box boundingBox;
	Node* children[8] = { nullptr , nullptr , nullptr , nullptr , nullptr , nullptr , nullptr , nullptr };
	vector<shared_ptr<Buffer>> points;


	shared_ptr<SampleGrid> lodgrid;
	vector<Point> lod;
	GlBuffer lod_glbuffer;

	Node(string name, Box boundingBox) {

		this->name = name;
		this->boundingBox = boundingBox;

		this->lodgrid = make_shared<SampleGrid>(boundingBox);
		
	}

	void traverse(function<void(Node*)> callback) {

		callback(this);

		for (auto child : children) {
			if (child != nullptr) {
				child->traverse(callback);
			}
		}

	}
};

struct Node_renderable{
	Node* node = nullptr;
	vector<shared_ptr<Buffer>> points;
	double priority = 0.0;
};

struct LOD_renderable{
	vector<Node_renderable> nodes;
};



inline Box childBoundingBoxOf(dvec3 min, dvec3 max, int index) {
	Box box;
	auto size = max - min;
	dvec3 center = min + (size * 0.5);

	if ((index & 0b100) == 0) {
		box.min.x = min.x;
		box.max.x = center.x;
	} else {
		box.min.x = center.x;
		box.max.x = max.x;
	}

	if ((index & 0b010) == 0) {
		box.min.y = min.y;
		box.max.y = center.y;
	} else {
		box.min.y = center.y;
		box.max.y = max.y;
	}

	if ((index & 0b001) == 0) {
		box.min.z = min.z;
		box.max.z = center.z;
	} else {
		box.min.z = center.z;
		box.max.z = max.z;
	}

	return box;
}

void addPointsToTree(Node* node, shared_ptr<Buffer> points);

void passPoints(Node* node, shared_ptr<Buffer> points){

	auto min = node->boundingBox.min;
	auto max = node->boundingBox.max;
	auto size = max - min;

	float minX = min.x;
	float minY = min.y;
	float minZ = min.z;
	float sizeX = size.x;
	float sizeY = size.y;
	float sizeZ = size.z;

	int numPoints = points->size / 32;
	int counters[8] = {0, 0, 0, 0, 0, 0, 0, 0};
	Point point;
	for(int i = 0; i < numPoints; i++){
		memcpy(&point, points->data_u8 + 32 * i, 32);

		int X = ((point.x - minX) / sizeX) > 0.5 ? 1 : 0;
		int Y = ((point.y - minY) / sizeY) > 0.5 ? 1 : 0;
		int Z = ((point.z - minZ) / sizeZ) > 0.5 ? 1 : 0;

		int index = (X << 2) | (Y << 1) | Z;

		counters[index]++;
	}

	shared_ptr<Buffer> buffers[8];
	for(int i = 0; i < 8; i++){
		int numChildPoints = counters[i];
		buffers[i] = make_shared<Buffer>(32 * numChildPoints);
	}

	for(int i = 0; i < numPoints; i++){
		memcpy(&point, points->data_u8 + 32 * i, 32);

		int X = ((point.x - minX) / sizeX) > 0.5 ? 1 : 0;
		int Y = ((point.y - minY) / sizeY) > 0.5 ? 1 : 0;
		int Z = ((point.z - minZ) / sizeZ) > 0.5 ? 1 : 0;

		int index = (X << 2) | (Y << 1) | Z;

		buffers[index]->write(&point, 32);
	}

	for(int childIndex = 0; childIndex < 8; childIndex++){
		auto points = buffers[childIndex];

		if(points->size > 0){
			if(node->children[childIndex] == nullptr){
				string name = node->name + to_string(childIndex);
				Box boundingBox = childBoundingBoxOf(min, max, childIndex);
				Node* child = new Node(name, boundingBox);
				
				//cout << "created " << child->name << endl;
				
				node->children[childIndex] = child;
			}

			addPointsToTree(node->children[childIndex], points);
		}

	}

}

void split(Node* node){
	
	for(auto points : node->points){
		passPoints(node, points);
	}

	node->points = vector<shared_ptr<Buffer>>();

}

void addPointsToTree(Node* node, shared_ptr<Buffer> points){

	int numNewPoints = points->size / 32;

	if(node->numPoints > NODE_CAPACITY){
		// PASS
		passPoints(node, points);
	}else if(node->numPoints <= NODE_CAPACITY && node->numPoints + numNewPoints > NODE_CAPACITY){
		// SPLIT
		node->points.push_back(points);
		split(node);
	}else{
		// APPEND
		node->points.push_back(points);
	}

	//if(false)
	{ // add to lod
		
		Point point;
		for(int i = 0; i < points->size / 32; i++){
			memcpy(&point, points->data_u8 + 32 * i, 32);

			bool success = node->lodgrid->add(point);

			if(success){
				node->lod.push_back(point);
			}
		}
	
	}

	node->numPoints += numNewPoints;
}

mutex mtx_construction;
mutex mtx_construction_queue;
mutex mtx_lod_renderable;
mutex mtx_loadTask;

struct SimLOD{

	vector<shared_ptr<PointSource>> sources;

	shared_ptr<PointSource> loadTask = nullptr;

	int64_t numLoaded = 0;

	Node* root = nullptr;
	deque<shared_ptr<Buffer>> construction_queue;
	

	shared_ptr<LOD_renderable> lod_renderable;
	shared_ptr<Camera> camera = nullptr;


	SimLOD(){

		
		if(sizeof(Point) != 32){
			exit(123);	
		}


		lod_renderable = make_shared<LOD_renderable>();

		// update lod_renderable
		thread t_visibility([this](){

			while(true){

				std::this_thread::sleep_for(10ms);

				if(!Debug::updateEnabled){
					continue;
				}

				mtx_construction.lock();

				vector<Node_renderable> renderableNodes;

				if(root == nullptr || camera == nullptr){
					mtx_construction.unlock();
					continue;
				}

				dmat4 viewProj = camera->proj * camera->view;

				Frustum frustum;
				frustum.set(viewProj);

				// update LOD_renderable
				auto camera = this->camera;
				root->traverse([&renderableNodes, camera, &frustum](Node* node) {
					
					auto box = node->boundingBox;
					auto center = box.center();

					bool visible = frustum.intersectsBox(box);

					
					double radius = glm::length(box.size()) / 2.0;
					double distance = glm::length(camera->position - center);

					auto worldView = camera->proj * camera->view;
					dvec4 projPos = worldView * dvec4(center, 1.0);
					dvec4 ndc = projPos / projPos.w;
					double d = glm::length(glm::dvec2(ndc.x, ndc.y));

					double P_distance = clamp(5.0 * radius / distance, 0.0, 1.0);
					double P_screen = clamp(std::exp(-(d * d)), 0.0, 1.0);
					double priority = P_distance + P_screen;

					//priority = (P_distance + P_screen) / 2.0;
					priority = P_distance;

					//if (distance < radius) {
					//	priority = 1.0;
					//}

					if(priority > 0.4 && visible){
						Node_renderable renderable;
						renderable.node = node;
						renderable.points = node->points;
						renderable.priority = priority;

						renderableNodes.push_back(renderable);
					}
					

					
				});

				auto lod = make_shared<LOD_renderable>();
				lod->nodes = renderableNodes;

				mtx_construction.unlock();

				// swap LOD_renderable
				mtx_lod_renderable.lock();
				this->lod_renderable = lod;
				mtx_lod_renderable.unlock();

			}

		});
		t_visibility.detach();

		thread t_loading([this](){
			
			while(true){

				std::this_thread::sleep_for(5ms);

				shared_ptr<Batch> batch = nullptr;

				{
					lock_guard<mutex> lock_0(mtx_loadTask);
					lock_guard<mutex> lock_1(mtx_construction_queue);

					if(this->loadTask != nullptr && construction_queue.size() < 5){
						batch = this->loadTask->batchQueue.front();
						this->loadTask->batchQueue.pop_front();
						this->loadTask = nullptr;
					}
				}

				if(batch != nullptr){
					auto path = fs::path(batch->source->path);
					

					cout << "loading " << path.filename() << ", " << batch->firstPoint << endl;
					auto laspoints = LasLoader::loadSync(batch->source->path, batch->firstPoint, batch->numPoints);
					cout << "    loaded " << path.filename() << ", " << batch->firstPoint << endl;

					lock_guard<mutex> lock(mtx_construction_queue);

					construction_queue.push_back(laspoints.buffer);
				}

			}
			

		});
		t_loading.detach();

		thread t_construction([this](){
			
			while (true) {
				std::this_thread::sleep_for(10ms);

				shared_ptr<Buffer> buffer = nullptr;

				{
					lock_guard<mutex> lock(mtx_construction_queue);

					if (!construction_queue.empty()) {
						buffer = construction_queue.front();
						construction_queue.pop_front();
					}
				}
				

				if(buffer != nullptr){
					lock_guard<mutex> lock(mtx_construction);

					int numPoints = buffer->size / 32;

					auto tStart = now();
					addPointsToTree(root, buffer);

					printElapsedTime("addPointsToTree", tStart);
				}


				
			}

		});
		t_construction.detach();


	}

	void update(Renderer* renderer){

		auto camera = renderer->camera;

		shared_ptr<PointSource> highest = nullptr;

		auto viewProj = camera->proj * camera->view;
		Frustum frustum;
		frustum.set(viewProj);
		
		for(int i = 0; i < sources.size(); i++){
			auto source = sources[i];

			auto box = source->boundingBox;
			auto center = box.center();


			if(Debug::updateEnabled){
				bool visible = frustum.intersectsBox(box);
				double radius = glm::length(box.size()) / 2.0;
				double distance = glm::length(camera->position - center);

				dvec4 projPos = viewProj * dvec4(center, 1.0);
				dvec4 ndc = projPos / projPos.w;
				double d = glm::length(glm::dvec2(ndc.x, ndc.y));

				double P_distance = clamp(10.0 * radius / distance, 0.0, 1.0);
				double P_screen = clamp(std::exp(-(d * d)), 0.0, 1.0);
				double priority = P_distance * P_screen;

				if (distance < radius) {
					priority = 1.0;
				}

				if(!visible){
					priority = 0.0;
				}

				source->priority = priority;
			}


			if((highest == nullptr || source->priority > highest->priority) && !source->batchQueue.empty()){
				highest = source;
			}
		}

		{ // schedule highest for loading

			mtx_loadTask.lock();

			if(loadTask == nullptr){
				loadTask = highest;
			}

			mtx_loadTask.unlock();
		}

		if(Debug::doCopyTree)
		{
			stringstream ss;
			ss<< std::setprecision(2) << std::fixed;

			this->root->traverse([&ss](Node* node){
				ss << leftPad(node->name, 4 * node->name.size()) << ", " << node->numPoints << ", " << node->points.size() << endl;
			});

			string str = ss.str();

			toClipboard(str);

			Debug::doCopyTree = false;
		}

	}

	void render(Renderer* renderer) {

		this->camera = renderer->camera;

		{
			mtx_lod_renderable.lock();

			static unordered_map<int, GlBuffer> glbuffers;


			int numRenderedNodes = 0;
			int numRenderedPoints_full = 0;
			int numRenderedPoints_lod = 0;

			for(auto node : lod_renderable->nodes){
				auto box = node.node->boundingBox;

				if(Debug::showBoundingBox){

					double w = 255.0 * node.priority;
					int wi = clamp(int(w), 0, 255);
					glm::ivec3 color = { 0, wi, 0 };
					//renderer->drawBoundingBox(box.center(), box.size(), color);
				}

				for(shared_ptr<Buffer> points : node.points){
					
					//Buffer* ptr = points.get();
					if(glbuffers.find(points->id) == glbuffers.end()){

						GlBuffer glbuffer;

						glCreateVertexArrays(1, &glbuffer.vao);
						glCreateBuffers(1, &glbuffer.vbo);
						glNamedBufferData(glbuffer.vbo, points->size, points->data, GL_DYNAMIC_DRAW);

						glbuffer.size = points->size;

						glbuffers[points->id] = glbuffer;	
					}

					auto glbuffer = glbuffers[points->id];

					if(Debug::boolMisc){
						renderer->drawPoints(points->data, points->size / 32);
					}else{
						renderer->drawPoints(glbuffer.vao, glbuffer.vbo, points->size / 32);
					}

					numRenderedPoints_full += points->size / 32;
				}

				if(node.points.size() == 0){
					renderer->drawPoints(node.node->lod.data(), node.node->lod.size());
					numRenderedPoints_lod += node.node->lod.size();
				}

				numRenderedNodes++;
			};

			mtx_lod_renderable.unlock();
			
			Debug::set("#nodes         ", formatNumber(numRenderedNodes));
			Debug::set("#points_full   ", formatNumber(numRenderedPoints_full));
			Debug::set("#points_lod    ", formatNumber(numRenderedPoints_lod));
		}

		for(auto source : sources){
			if(Debug::showBoundingBox){

				auto box = source->boundingBox;
				glm::ivec3 color = { 255, 0, 0 };
				renderer->drawBoundingBox(box.center(), box.size(), color);
			}
		}
		

	}

	void load(vector<string> files){
		for(string file : files){
			load(file);
		}

		Box boundingBox;
		for (auto source : sources) {
			boundingBox.expand(source->boundingBox);
		}

		root = new Node("r", boundingBox.cube());
	}

	void load(string path){

		auto buffer = new Buffer(2 * 1024);
		readBinaryFile(path, 0, buffer->size, buffer->data);

		uint64_t offsetToPointData = buffer->get<uint32_t>(96);
		int format = buffer->get<uint8_t>(104);

		int versionMajor = buffer->get<uint8_t>(24);
		int versionMinor = buffer->get<uint8_t>(25);

		glm::dvec3 min = {
			buffer->get<double>(187),
			buffer->get<double>(203),
			buffer->get<double>(219)};

		glm::dvec3 max = {
			buffer->get<double>(179),
			buffer->get<double>(195),
			buffer->get<double>(211)};

		int64_t numPoints = 0;

		if(versionMajor == 1 && versionMinor <= 3){
			numPoints = buffer->get<uint32_t>(107);
		}else{
			numPoints = buffer->get<int64_t>(247);
		}

		// point 1 colors
		uint16_t R = buffer->get<uint16_t>(offsetToPointData + 28);
		uint16_t G = buffer->get<uint16_t>(offsetToPointData + 30);
		uint16_t B = buffer->get<uint16_t>(offsetToPointData + 32);

		glm::ivec3 color = {
			R / 255,
			G / 255,
			B / 255
		};


		Box box;
		box.min = min;
		box.max = max;
		box.color = color;

		auto source = make_shared<PointSource>();
		deque<shared_ptr<Batch>> batchDescriptors;
		for (int i = 0; i < numPoints; i += MAX_POINTS_PER_BATCH) {

			int batchSize = std::min(int(numPoints - i), MAX_POINTS_PER_BATCH);

			auto batch = make_shared<Batch>();
			batch->firstPoint = i;
			batch->numPoints = batchSize;
			batch->source = source;

			batchDescriptors.push_back(batch);
		}

		source->boundingBox = box;
		source->numPoints = numPoints;
		source->path = path;
		source->batchQueue = batchDescriptors;

		sources.push_back(source);
	
	}

};