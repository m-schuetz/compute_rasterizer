
#pragma once

#include <string>
#include <execution>
#include <algorithm>
// #include <stack>

#include "nlohmann/json.hpp"

#include "glm/common.hpp"
#include "glm/matrix.hpp"
#include "glm/vec3.hpp"
#include <glm/gtx/transform.hpp>
#include "unsuck.hpp"
#include "Shader.h"
#include "Resources.h"

using namespace std;
using glm::vec3;
using nlohmann::json;

struct PotreeData : public Resource {

#define STEPS_30BIT 1073741824
#define MASK_30BIT 1073741823
#define STEPS_20BIT 1048576
#define MASK_20BIT 1048575
#define STEPS_10BIT 1024
#define MASK_10BIT 1023

	struct LoaderTask{
		shared_ptr<Buffer> buffer = nullptr;
		shared_ptr<Buffer> buffer_12b = nullptr;
		shared_ptr<Buffer> buffer_8b = nullptr;
		shared_ptr<Buffer> buffer_4b = nullptr;
		shared_ptr<Buffer> buffer_colors = nullptr;
		int64_t pointOffset = 0;
		int64_t numPoints = 0;
	};

	struct Node{
		string name = "";
		Box boundingBox;
		int nodeType = 0;
		int64_t numPoints = 0;
		int64_t byteOffset = 0;
		int64_t byteSize = 0;
		int64_t hierarchyByteOffset = 0;
		int64_t hierarchyByteSize = 0;
		float spacing = 1.0;
		int level = 0;
		int loadIndex = 0;

		shared_ptr<Node> parent = nullptr;

		shared_ptr<Node> children[8] = {
			nullptr, nullptr, nullptr, nullptr, 
			nullptr, nullptr, nullptr, nullptr};

	};

	struct Bin{
		vector<Node> nodes;
		float weight = 100.0;
		int64_t firstPoint = 2'000'000'000;
		int64_t numPoints = 0;
	};

	void traverse(Node* node, std::function<void(Node*)> callback) {

		callback(node);

		for (auto child : node->children) {
			if (child) {
				traverse(child.get(), callback);
			}
		}

	}

	string path = "";
	shared_ptr<LoaderTask> task = nullptr;
	mutex mtx_state;

	bool metadataLoaded = false;
	int64_t numPoints = 0;
	int64_t numPointsLoaded = 0;
	dvec3 scale = {1.0, 1.0, 1.0};
	dvec3 offset = {0.0, 0.0, 0.0};
	dvec3 boxMin;
	dvec3 boxMax;
	float spacing = 1.0;
	int64_t firstHierarchyChunkSize = 0;
	int64_t bytesPerPoint = 0;
	int64_t rgbOffset = 0;

	GLBuffer ssBatches;
	GLBuffer ssXyz_12b;
	GLBuffer ssXyz_8b;
	GLBuffer ssXyz_4b;
	GLBuffer ssColors;
	GLBuffer ssSelection;

	shared_ptr<Node> root = nullptr;
	vector<Node> nodes;
	vector<shared_ptr<Bin>> bins;

	PotreeData(){

	}

	Box createChildAABB(Box aabb, int index){
		vec3 min = aabb.min;
		vec3 max = aabb.max;
		vec3 size = max - min;

		if ((index & 0b0001) > 0) {
			min.z += size.z / 2;
		} else {
			max.z -= size.z / 2;
		}

		if ((index & 0b0010) > 0) {
			min.y += size.y / 2;
		} else {
			max.y -= size.y / 2;
		}
		
		if ((index & 0b0100) > 0) {
			min.x += size.x / 2;
		} else {
			max.x -= size.x / 2;
		}

		Box box;
		box.min = min;
		box.max = max;

		return box;
	}


	void loadMetadata(){
		auto strMetadata = readTextFile(path + "/metadata.json");
		auto jsMetadata = json::parse(strMetadata);

		boxMin.x = jsMetadata["boundingBox"]["min"][0];
		boxMin.y = jsMetadata["boundingBox"]["min"][1];
		boxMin.z = jsMetadata["boundingBox"]["min"][2];
		boxMax.x = jsMetadata["boundingBox"]["max"][0];
		boxMax.y = jsMetadata["boundingBox"]["max"][1];
		boxMax.z = jsMetadata["boundingBox"]["max"][2];

		scale.x = jsMetadata["scale"][0];
		scale.y = jsMetadata["scale"][1];
		scale.z = jsMetadata["scale"][2];

		offset.x = jsMetadata["offset"][0];
		offset.y = jsMetadata["offset"][1];
		offset.z = jsMetadata["offset"][2];

		// numPoints = jsMetadata["points"];
		numPoints = jsMetadata["points"];
		//numPoints = std::min(numPoints, 1'000'000'000ll);
		numPointsLoaded = numPoints;
		spacing = jsMetadata["spacing"];

		firstHierarchyChunkSize = jsMetadata["hierarchy"]["firstChunkSize"];

		bytesPerPoint = 0;
		auto jsAttributes = jsMetadata["attributes"];
		for(int i = 0; i < jsAttributes.size(); i++){
			auto jsAttribute = jsAttributes[i];

			string name = jsAttribute["name"];

			if(name == "rgb"){
				rgbOffset = bytesPerPoint;
			}

			bytesPerPoint += jsAttribute["size"];
		}

		metadataLoaded = true;
	}

	void parseHierarchy(shared_ptr<Node> node, shared_ptr<Buffer> buffer){

		int bytesPerNode = 22;

		int numNodes = node->hierarchyByteSize / bytesPerNode;

		static int loadIndex = 0;

		vector<shared_ptr<Node>> nodes(numNodes);
		nodes[0] = node;
		int nodePos = 1;

		for(int i = 0; i < numNodes; i++){
			auto current = nodes[i];

			int type           = buffer->get< uint8_t>(node->hierarchyByteOffset + i * bytesPerNode + 0);
			int childMask      = buffer->get< uint8_t>(node->hierarchyByteOffset + i * bytesPerNode + 1);
			int numPoints      = buffer->get<uint32_t>(node->hierarchyByteOffset + i * bytesPerNode + 2);
			int64_t byteOffset = buffer->get< int64_t>(node->hierarchyByteOffset + i * bytesPerNode + 6);
			int64_t byteSize   = buffer->get< int64_t>(node->hierarchyByteOffset + i * bytesPerNode + 14);

			if(current->nodeType == 2){
				current->byteOffset = byteOffset;
				current->byteSize = byteSize;
				current->numPoints = numPoints;
			}else if(type == 2){
				current->hierarchyByteOffset = byteOffset;
				current->hierarchyByteSize = byteSize;
				current->numPoints = numPoints;
			}else{
				current->byteOffset = byteOffset;
				current->byteSize = byteSize;
				current->numPoints = numPoints;
			}

			current->nodeType = type;

			if(current->nodeType == 2){
				continue;
			}

			for(int childIndex = 0; childIndex < 8; childIndex++){
				bool childExists = ((1 << childIndex) & childMask) != 0;

				if(!childExists){
					continue;
				};

				auto childAABB = createChildAABB(current->boundingBox, childIndex);
				auto child = make_shared<Node>();
				child->name = current->name + to_string(childIndex);
				child->spacing = current->spacing / 2;
				child->level = current->level + 1;
				child->boundingBox = childAABB;
				child->loadIndex = loadIndex;
				loadIndex++;

				current->children[childIndex] = child;
				child->parent = current;

				nodes[nodePos] = child;
				nodePos++;
			}
		}

		for(auto node : nodes){
			if(node->nodeType == 2){
				parseHierarchy(node, buffer);
			}
		}

	}

	void loadHierarchy(){

		auto buffer = readBinaryFile(path + "/hierarchy.bin");

		auto root = make_shared<Node>();
		root->name = "r";
		root->nodeType = 2;
		root->hierarchyByteOffset = 0;
		root->hierarchyByteSize = firstHierarchyChunkSize;
		root->spacing = spacing;
		// root->boundingBox = Box(boxMin, boxMax);
		root->boundingBox.min = {0.0, 0.0, 0.0};
		root->boundingBox.max = boxMax - boxMin;


		parseHierarchy(root, buffer);

		vector<Node> nodes;

		traverse(root.get(), [&nodes](Node* node) {
			nodes.push_back(*node);
		});

		std::sort(std::execution::par_unseq, nodes.begin(), nodes.end(), [](Node a, Node b) {
			return a.byteOffset < b.byteOffset;
		});

		shared_ptr<Bin> currentBin = make_shared<Bin>();
		vector<shared_ptr<Bin>> bins;

		for(auto node : nodes){

			currentBin->nodes.push_back(node);
			currentBin->firstPoint = std::min(currentBin->firstPoint, int64_t(node.byteOffset / this->bytesPerPoint));
			currentBin->numPoints += node.numPoints;
			currentBin->weight = std::min(float(node.level), currentBin->weight);

			if(currentBin->numPoints > 1'000'000){
				bins.push_back(currentBin);

				currentBin = make_shared<Bin>();
			}

		}
		if(currentBin->numPoints > 0){
			bins.push_back(currentBin);
		}

		std::sort(bins.begin(), bins.end(), [](shared_ptr<Bin> a, shared_ptr<Bin> b) {
			return a->weight < b->weight;
		});

		this->nodes = nodes;
		this->root = root;
		this->bins = bins;

	}

	static shared_ptr<PotreeData> create(string path){
		auto data = make_shared<PotreeData>();
		data->path = path;
		data->loadMetadata();
		data->loadHierarchy();

		return data;
	}

	void load(Renderer* renderer){

		cout << "PotreeData::load()" << endl;

		{
			lock_guard<mutex> lock(mtx_state);

			if(state != ResourceState::UNLOADED){
				return;
			}else{
				state = ResourceState::LOADING;
			}
		}

		this->ssBatches = renderer->createBuffer(64 * this->nodes.size());
		this->ssXyz_12b = renderer->createBuffer(4 * this->numPoints);
		this->ssXyz_8b = renderer->createBuffer(4 * this->numPoints);
		this->ssXyz_4b = renderer->createBuffer(4 * this->numPoints);
		this->ssColors = renderer->createBuffer(4 * this->numPoints);
		this->ssSelection = renderer->createBuffer(256);
		// this->ssSelection = renderer->createBuffer(4 * this->numPoints);

		GLuint zero = 0;
		// glClearNamedBufferData(this->ssBatches.handle, GL_R32UI, GL_RED, GL_UNSIGNED_INT, &zero);
		glClearNamedBufferData(this->ssXyz_12b.handle, GL_R32UI, GL_RED, GL_UNSIGNED_INT, &zero);
		glClearNamedBufferData(this->ssXyz_8b.handle, GL_R32UI, GL_RED, GL_UNSIGNED_INT, &zero);
		glClearNamedBufferData(this->ssXyz_4b.handle, GL_R32UI, GL_RED, GL_UNSIGNED_INT, &zero);
		glClearNamedBufferData(this->ssColors.handle, GL_R32UI, GL_RED, GL_UNSIGNED_INT, &zero);
		glClearNamedBufferData(this->ssSelection.handle, GL_R32UI, GL_RED, GL_UNSIGNED_INT, &zero);

		{ // create batch buffer
			Buffer buffer(64 * this->nodes.size());

			for(int i = 0; i < this->nodes.size(); i++){
				Node node = this->nodes[i];

				int32_t pointOffset = node.byteOffset / this->bytesPerPoint;

				buffer.set<float>(node.boundingBox.min.x, 64 * i +  4);
				buffer.set<float>(node.boundingBox.min.y, 64 * i +  8);
				buffer.set<float>(node.boundingBox.min.z, 64 * i + 12);
				buffer.set<float>(node.boundingBox.max.x, 64 * i + 16);
				buffer.set<float>(node.boundingBox.max.y, 64 * i + 20);
				buffer.set<float>(node.boundingBox.max.z, 64 * i + 24);
				buffer.set<int32_t>(node.numPoints, 64 * i + 28);
				buffer.set<int32_t>(pointOffset, 64 * i + 32);
				buffer.set<int32_t>(node.level, 64 * i + 36);
			}

			glNamedBufferSubData(ssBatches.handle, 0, buffer.size, buffer.data);
		}

		PotreeData *ref = this;
		thread t([ref](){

			//for (auto bin : ref->bins) {
			//	static int numPointsInBin = 0;
			//	numPointsInBin += bin->numPoints;
			//	Debug::set("numPointsInBin", formatNumber(numPointsInBin));
			//}

			Debug::set("#bins", formatNumber(ref->bins.size()));

			//for(auto bin : ref->bins)
			for(int j = 0; j < ref->bins.size(); j++)
			{
				auto bin = ref->bins[j];

				{ // abort loader thread if state is set to unloading
					lock_guard<mutex> lock(ref->mtx_state);

					if(ref->state == ResourceState::UNLOADING){
						cout << "stopping loader thread for " << ref->path << endl;

						ref->state = ResourceState::UNLOADED;

						return;
					}
				}

				if(ref->task){
					this_thread::sleep_for(1ms);
					j--;
					continue;
				}

				int pointsInBatch = bin->numPoints;

				static int binsLoaded = 0;
				static int numPointsLoaded = 0;
				numPointsLoaded += pointsInBatch;
				binsLoaded++;
				Debug::set("numPointsLoaded", formatNumber(numPointsLoaded));
				Debug::set("binsLoaded", formatNumber(binsLoaded));

				int64_t start = int64_t(ref->bytesPerPoint) * int64_t(bin->firstPoint);
				int64_t size = ref->bytesPerPoint * pointsInBatch;
				auto source = readBinaryFile(ref->path + "/octree.bin", start, size);
				auto target = make_shared<Buffer>(16 * pointsInBatch);
				auto bufXyz_12b = make_shared<Buffer>(4 * pointsInBatch);
				auto bufXyz_8b = make_shared<Buffer>(4 * pointsInBatch);
				auto bufXyz_4b = make_shared<Buffer>(4 * pointsInBatch);
				auto bufColors = make_shared<Buffer>(4 * pointsInBatch);

				int pointsProcesed = 0;
				for (int nodeIndex = 0; nodeIndex < bin->nodes.size(); nodeIndex++) {
					auto node = bin->nodes[nodeIndex];
					auto boxSize = node.boundingBox.size();
					auto wgMin = node.boundingBox.min;
					auto wgMax = node.boundingBox.max;

					for (int pointIndex = 0; pointIndex < node.numPoints; pointIndex++) {

						int32_t pointOffset = pointsProcesed * ref->bytesPerPoint;

						int32_t X = source->get<int32_t>(pointOffset + 0);
						int32_t Y = source->get<int32_t>(pointOffset + 4);
						int32_t Z = source->get<int32_t>(pointOffset + 8);

						double x = double(X) * ref->scale.x + ref->offset.x - ref->boxMin.x;
						double y = double(Y) * ref->scale.y + ref->offset.y - ref->boxMin.y;
						double z = double(Z) * ref->scale.z + ref->offset.z - ref->boxMin.z;

						int32_t R = source->get<uint16_t>(pointOffset + ref->rgbOffset + 0);
						int32_t G = source->get<uint16_t>(pointOffset + ref->rgbOffset + 2);
						int32_t B = source->get<uint16_t>(pointOffset + ref->rgbOffset + 4);

						uint8_t r = R > 255 ? R / 256 : R;
						uint8_t g = G > 255 ? G / 256 : G;
						uint8_t b = B > 255 ? B / 256 : B;

						target->set<  float>(  x, 16 * pointsProcesed +  0);
						target->set<  float>(  y, 16 * pointsProcesed +  4);
						target->set<  float>(  z, 16 * pointsProcesed +  8);
						target->set<uint8_t>(  r, 16 * pointsProcesed + 12);
						target->set<uint8_t>(  g, 16 * pointsProcesed + 13);
						target->set<uint8_t>(  b, 16 * pointsProcesed + 14);
						target->set<uint8_t>(  0, 16 * pointsProcesed + 15);



						{ // encoded

							bufColors->set<uint8_t>(  r, 4 * pointsProcesed + 0);
							bufColors->set<uint8_t>(  g, 4 * pointsProcesed + 1);
							bufColors->set<uint8_t>(  b, 4 * pointsProcesed + 2);
							bufColors->set<uint8_t>(  0, 4 * pointsProcesed + 3);

							uint32_t X = uint32_t(((x - wgMin.x) / boxSize.x) * STEPS_30BIT) & MASK_30BIT;
							uint32_t Y = uint32_t(((y - wgMin.y) / boxSize.y) * STEPS_30BIT) & MASK_30BIT;
							uint32_t Z = uint32_t(((z - wgMin.z) / boxSize.z) * STEPS_30BIT) & MASK_30BIT;

							{ // 4 byte

								uint32_t X_4b = (X >> 20) & MASK_10BIT;
								uint32_t Y_4b = (Y >> 20) & MASK_10BIT;
								uint32_t Z_4b = (Z >> 20) & MASK_10BIT;

								uint32_t encoded = X_4b | (Y_4b << 10) | (Z_4b << 20);

								bufXyz_4b->set<uint32_t>(encoded, 4 * pointsProcesed);
							}

							{ // 8 byte

								uint32_t X_8b = (X >> 10) & MASK_10BIT;
								uint32_t Y_8b = (Y >> 10) & MASK_10BIT;
								uint32_t Z_8b = (Z >> 10) & MASK_10BIT;

								uint32_t encoded = X_8b | (Y_8b << 10) | (Z_8b << 20);

								bufXyz_8b->set<uint32_t>(encoded, 4 * pointsProcesed);
							}
							
							{ // 12 byte

								uint32_t X_12b = (X >> 10) & MASK_10BIT;
								uint32_t Y_12b = (Y >> 10) & MASK_10BIT;
								uint32_t Z_12b = (Z >> 10) & MASK_10BIT;

								uint32_t encoded = X_12b | (Y_12b << 10) | (Z_12b << 20);

								bufXyz_12b->set<uint32_t>(encoded, 4 * pointsProcesed);
							}

						}


						pointsProcesed++;
					}

				}

				auto task = make_shared<LoaderTask>();
				task->buffer = target;
				task->buffer_12b = bufXyz_12b;
				task->buffer_8b = bufXyz_8b;
				task->buffer_4b = bufXyz_4b;
				task->buffer_colors = bufColors;
				task->pointOffset = bin->firstPoint;
				task->numPoints = pointsInBatch;

				ref->task = task;

			}

			//cout << "finished loading " << formatNumber(pointsRead) << " points" << endl;

			{ // check if resource was marked as unloading in the meantime
				lock_guard<mutex> lock(ref->mtx_state);

				if(ref->state == ResourceState::UNLOADING){
					cout << "stopping loader thread for " << ref->path << endl;

					ref->state = ResourceState::UNLOADED;
				}else if(ref->state == ResourceState::LOADING){
					ref->state = ResourceState::LOADED;
				}
			}

		});
		t.detach();
		
	}

	void unload(Renderer* renderer){

		cout << "PotreeData::unload()" << endl;

		numPointsLoaded = 0;

		glDeleteBuffers(1, &ssXyz_12b.handle);
		glDeleteBuffers(1, &ssXyz_8b.handle);
		glDeleteBuffers(1, &ssXyz_4b.handle);
		glDeleteBuffers(1, &ssColors.handle);
		glDeleteBuffers(1, &ssBatches.handle);

		lock_guard<mutex> lock(mtx_state);

		if(state == ResourceState::LOADED){
			state = ResourceState::UNLOADED;
		}else if(state == ResourceState::LOADING){
			// if loader thread is still running, notify thread by marking resource as "unloading"
			state = ResourceState::UNLOADING;
		}
	}

	void process(Renderer* renderer){

		if(this->task){
			auto task = this->task;

			glNamedBufferSubData(this->ssXyz_12b.handle, 
				4 * task->pointOffset, 
				task->buffer_12b->size, 
				task->buffer_12b->data);

			glNamedBufferSubData(this->ssXyz_8b.handle, 
				4 * task->pointOffset, 
				task->buffer_8b->size, 
				task->buffer_12b->data);

			glNamedBufferSubData(this->ssXyz_4b.handle, 
				4 * task->pointOffset, 
				task->buffer_4b->size, 
				task->buffer_4b->data);

			glNamedBufferSubData(this->ssColors.handle,
				4 * task->pointOffset,
				task->buffer_colors->size,
				task->buffer_colors->data);

			// this->numPointsLoaded += task->numPoints;
			this->task = nullptr;

		}

	}

};