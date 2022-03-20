
#pragma once

#include <execution>
#include <limits>


#include "TaskPool.h"

namespace batchwise_multithreaded_2{
	constexpr uint64_t NODE_CAPACITY = 1'000'000;
	constexpr uint64_t MAX_BATCH_SIZE = 1'000'000;
	constexpr int MORTON_LEVELS = 20;
	double MORTON_GRID_SIZE = pow(2, MORTON_LEVELS); // 10: 1024

	mutex mtx_add;

	struct Point {
		double x;
		double y;
		double z;
		uint64_t mortonCode;
	};

	struct Cube{
		glm::dvec3 min = {Infinity, Infinity, Infinity};
		double size = 0.0;
	};

	struct Cube_i{
		glm::ivec3 min = {INT_MAX, INT_MAX, INT_MAX};
		int size = 0;

		bool encompases(Cube_i cube){
			
			bool inside_x = (cube.min.x >= this->min.x) && ((cube.min.x + cube.size) <= (this->min.x + this->size));
			bool inside_y = (cube.min.y >= this->min.y) && ((cube.min.y + cube.size) <= (this->min.y + this->size));
			bool inside_z = (cube.min.z >= this->min.z) && ((cube.min.z + cube.size) <= (this->min.z + this->size));

			if(inside_x && inside_y && inside_z){
				return true;
			}else{
				return false;
			}

		}
	};

	struct Batch{
		shared_ptr<Buffer> buffer;
		int first = 0;
		int size = 0;
		Cube_i cube;
	};

	struct Node {

		string name = "";
		uint64_t numPoints = 0;
		int index = 0;
		int level = 0;
		Box boundingBox;
		Cube cube;
		Cube_i cube_i;

		Node* children[8] = { nullptr , nullptr , nullptr , nullptr , nullptr , nullptr , nullptr , nullptr };
		vector<Batch> points;
		mutex mtx;

		Node() {

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

	struct Task{
		string file;
		int64_t firstPoint;
		int64_t numPoints;
	};

	shared_ptr<TaskPool<Task>> pool = nullptr;

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

	inline Cube_i childCubeOf(Node* node, int index){
		Cube_i childCube;
		childCube.size = node->cube_i.size / 2;

		if ((index & 0b100) == 0) {
			childCube.min.x = node->cube_i.min.x;
		} else {
			childCube.min.x = node->cube_i.min.x + childCube.size;
		}

		if ((index & 0b010) == 0) {
			childCube.min.y = node->cube_i.min.y;
		} else {
			childCube.min.y = node->cube_i.min.y + childCube.size;
		}

		if ((index & 0b001) == 0) {
			childCube.min.z = node->cube_i.min.z;
		} else {
			childCube.min.z = node->cube_i.min.z + childCube.size;
		}

		return childCube;
	}

	

	void pass(Node* node, Batch batch);
	void addPoints(Node* node, Batch batch);

	void split(Node* node){

		//cout << repeat(" ", 4 * node->level) << "split( " << node->name << ");" << endl;
		
		auto pointsList = node->points;
		node->points = vector<Batch>();

		for(auto points : pointsList){
			pass(node, points);
		}		

	}

	void pass(Node* node, Batch batch){

		//cout << repeat(" ", 4 * node->level) << "pass( " << node->name << ", " << (pointBuffer->size / sizeof(Point)) << ")" << endl;

		for(int i = 0; i < 8; i++){
			
			if(node->children[i] == nullptr){
				Node* child = new Node();
				child->name = node->name + to_string(i);
				child->boundingBox = childBoundingBoxOf(node->boundingBox.min, node->boundingBox.max, i);
				child->cube_i = childCubeOf(node, i);
				child->level = node->level + 1;
				child->index = i;

				node->children[i] = child;
			}

			// FAST PATH
			{
				Node* child = node->children[i];

				if(child->cube_i.encompases(batch.cube)){
					addPoints(child, batch);
					return;
				}
			}

		}

		int counters[8] = {0, 0, 0, 0, 0, 0, 0, 0};

		//for(auto pointBuffer : points)
		{

			Point* points = reinterpret_cast<Point*>(batch.buffer->data);

			for(int i = 0; i < batch.size; i++){

				uint64_t shift_levels = (MORTON_LEVELS - node->level - 1);
				uint64_t shift = 3 * shift_levels;
				uint64_t shifted = points[i].mortonCode >> shift;
				uint64_t childIndex = shifted & 0b111;

				counters[childIndex]++;
			}
		}

		vector<Batch> outputs(8);

		int offset = 0;
		for(int i = 0; i < 8; i++){
			int numPoints = counters[i];
			outputs[i].buffer = make_shared<Buffer>(numPoints * sizeof(Point));
			outputs[i].cube = childCubeOf(node, i);
			outputs[i].size = numPoints;

			memcpy(outputs[i].buffer->data_u8, batch.buffer->data_u8 + offset, numPoints * sizeof(Point));
			offset += numPoints * sizeof(Point);
		}


		for(int i = 0; i < 8; i++){
			int numPoints = counters[i];

			if(numPoints > 0){
				Node* child = node->children[i];

				auto batch = outputs[i];

				addPoints(child, batch);
			}
		}
	}

	void addPoints(Node* node, Batch batch){

		//cout << repeat(" ", 4 * node->level) << "addPoints( " << node->name << ", ..., " << numPoints << ");" << endl;

		// ADD
		if(node->numPoints <= NODE_CAPACITY){
			node->points.push_back(batch);
		}

		node->numPoints += batch.size;

		if(node->name == "r0"){
			int a = 10;
		}

		if(node->numPoints > NODE_CAPACITY && node->points.size() > 0){
			// SPLIT
			split(node);
		}else if(node->numPoints > NODE_CAPACITY) {
			// PASS
			pass(node, batch);
		}

		
	}


	void run(Metadata metadata, LasFile lasfile){

		cout << "loading " << lasfile.path << endl;
		cout << "#points: " << lasfile.numPoints << endl;

		Node* root = new Node();
		root->name = "r";
		root->boundingBox = metadata.boundingBox.cube();
		root->level = 0;
		Cube cube;
		cube.min = metadata.boundingBox.cube().min;
		cube.size = metadata.boundingBox.cube().size().x;
		root->cube = cube;
		Cube_i cube_i;
		cube_i.min = {0, 0, 0};
		cube_i.size = pow(2, MORTON_LEVELS);
		root->cube_i = cube_i;

		string file = lasfile.path;

		auto tStart = now();

		pool = make_shared<TaskPool<Task>>(20, [&metadata, &lasfile, root](shared_ptr<Task> task){

			auto points = LasLoader::loadSync(task->file, task->firstPoint, task->numPoints);

			cout << "loaded " << points.numPoints << endl;

			Point* ppoints = reinterpret_cast<Point*>(points.buffer->data);

			Box box = metadata.boundingBox;
			dvec3 min = box.min;
			dvec3 boxSize = box.size();
			double cubeSize = std::max(std::max(boxSize.x, boxSize.y), boxSize.z);

			Cube_i cube;

			auto toMC = [min, cubeSize, &cube](Point point){
				int32_t mx = MORTON_GRID_SIZE * (point.x - min.x) / cubeSize;
				int32_t my = MORTON_GRID_SIZE * (point.y - min.y) / cubeSize;
				int32_t mz = MORTON_GRID_SIZE * (point.z - min.z) / cubeSize;

				cube.min.x = std::min(cube.min.x, mx);
				cube.min.y = std::min(cube.min.y, my);
				cube.min.z = std::min(cube.min.z, mz);

				int sx = std::max(cube.size, (mx - cube.min.x));
				int sy = std::max(cube.size, (my - cube.min.y));
				int sz = std::max(cube.size, (mz - cube.min.z));
				cube.size = std::max(std::max(sx, sy), std::max(sz, cube.size));

				int64_t mc = morton::encode(mx, my, mz);

				return mc;
			};

			for(int i = 0; i < points.numPoints; i++){
				ppoints[i].mortonCode = toMC(ppoints[i]);
			}

			//stringstream ss;
			//ss << "=== CUBE ===" << endl;
			//ss << cube.min.x << ", " << cube.min.y << ", " << cube.min.z << endl;
			//ss << cube.size << endl;
			//cout << ss.str();

			std::sort(ppoints, ppoints + points.numPoints, [](Point& a, Point& b){
				return a.mortonCode < b.mortonCode;
			});

			Batch batch;
			batch.buffer = points.buffer;
			batch.cube = cube;
			batch.first = 0;
			batch.size = points.numPoints;

			lock_guard<mutex> lock(mtx_add);
			//cout << "finished loading node! adding to tree" << endl;

			auto tStart = now();
			addPoints(root, batch);
			printElapsedTime("addPoints", tStart);

		});


		for(int firstPoint = 0; firstPoint < lasfile.numPoints; firstPoint += MAX_BATCH_SIZE){

			auto task = make_shared<Task>();
			task->file = lasfile.path;
			task->firstPoint = firstPoint;
			task->numPoints = MAX_BATCH_SIZE;
			pool->addTask(task);

		}

		pool->waitTillEmpty();
		pool->close();

		//root->traverse([](Node* node){
		//	
		//	if(node->numPoints == 0){
		//		return;
		//	}

		//	cout << repeat(" ", 4 * node->level) << node->name << " - points: " << formatNumber(node->numPoints) << endl;

		//});


		using namespace std::chrono_literals;

		cout << "done" << endl;

		cout << "====================================" << endl;
		cout << "# ADD BATCHED MULTITHREADED" << endl;
		printElapsedTime("# duration", tStart);
		cout << "====================================" << endl;

	}

}