
#pragma once

#include <execution>

#include "TaskPool.h"

namespace add_voxelized{
	constexpr uint64_t NODE_CAPACITY = 1'000'000;
	constexpr uint64_t MAX_BATCH_SIZE = 1'000'000;
	constexpr int MORTON_LEVELS = 10;
	double MORTON_GRID_SIZE = pow(2, MORTON_LEVELS); // 10: 1024

	mutex mtx_add;

	struct Point {
		double x;
		double y;
		double z;
		uint64_t mortonCode;
	};

	struct Node {

		string name = "";
		uint64_t numPoints = 0;
		int index = 0;
		int level = 0;
		Box boundingBox;
		Node* children[8] = { nullptr , nullptr , nullptr , nullptr , nullptr , nullptr , nullptr , nullptr };
		vector<shared_ptr<Buffer>> points;
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

	void pass(Node* node, shared_ptr<Buffer> points);
	void addPoints(Node* node, shared_ptr<Buffer> points, int numPoints);

	void split(Node* node){

		//cout << repeat(" ", 4 * node->level) << "split( " << node->name << ");" << endl;
		
		auto pointsList = node->points;
		node->points = vector<shared_ptr<Buffer>>();

		for(auto points : pointsList){
			pass(node, points);
		}		

	}

	void pass(Node* node, shared_ptr<Buffer> pointBuffer){

		//cout << repeat(" ", 4 * node->level) << "pass( " << node->name << ", " << (pointBuffer->size / sizeof(Point)) << ")" << endl;

		int counters[8] = {0, 0, 0, 0, 0, 0, 0, 0};

		//for(auto pointBuffer : points)
		{

			Point* points = reinterpret_cast<Point*>(pointBuffer->data);
			int size = pointBuffer->size / sizeof(Point);


			for(int i = 0; i < size; i++){

				uint64_t shift_levels = (MORTON_LEVELS - node->level - 1);
				uint64_t shift = 3 * shift_levels;
				uint64_t shifted = points[i].mortonCode >> shift;
				uint64_t childIndex = shifted & 0b111;

				counters[childIndex]++;
			}
		}

		vector<shared_ptr<Buffer>> outputs(8);


		for(int i = 0; i < 8; i++){
			int numPoints = counters[i];
			outputs[i] = make_shared<Buffer>(numPoints * sizeof(Point));
		}

		//for(auto pointBuffer : points)
		{

			Point* points = reinterpret_cast<Point*>(pointBuffer->data);
			int size = pointBuffer->size / sizeof(Point);

			for(int i = 0; i < size; i++){

				uint64_t shift_levels = (MORTON_LEVELS - node->level - 1);
				uint64_t shift = 3 * shift_levels;
				uint64_t shifted = points[i].mortonCode >> shift;
				uint64_t childIndex = shifted & 0b111;

				//outputs[childIndex].push_back(points[i]);

				auto target = outputs[childIndex];
				memcpy(target->data_u8 + target->pos, points + i, sizeof(Point));
				target->pos += sizeof(Point);

			}
		}


		for(int i = 0; i < 8; i++){
			int numPoints = counters[i];

			if(numPoints > 0){

				if(node->children[i] == nullptr){
					Node* child = new Node();
					child->name = node->name + to_string(i);
					child->boundingBox = childBoundingBoxOf(node->boundingBox.min, node->boundingBox.max, i);
					child->level = node->level + 1;
					child->index = i;

					node->children[i] = child;
				}

				Node* child = node->children[i];

				auto points = outputs[i];

				addPoints(child, points, numPoints);
			}
		}
	}

	void addPoints(Node* node, shared_ptr<Buffer> points, int numPoints){

		//cout << repeat(" ", 4 * node->level) << "addPoints( " << node->name << ", ..., " << numPoints << ");" << endl;

		// ADD
		if(node->numPoints <= NODE_CAPACITY){
			node->points.push_back(points);
		}

		node->numPoints += numPoints;

		if(node->name == "r0"){
			int a = 10;
		}

		if(node->numPoints > NODE_CAPACITY && node->points.size() > 0){
			// SPLIT
			split(node);
		}else if(node->numPoints > NODE_CAPACITY) {
			// PASS
			pass(node, points);
		}

		
	}


	void run(Metadata metadata, LasFile lasfile){

		cout << "loading " << lasfile.path << endl;
		cout << "#points: " << lasfile.numPoints << endl;

		Node* root = new Node();
		root->name = "r";
		root->boundingBox = metadata.boundingBox.cube();
		root->level = 0;

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

			auto toMC = [min, cubeSize](Point point){
				int32_t mx = MORTON_GRID_SIZE * (point.x - min.x) / cubeSize;
				int32_t my = MORTON_GRID_SIZE * (point.y - min.y) / cubeSize;
				int32_t mz = MORTON_GRID_SIZE * (point.z - min.z) / cubeSize;

				int64_t mc = morton::encode(mx, my, mz);

				return mc;
			};

			for(int i = 0; i < points.numPoints; i++){
				ppoints[i].mortonCode = toMC(ppoints[i]);
			}

			//std::sort(ppoints, ppoints + points.numPoints, [](Point& a, Point& b){
			//	return a.mortonCode < b.mortonCode;
			//});

			struct MC{
				int64_t mc;
				int32_t index;
			};

			vector<MC> mcs(points.numPoints);
			for(int i = 0; i < points.numPoints; i++){
				mcs[i].mc = toMC(ppoints[i]);
				mcs[i].index = i;
			}

			std::sort(mcs.begin(), mcs.end(), [](MC& a, MC& b){
				return a.mc < b.mc;
			});

			auto targetBuffer = make_shared<Buffer>(points.buffer->size);
			Point* targetPoints = reinterpret_cast<Point*>(targetBuffer->data);

			for(int i = 0; i < points.numPoints; i++){
				targetPoints[i] = ppoints[mcs[i].index];
			}


			int currentVoxelIndex = -1;
			int currentVoxelStart = 0;
			int currentVoxelSize = 0;

			for(int i = 0; i < points.numPoints; i++){
			
				Point point = targetPoints[i];
				int voxelIndex = point.mortonCode >> 6;

				if(voxelIndex == currentVoxelIndex){
					currentVoxelSize++;
				}else{
					// finish voxel
					cout << "finish voxel. index(" << currentVoxelIndex << "), start(" << currentVoxelStart << "), size(" << currentVoxelSize << ")" << endl;

					// start new voxel
					currentVoxelIndex = voxelIndex;
					currentVoxelStart = i;
					currentVoxelSize = 1;
				}


			}
			



			//lock_guard<mutex> lock(mtx_add);
			//cout << "finished loading node! adding to tree" << endl;
			//addPoints(root, targetBuffer, points.numPoints);



		});


		for(int firstPoint = 0; firstPoint < lasfile.numPoints; firstPoint += MAX_BATCH_SIZE){

			auto task = make_shared<Task>();
			task->file = lasfile.path;
			task->firstPoint = firstPoint;
			task->numPoints = MAX_BATCH_SIZE;
			pool->addTask(task);

			break;

		}

		pool->waitTillEmpty();
		pool->close();

		root->traverse([](Node* node){
			
			cout << repeat(" ", 4 * node->level) << node->name << " - points: " << formatNumber(node->numPoints) << endl;

		});


		using namespace std::chrono_literals;

		cout << "done" << endl;

		cout << "====================================" << endl;
		cout << "# ADD BATCHED MULTITHREADED" << endl;
		printElapsedTime("# duration", tStart);
		cout << "====================================" << endl;

	}

}