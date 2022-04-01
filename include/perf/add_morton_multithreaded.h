
#pragma once

#include <algorithm>
#include <vector>
#include <string>
#include <functional>
#include <thread>
#include <mutex>

#include "Box.h"
#include "unsuck.hpp"
#include "LasLoader.h"
#include "base.h"
#include "utils.h"


using namespace std;

namespace add_morton_multithreaded {

	constexpr uint64_t NODE_CAPACITY = 1'000'000;
	constexpr uint64_t MAX_BATCH_SIZE = 1'000'000;

	struct Point {
		double x;
		double y;
		double z;
		uint8_t r;
		uint8_t g;
		uint8_t b;
		uint8_t a;
		uint32_t mortonCode;
	};

	struct Node {

		string name = "";
		uint64_t numPoints = 0;
		int index = 0;
		Box boundingBox;
		Node* children[8] = { nullptr , nullptr , nullptr , nullptr , nullptr , nullptr , nullptr , nullptr };
		vector<shared_ptr<Buffer>> points;

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

	void addPoints(Node* node, shared_ptr<Buffer> points);

	void passPoints(Node* node, shared_ptr<Buffer> points) {

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
		int counters[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
		Point point;
		for (int i = 0; i < numPoints; i++) {
			memcpy(&point, points->data_u8 + 32 * i, 32);

			int X = ((point.x - minX) / sizeX) > 0.5 ? 1 : 0;
			int Y = ((point.y - minY) / sizeY) > 0.5 ? 1 : 0;
			int Z = ((point.z - minZ) / sizeZ) > 0.5 ? 1 : 0;

			int index = (X << 2) | (Y << 1) | Z;

			counters[index]++;
		}

		shared_ptr<Buffer> buffers[8];
		for (int i = 0; i < 8; i++) {
			int numChildPoints = counters[i];
			buffers[i] = make_shared<Buffer>(32 * numChildPoints);
		}

		for (int i = 0; i < numPoints; i++) {
			memcpy(&point, points->data_u8 + 32 * i, 32);

			int X = ((point.x - minX) / sizeX) > 0.5 ? 1 : 0;
			int Y = ((point.y - minY) / sizeY) > 0.5 ? 1 : 0;
			int Z = ((point.z - minZ) / sizeZ) > 0.5 ? 1 : 0;

			int index = (X << 2) | (Y << 1) | Z;

			buffers[index]->write(&point, 32);
		}

		for (int childIndex = 0; childIndex < 8; childIndex++) {
			auto points = buffers[childIndex];

			if (points->size > 0) {
				if (node->children[childIndex] == nullptr) {
					Node* child = new Node();
					child->name = node->name + to_string(childIndex);
					child->boundingBox = childBoundingBoxOf(min, max, childIndex);

					cout << "created " << child->name << endl;

					node->children[childIndex] = child;
				}

				addPoints(node->children[childIndex], points);
			}

		}

	}

	void split(Node* node) {

		for (auto points : node->points) {
			passPoints(node, points);
		}

		node->points = vector<shared_ptr<Buffer>>();

	}

	struct Target {
		Node* node = nullptr;
		uint32_t mask = 0;
	};

	Target findTarget(Node* root, Point point) {

		Node* current = root;

		int level = 0;

		do {

			int shift = 3 * (8 - level);
			int childIndex = (point.mortonCode >> shift) & 0b111;

			Node* child = current->children[childIndex];

			if (child == nullptr) {
				break;
			} else {
				current = child;
			}

			level++;
		} while (level <= 8);

		uint32_t mask = 0;
		for (int i = 0; i <= level; i++) {
			int shift = 3 * (8 - level);
			mask = mask | (0b111 << shift);
		}

		Target target = { current, mask };
		
		return target;
	}

	void addPoints(Node* node, shared_ptr<Buffer> buffer) {

		int numPoints = buffer->size / 32;
		Point* points = reinterpret_cast<Point*>(buffer->data);

		for (int i = 0; i < numPoints; i++) {

			Point point = points[i];

			Target target = findTarget(node, point);

			int splitSize = 1;
			do {

				Point p = points[i + splitSize];

				uint32_t startMask = (point.mortonCode & target.mask);
				uint32_t toMask = (p.mortonCode & target.mask);

				if (startMask != toMask) {
					break;
				}

			} while(i + splitSize < numPoints);

			shared_ptr<Buffer> part = make_shared<Buffer>(32 * splitSize);
			memcpy(part->data, buffer->data_u8 + 32 * i, 32 * splitSize);



			// 1. find node to attach to 
			// 2. add all consecutive points that fall into same node
			//    check morton code with mask to see if its the same node





		}


		//int numNewPoints = points->size / 32;

		//if (node->numPoints > NODE_CAPACITY) {
		//	// PASS
		//	passPoints(node, points);
		//} else if (node->numPoints <= NODE_CAPACITY && node->numPoints + numNewPoints > NODE_CAPACITY) {
		//	// SPLIT
		//	node->points.push_back(points);
		//	split(node);
		//} else {
		//	// APPEND
		//	node->points.push_back(points);
		//}

		//node->numPoints += numNewPoints;
	}

	void run(Metadata metadata, LasFile lasfile) {

		Node* root = new Node();
		root->name = "r";
		root->boundingBox = metadata.boundingBox.cube();

		string file = lasfile.path;
		bool done = false;
		int64_t semaphore = lasfile.numPoints;
		mutex mtx;
		auto tStart = now();


		for (int firstPoint = 0; firstPoint < lasfile.numPoints; firstPoint += MAX_BATCH_SIZE) {

			cout << "loading " << firstPoint << ", " << MAX_BATCH_SIZE << endl;

			LasLoader::load(file, firstPoint, MAX_BATCH_SIZE, [&semaphore, root, &mtx, lasfile](shared_ptr<Buffer> buffer, int64_t numLoaded) {

				Point* points = reinterpret_cast<Point*>(buffer->data);

				auto min = root->boundingBox.min;
				auto max = root->boundingBox.max;
				auto size = root->boundingBox.size();
				double gridSize = pow(2.0, 9.0);

				for (int i = 0; i < numLoaded; i++) {
					Point point = points[i];

					uint32_t X = clamp(gridSize * (point.x - min.x) / size.x, 0.0, gridSize - 1.0);
					uint32_t Y = clamp(gridSize * (point.y - min.y) / size.y, 0.0, gridSize - 1.0);
					uint32_t Z = clamp(gridSize * (point.z - min.z) / size.z, 0.0, gridSize - 1.0);

					uint64_t mortonCode = morton::encode(X, Y, Z);
					uint32_t mc32 = mortonCode & 0xFFFFFFFF;

					assert(mortonCode == mc32);

					point.mortonCode = mc32;

					points[i] = point;
				}

				std::sort(points, points + numLoaded, [](Point& a, Point& b) {
					return a.mortonCode - b.mortonCode;
				});

				lock_guard<mutex> lock(mtx);

				auto tStartBuilding = now();

				addPoints(root, buffer);

				printElapsedTime("added points", tStartBuilding);

				semaphore = semaphore - numLoaded;

			});
		}


		using namespace std::chrono_literals;

		while (semaphore != 0) {
			std::this_thread::sleep_for(10ms);
		}

		cout << "done" << endl;

		cout << "====================================" << endl;
		cout << "# ADD MORTON MULTITHREADED" << endl;
		printElapsedTime("# duration", tStart);
		cout << "====================================" << endl;

	}

};