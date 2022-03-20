
#pragma once

namespace pointwise{


	constexpr uint64_t NODE_CAPACITY = 1'000'000;
	constexpr uint64_t MAX_BATCH_SIZE = 1'000'000;

	struct Point {
		float x;
		float y;
		float z;
		uint8_t r;
		uint8_t g;
		uint8_t b;
		uint8_t a;
	};

	struct Node {

		string name = "";
		uint64_t numPoints = 0;
		int index = 0;
		Box boundingBox;
		Node* children[8] = { nullptr , nullptr , nullptr , nullptr , nullptr , nullptr , nullptr , nullptr };
		shared_ptr<vector<Point>> points;

		Node() {
			points = make_shared<vector<Point>>();
			points->reserve(NODE_CAPACITY / 10);
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

	void addPoint(Node* node, Point point);
	void passPoint(Node* node, Point point);

	void split(Node* node);

	void split(Node* node) {
		auto points = node->points;

		auto min = node->boundingBox.min;
		auto max = node->boundingBox.max;
		auto size = node->boundingBox.size();

		for(int i = 0; i < node->points->size(); i++) {
			Point point = node->points->at(i);

			passPoint(node, point);
		}

		node->points = make_shared<vector<Point>>();
	}

	void passPoint(Node* node, Point point) {
		auto min = node->boundingBox.min;
		auto max = node->boundingBox.max;
		auto size = node->boundingBox.size();

		double dx = (point.x - min.x) / size.x;
		double dy = (point.y - min.y) / size.y;
		double dz = (point.z - min.z) / size.z;

		int ix = dx < 0.5 ? 0 : 1;
		int iy = dy < 0.5 ? 0 : 1;
		int iz = dz < 0.5 ? 0 : 1;
		int childIndex = (ix << 2) | (iy << 1) | iz;

		if (node->children[childIndex] == nullptr) {
			Node* child = new Node();
			child->name = node->name + to_string(childIndex);
			child->boundingBox = childBoundingBoxOf(min, max, childIndex);

			//cout << "created " << child->name << endl;

			node->children[childIndex] = child;
		}

		addPoint(node->children[childIndex], point);
	}

	void addPoint(Node* node, Point point) {


		// ADD TO THIS NODE
		if (node->numPoints < NODE_CAPACITY) {
			node->points->push_back(point);
		}  

		node->numPoints++;

		// SPLIT NODE
		if (node->numPoints == NODE_CAPACITY) {
			split(node);
		}

		// PASS TO CHILD NODE
		if (node->numPoints > NODE_CAPACITY) {
			passPoint(node, point);
		}

	}

	void add_pointwise(Metadata metadata, LasFile lasfile){

		Node* root = new Node();
		root->name = "r";
		root->boundingBox = metadata.boundingBox.cube();

		string file = lasfile.path;
		bool done = false;
		int64_t semaphore = lasfile.numPoints;
		mutex mtx;
		auto tStart = now();


		for(int firstPoint = 0; firstPoint < lasfile.numPoints; firstPoint += MAX_BATCH_SIZE){

			cout << "loading " << firstPoint << ", " << MAX_BATCH_SIZE << endl;

			LasLoader::load(file, firstPoint, MAX_BATCH_SIZE, [&semaphore, root, &mtx, lasfile](shared_ptr<Buffer> buffer, int64_t numLoaded){


				mtx.lock();

				auto tStartBuilding = now();

				for (int i = 0; i < numLoaded; i++) {
					Point point;
					memcpy(&point, buffer->data_u8 + 16 * i, 16);

					addPoint(root, point);
				}

				printElapsedTime("added points", tStartBuilding);

				semaphore = semaphore - numLoaded;

				mtx.unlock();
			});
		}


		using namespace std::chrono_literals;

		while(semaphore != 0){
			std::this_thread::sleep_for(10ms);
		}

		cout << "done" << endl;

		cout << "====================================" << endl;
		cout << "# ADD POINTWISE" << endl;
		printElapsedTime("# duration", tStart);
		cout << "====================================" << endl;

	}

}