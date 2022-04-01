
#pragma once

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
	uint32_t padding;
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

void split(Node* node){

	for(auto points : node->points){
		passPoints(node, points);
	}

	node->points = vector<shared_ptr<Buffer>>();

}

void addPoints(Node* node, shared_ptr<Buffer> points){

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

	node->numPoints += numNewPoints;
}

void add_batched(Metadata metadata, LasFile lasfile){

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

			addPoints(root, buffer);

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
	cout << "# ADD BATCHED" << endl;
	printElapsedTime("# duration", tStart);
	cout << "====================================" << endl;

}
