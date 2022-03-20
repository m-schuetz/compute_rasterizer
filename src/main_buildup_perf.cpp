
#include <iostream>
#include <thread>
#include <mutex>

#include "LasLoader.h"
#include "unsuck.hpp"
#include "Box.h"
#include "utils.h"

#include "perf/base.h"
#include "perf/add_batched.h"
#include "perf/add_pointwise.h"
#include "perf/batchwise_multithreaded.h"
#include "perf/add_voxelized.h"
#include "perf/add_multithreaded_2.h"
#include "perf/add_morton_multithreaded.h"

using namespace std;



Metadata loadMetadata(string lasdir){

	vector<string> files;
	for (const auto & entry : fs::directory_iterator(lasdir)){
		string filepath = entry.path().string();

		if(iEndsWith(filepath, ".las")){
			files.push_back(filepath);
		}
	}

	Box fullBoundingBox;
	vector<LasFile> lasfiles;
	for(string file : files){

		auto buffer = new Buffer(2 * 1024);
		readBinaryFile(file, 0, buffer->size, buffer->data);

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

		Box boundingBox;
		boundingBox.expand(min);
		boundingBox.expand(max);
		
		fullBoundingBox.expand(boundingBox);

		LasFile lasfile;
		lasfile.path = file;
		lasfile.format = format;
		lasfile.numPoints = numPoints;
		lasfile.offsetToPointData = offsetToPointData;

		lasfiles.push_back(lasfile);
	}

	Metadata metadata;
	metadata.files = lasfiles;
	metadata.boundingBox = fullBoundingBox;

	return metadata;
}

int main(){

	//string file = "D:/dev/pointclouds/eclepens.las";
	//string file = "D:/dev/pointclouds/heidentor.las";
	//string file = "F:/pointclouds/CA13/ot_35121D1203C_1_1.las";
	//string lasdir = "F:/pointclouds/CA13_selection";
	string lasdir = "F:/pointclouds/CA13";
	//string lasdir = "F:/pointclouds/CA13_selection";
	//string lasdir = "F:/pointclouds/ca13_single";

	auto metadata = loadMetadata(lasdir);

	//LasFile lasfile = metadata.files[0];

	//cout << lasfile.path << endl;

	//auto path = fs::path(lasfile.path);
	//cout << path.filename().string() << endl;

	LasFile selected;
	for(auto lasfile : metadata.files){
		auto path = fs::path(lasfile.path);
		if(path.filename().string() == "ot_35120A4202A_1_1.las"){

			batchwise_multithreaded_2::run(metadata, lasfile);
			//add_voxelized::run(metadata, lasfile);

		}
	}
	

	//pointwise::add_pointwise(metadata, lasfile);
	//add_batched(metadata, lasfile);
	//batchwise_multithreaded::run(metadata, lasfile);

	return 0;
}