
#pragma once

#include <string>
#include <filesystem>

#include "glm/common.hpp"
#include "glm/matrix.hpp"
#include "glm/vec3.hpp"
#include <glm/gtx/transform.hpp>
#include "unsuck.hpp"
#include "Shader.h"
#include "Resources.h"
#include "TaskPool.h"
#include "laszip_api.h"

using namespace std;
using glm::vec3;

namespace fs = std::filesystem;


struct LasFile{
	int64_t fileIndex = 0;
	string path;
	int64_t numPoints = 0;
	int64_t numPointsLoaded = 0;
	uint32_t offsetToPointData = 0;
	int pointFormat = 0;
	uint32_t bytesPerPoint = 0;
	dvec3 scale = {1.0, 1.0, 1.0};
	dvec3 offset = {0.0, 0.0, 0.0};
	dvec3 boxMin;
	dvec3 boxMax;
	
	int64_t numBatches = 0;

	// index of first point in the sparse gpu buffer
	int64_t sparse_point_offset = 0;

	bool isSelected = false;
	bool isHovered = false;
	bool isDoubleClicked = false;
};

struct LasLoaderSparse {

	int64_t MAX_POINTS = 1'000'000'000;
	int64_t PAGE_SIZE = 0;

	mutex mtx_upload;
	mutex mtx_load;

	struct LoadTask{
		shared_ptr<LasFile> lasfile;
		int64_t firstPoint;
		int64_t numPoints;
	};

	struct UploadTask{
		shared_ptr<LasFile> lasfile;
		int64_t sparse_pointOffset;
		int64_t sparse_batchOffset;
		int64_t numPoints;
		int64_t numBatches;
		shared_ptr<Buffer> bXyzLow;
		shared_ptr<Buffer> bXyzMed;
		shared_ptr<Buffer> bXyzHig;
		shared_ptr<Buffer> bColors;
		shared_ptr<Buffer> bBatches;
	};

	vector<shared_ptr<LasFile>> files;
	vector<LoadTask> loadTasks;
	vector<UploadTask> uploadTasks;

	int64_t numPoints = 0;
	int64_t numPointsLoaded = 0;
	int64_t numBatches = 0;
	int64_t numBatchesLoaded = 0;
	int64_t bytesReserved = 0;
	int64_t numFiles = 0;

	shared_ptr<Renderer> renderer = nullptr;

	GLBuffer ssBatches;
	GLBuffer ssXyzLow;
	GLBuffer ssXyzMed;
	GLBuffer ssXyzHig;
	GLBuffer ssColors;
	GLBuffer ssLoadBuffer;

	LasLoaderSparse(shared_ptr<Renderer> renderer);

	void add(vector<string> files, std::function<void(vector<shared_ptr<LasFile>>)> callback);

	void spawnLoader();

	void process();

};