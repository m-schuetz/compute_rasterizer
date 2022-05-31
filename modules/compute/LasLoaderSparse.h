
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

#define STEPS_30BIT 1073741824
#define MASK_30BIT 1073741823
#define STEPS_20BIT 1048576
#define MASK_20BIT 1048575
#define STEPS_10BIT 1024
#define MASK_10BIT 1023

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

	struct Batch{
		int64_t chunk_pointOffset;
		int64_t file_pointOffset;
		int64_t sparse_pointOffset;
		int64_t numPoints;
		int64_t file_index;

		dvec3 min = {Infinity, Infinity, Infinity};
		dvec3 max = {-Infinity, -Infinity, -Infinity};
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


	LasLoaderSparse(shared_ptr<Renderer> renderer){

		this->renderer = renderer;

		int pageSize = 0;
		glGetIntegerv(GL_SPARSE_BUFFER_PAGE_SIZE_ARB, &pageSize);
		PAGE_SIZE = pageSize;

		{ // create (sparse) buffers
			this->ssBatches = renderer->createBuffer(64 * 200'000);
			this->ssXyzLow = renderer->createSparseBuffer(4 * MAX_POINTS);
			this->ssXyzMed = renderer->createSparseBuffer(4 * MAX_POINTS);
			this->ssXyzHig = renderer->createSparseBuffer(4 * MAX_POINTS);
			this->ssColors = renderer->createSparseBuffer(4 * MAX_POINTS);
			this->ssLoadBuffer = renderer->createBuffer(200 * MAX_POINTS_PER_BATCH);

			GLuint zero = 0;
			glClearNamedBufferData(this->ssBatches.handle, GL_R32UI, GL_RED, GL_UNSIGNED_INT, &zero);
		}

		int numThreads = 1;
		auto cpuData = getCpuData();

		if(cpuData.numProcessors == 1) numThreads = 1;
		if(cpuData.numProcessors == 2) numThreads = 1;
		if(cpuData.numProcessors == 3) numThreads = 2;
		if(cpuData.numProcessors == 4) numThreads = 3;
		if(cpuData.numProcessors == 5) numThreads = 4;
		if(cpuData.numProcessors == 6) numThreads = 4;
		if(cpuData.numProcessors == 7) numThreads = 5;
		if(cpuData.numProcessors == 8) numThreads = 5;
		if(cpuData.numProcessors  > 8) numThreads = (cpuData.numProcessors / 2) + 1;

		cout << "start loading points with " << numThreads << " threads" << endl;

		for(int i = 0; i < numThreads; i++){
			spawnLoader();
		}
		// spawnLoader();
	}

	void add(vector<string> files, std::function<void(vector<shared_ptr<LasFile>>)> callback){

		vector<shared_ptr<LasFile>> lasfiles;
		static mutex mtx_lasfiles;

		struct Task{
			string file;
			int fileIndex;
		};

		auto ref = this;

		auto processor = [ref, &lasfiles](shared_ptr<Task> task){

			auto lasfile = make_shared<LasFile>();
			lasfile->fileIndex = task->fileIndex;
			lasfile->path = task->file;

			auto buffer_header = readBinaryFile(lasfile->path, 0, 375);

			int versionMajor = buffer_header->get<uint8_t>(24);
			int versionMinor = buffer_header->get<uint8_t>(25);

			if(versionMajor == 1 && versionMinor < 4){
				lasfile->numPoints = buffer_header->get<uint32_t>(107);
			}else{
				lasfile->numPoints = buffer_header->get<uint64_t>(247);
			}

			lasfile->numPoints = min(lasfile->numPoints, 1'000'000'000ll);

			lasfile->offsetToPointData = buffer_header->get<uint32_t>(96);
			lasfile->pointFormat = buffer_header->get<uint8_t>(104);
			lasfile->bytesPerPoint = buffer_header->get<uint16_t>(105);
			
			lasfile->scale.x = buffer_header->get<double>(131);
			lasfile->scale.y = buffer_header->get<double>(139);
			lasfile->scale.z = buffer_header->get<double>(147);
			
			lasfile->offset.x = buffer_header->get<double>(155);
			lasfile->offset.y = buffer_header->get<double>(163);
			lasfile->offset.z = buffer_header->get<double>(171);
			
			lasfile->boxMin.x = buffer_header->get<double>(187);
			lasfile->boxMin.y = buffer_header->get<double>(203);
			lasfile->boxMin.z = buffer_header->get<double>(219);
			
			lasfile->boxMax.x = buffer_header->get<double>(179);
			lasfile->boxMax.y = buffer_header->get<double>(195);
			lasfile->boxMax.z = buffer_header->get<double>(211);

			
			
			{
				unique_lock<mutex> lock1(mtx_lasfiles);
				
				lasfile->numBatches = lasfile->numPoints / POINTS_PER_WORKGROUP + 1;

				lasfile->sparse_point_offset = ref->numPoints;
				
				ref->files.push_back(lasfile);
				ref->numPoints += lasfile->numPoints;
				ref->numBatches += lasfile->numBatches;

				lasfiles.push_back(lasfile);
			}

			{ // create load tasks

				unique_lock<mutex> lock2(ref->mtx_load);
				
				int64_t pointOffset = 0;

				while(pointOffset < lasfile->numPoints){

					int64_t remaining = lasfile->numPoints - pointOffset;
					int64_t pointsInBatch = min(int64_t(MAX_POINTS_PER_BATCH), remaining);

					LoadTask task;
					task.lasfile = lasfile;
					task.firstPoint = pointOffset;
					task.numPoints = pointsInBatch;

					ref->loadTasks.push_back(task);

					pointOffset += pointsInBatch;
				}
			}

		};

		auto cpuData = getCpuData();
		int numThreads = cpuData.numProcessors;

		//if(cpuData.numProcessors == 1) numThreads = 1;
		//if(cpuData.numProcessors == 2) numThreads = 1;
		//if(cpuData.numProcessors == 3) numThreads = 2;
		//if(cpuData.numProcessors == 4) numThreads = 3;
		//if(cpuData.numProcessors == 5) numThreads = 4;
		//if(cpuData.numProcessors == 6) numThreads = 4;
		//if(cpuData.numProcessors == 7) numThreads = 5;
		//if(cpuData.numProcessors == 8) numThreads = 5;
		//if(cpuData.numProcessors  > 8) numThreads = (cpuData.numProcessors / 2) + 1;

		//cout << "start loading file metadata with " << numThreads << " threads" << endl;

		TaskPool<Task> pool(numThreads, processor);

		for(auto file : files){
			auto task = make_shared<Task>();
			task->file = file;
			task->fileIndex = this->numFiles;
			this->numFiles++;

			pool.addTask(task);
		}

		pool.close();
		pool.waitTillEmpty();

		callback(lasfiles);

	}

	// add las files that are to be loaded progressively
	shared_ptr<LasFile> add(string path){

		auto lasfile = make_shared<LasFile>();

		lasfile->fileIndex = this->numFiles;
		this->numFiles++;

		{ // load lasfile metadata
			lasfile->path = path;

			auto buffer_header = readBinaryFile(path, 0, 375);

			int versionMajor = buffer_header->get<uint8_t>(24);
			int versionMinor = buffer_header->get<uint8_t>(25);

			if(versionMajor == 1 && versionMinor < 4){
				lasfile->numPoints = buffer_header->get<uint32_t>(107);
			}else{
				lasfile->numPoints = buffer_header->get<uint64_t>(247);
			}

			lasfile->numPoints = min(lasfile->numPoints, 1'000'000'000ll);

			lasfile->offsetToPointData = buffer_header->get<uint32_t>(96);
			lasfile->pointFormat = buffer_header->get<uint8_t>(104);
			lasfile->bytesPerPoint = buffer_header->get<uint16_t>(105);
			
			lasfile->scale.x = buffer_header->get<double>(131);
			lasfile->scale.y = buffer_header->get<double>(139);
			lasfile->scale.z = buffer_header->get<double>(147);
			
			lasfile->offset.x = buffer_header->get<double>(155);
			lasfile->offset.y = buffer_header->get<double>(163);
			lasfile->offset.z = buffer_header->get<double>(171);
			
			lasfile->boxMin.x = buffer_header->get<double>(187);
			lasfile->boxMin.y = buffer_header->get<double>(203);
			lasfile->boxMin.z = buffer_header->get<double>(219);
			
			lasfile->boxMax.x = buffer_header->get<double>(179);
			lasfile->boxMax.y = buffer_header->get<double>(195);
			lasfile->boxMax.z = buffer_header->get<double>(211);
			
			lasfile->numBatches = lasfile->numPoints / POINTS_PER_WORKGROUP + 1;

			lasfile->sparse_point_offset = this->numPoints;
			
			this->files.push_back(lasfile);
			this->numPoints += lasfile->numPoints;
			this->numBatches += lasfile->numBatches;
		}


		{ // create load tasks
			
			int64_t pointOffset = 0;

			unique_lock<mutex> lock(mtx_load);

			while(pointOffset < lasfile->numPoints){

				int64_t remaining = lasfile->numPoints - pointOffset;
				int64_t pointsInBatch = min(int64_t(MAX_POINTS_PER_BATCH), remaining);

				LoadTask task;
				task.lasfile = lasfile;
				task.firstPoint = pointOffset;
				task.numPoints = pointsInBatch;

				loadTasks.push_back(task);

				pointOffset += pointsInBatch;
			}
		}

		return lasfile;
	}

	// continue progressively loading some data
	// batch: workgroup batch
	// chunk: multiple(~100) workgroup batches loaded from file at once
	void spawnLoader(){

		auto ref = this;

		thread t([ref](){

			while(true){

				std::this_thread::sleep_for(10ms);

				unique_lock<mutex> lock_load(ref->mtx_load);
				
				if(ref->loadTasks.size() == 0){
					lock_load.unlock();

					continue;
				}

				auto task = ref->loadTasks.back();
				ref->loadTasks.pop_back();

				lock_load.unlock();

				static int64_t numBatchesLoaded = 0;

				auto lasfile = task.lasfile;
				string path = lasfile->path;
				int64_t file_byteOffset = lasfile->offsetToPointData + task.firstPoint * lasfile->bytesPerPoint;
				int64_t file_byteSize = task.numPoints * lasfile->bytesPerPoint;
				auto source = readBinaryFile(path, file_byteOffset, file_byteSize);
				// int64_t sparse_batchOffset = numBatchesLoaded;
				int64_t sparse_pointOffset = lasfile->sparse_point_offset + task.firstPoint;

				// compute batch metadata
				int64_t numBatches = task.numPoints / POINTS_PER_WORKGROUP;
				if((task.numPoints % POINTS_PER_WORKGROUP) != 0){
					numBatches++;
				}


				vector<Batch> batches;

				int64_t chunk_pointsProcessed = 0;
				for(int i = 0; i < numBatches; i++){

					int64_t remaining = task.numPoints - chunk_pointsProcessed;
					int64_t numPointsInBatch = std::min(int64_t(POINTS_PER_WORKGROUP), remaining);

					Batch batch;
					
					batch.min = {Infinity, Infinity, Infinity};
					batch.max = {-Infinity, -Infinity, -Infinity};
					batch.chunk_pointOffset = chunk_pointsProcessed;
					batch.file_pointOffset = task.firstPoint + chunk_pointsProcessed;
					batch.sparse_pointOffset = sparse_pointOffset + chunk_pointsProcessed;
					batch.numPoints = numPointsInBatch;

					batches.push_back(batch);

					chunk_pointsProcessed += numPointsInBatch;
				}

				auto bBatches = make_shared<Buffer>(64 * numBatches); 
				auto bXyzLow  = make_shared<Buffer>(4 * task.numPoints);
				auto bXyzMed  = make_shared<Buffer>(4 * task.numPoints);
				auto bXyzHig  = make_shared<Buffer>(4 * task.numPoints);
				auto bColors  = make_shared<Buffer>(4 * task.numPoints);

				dvec3 boxMin = lasfile->boxMin;
				dvec3 cScale = lasfile->scale;
				dvec3 cOffset = lasfile->offset;

				// load batches/points
				for(int batchIndex = 0; batchIndex < numBatches; batchIndex++){
					Batch& batch = batches[batchIndex];

					// compute batch bounding box
					for(int i = 0; i < batch.numPoints; i++){
						int index_pointFile = batch.chunk_pointOffset + i;
						
						int32_t X = source->get<int32_t>(index_pointFile * lasfile->bytesPerPoint + 0);
						int32_t Y = source->get<int32_t>(index_pointFile * lasfile->bytesPerPoint + 4);
						int32_t Z = source->get<int32_t>(index_pointFile * lasfile->bytesPerPoint + 8);

						double x = double(X) * cScale.x + cOffset.x - boxMin.x;
						double y = double(Y) * cScale.y + cOffset.y - boxMin.y;
						double z = double(Z) * cScale.z + cOffset.z - boxMin.z;

						batch.min.x = std::min(batch.min.x, x);
						batch.min.y = std::min(batch.min.y, y);
						batch.min.z = std::min(batch.min.z, z);
						batch.max.x = std::max(batch.max.x, x);
						batch.max.y = std::max(batch.max.y, y);
						batch.max.z = std::max(batch.max.z, z);
					}

					dvec3 batchBoxSize = batch.max - batch.min;

					{
						int64_t batchByteOffset = 64 * batchIndex;
						
						bBatches->set<float>(batch.min.x                  , batchByteOffset +  4);
						bBatches->set<float>(batch.min.y                  , batchByteOffset +  8);
						bBatches->set<float>(batch.min.z                  , batchByteOffset + 12);
						bBatches->set<float>(batch.max.x                  , batchByteOffset + 16);
						bBatches->set<float>(batch.max.y                  , batchByteOffset + 20);
						bBatches->set<float>(batch.max.z                  , batchByteOffset + 24);
						bBatches->set<uint32_t>(batch.numPoints           , batchByteOffset + 28);
						bBatches->set<uint32_t>(batch.sparse_pointOffset  , batchByteOffset + 32);
						bBatches->set<uint32_t>(lasfile->fileIndex        , batchByteOffset + 36);
					}

					int offset_rgb = 0;
					if(lasfile->pointFormat == 2){
						offset_rgb = 20;
					}else if(lasfile->pointFormat == 3){
						offset_rgb = 28;
					}else if(lasfile->pointFormat == 7){
						offset_rgb = 30;
					}else if(lasfile->pointFormat == 8){
						offset_rgb = 30;
					}

					// load data
					for(int i = 0; i < batch.numPoints; i++){
						int index_pointFile = batch.chunk_pointOffset + i;
						
						int32_t X = source->get<int32_t>(index_pointFile * lasfile->bytesPerPoint + 0);
						int32_t Y = source->get<int32_t>(index_pointFile * lasfile->bytesPerPoint + 4);
						int32_t Z = source->get<int32_t>(index_pointFile * lasfile->bytesPerPoint + 8);

						double x = double(X) * cScale.x + cOffset.x - boxMin.x;
						double y = double(Y) * cScale.y + cOffset.y - boxMin.y;
						double z = double(Z) * cScale.z + cOffset.z - boxMin.z;

						uint32_t X30 = uint32_t(((x - batch.min.x) / batchBoxSize.x) * STEPS_30BIT);
						uint32_t Y30 = uint32_t(((y - batch.min.y) / batchBoxSize.y) * STEPS_30BIT);
						uint32_t Z30 = uint32_t(((z - batch.min.z) / batchBoxSize.z) * STEPS_30BIT);

						X30 = min(X30, uint32_t(STEPS_30BIT - 1));
						Y30 = min(Y30, uint32_t(STEPS_30BIT - 1));
						Z30 = min(Z30, uint32_t(STEPS_30BIT - 1));

						{ // low
							uint32_t X_low = (X30 >> 20) & MASK_10BIT;
							uint32_t Y_low = (Y30 >> 20) & MASK_10BIT;
							uint32_t Z_low = (Z30 >> 20) & MASK_10BIT;

							uint32_t encoded = X_low | (Y_low << 10) | (Z_low << 20);

							bXyzLow->set<uint32_t>(encoded, 4 * index_pointFile);
						}

						{ // med
							uint32_t X_med = (X30 >> 10) & MASK_10BIT;
							uint32_t Y_med = (Y30 >> 10) & MASK_10BIT;
							uint32_t Z_med = (Z30 >> 10) & MASK_10BIT;

							uint32_t encoded = X_med | (Y_med << 10) | (Z_med << 20);

							bXyzMed->set<uint32_t>(encoded, 4 * index_pointFile);
						}

						{ // hig
							uint32_t X_hig = (X30 >>  0) & MASK_10BIT;
							uint32_t Y_hig = (Y30 >>  0) & MASK_10BIT;
							uint32_t Z_hig = (Z30 >>  0) & MASK_10BIT;

							uint32_t encoded = X_hig | (Y_hig << 10) | (Z_hig << 20);

							bXyzHig->set<uint32_t>(encoded, 4 * index_pointFile);
						}

						{ // RGB
							

							int R = source->get<uint16_t>(index_pointFile * lasfile->bytesPerPoint + offset_rgb + 0);
							int G = source->get<uint16_t>(index_pointFile * lasfile->bytesPerPoint + offset_rgb + 2);
							int B = source->get<uint16_t>(index_pointFile * lasfile->bytesPerPoint + offset_rgb + 4);

							R = R < 256 ? R : R / 256;
							G = G < 256 ? G : G / 256;
							B = B < 256 ? B : B / 256;

							uint32_t color = R | (G << 8) | (B << 16);

							bColors->set<uint32_t>(color, 4 * index_pointFile);
						}
					}
				}

				UploadTask uploadTask;
				uploadTask.lasfile = lasfile;
				uploadTask.sparse_pointOffset = sparse_pointOffset;
				uploadTask.numPoints = task.numPoints;
				uploadTask.numBatches = numBatches;
				uploadTask.bXyzLow = bXyzLow;
				uploadTask.bXyzMed = bXyzMed;
				uploadTask.bXyzHig = bXyzHig;
				uploadTask.bColors = bColors;
				uploadTask.bBatches = bBatches;
				
				unique_lock<mutex> lock_upload(ref->mtx_upload);
				ref->uploadTasks.push_back(uploadTask);
				lock_upload.unlock();

			}
			
		});
		t.detach();

	}

	void process(){

		// static int numProcessed = 0;

		// FETCH TASK
		unique_lock<mutex> lock(mtx_upload);

		if(uploadTasks.size() == 0){
			return;
		}

		auto task = uploadTasks.back();
		uploadTasks.pop_back();

		lock.unlock();

		// UPLOAD DATA TO GPU

		{ // commit physical memory in sparse buffers
			int64_t offset = 4 * task.sparse_pointOffset;
			int64_t pageAlignedOffset = offset - (offset % PAGE_SIZE);

			int64_t size = 4 * task.numPoints;
			int64_t pageAlignedSize = size - (size % PAGE_SIZE) + PAGE_SIZE;
			pageAlignedSize = std::min(pageAlignedSize, 4 * MAX_POINTS);

			cout << "commiting, offset: " << formatNumber(pageAlignedOffset) << ", size: " << formatNumber(pageAlignedSize) << endl;

			for(auto glBuffer : {ssXyzLow, ssXyzMed, ssXyzHig, ssColors}){
				glBindBuffer(GL_SHADER_STORAGE_BUFFER, glBuffer.handle);
				glBufferPageCommitmentARB(GL_SHADER_STORAGE_BUFFER, pageAlignedOffset, pageAlignedSize, GL_TRUE);
				glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
			}
		}

		static int64_t numBatchesLoaded = 0;

		// upload batch metadata
		glNamedBufferSubData(ssBatches.handle, 
			64 * numBatchesLoaded, 
			task.bBatches->size, 
			task.bBatches->data);

		numBatchesLoaded += task.numBatches;

		// upload batch points
		glNamedBufferSubData(ssXyzLow.handle, 4 * task.sparse_pointOffset, 4 * task.numPoints, task.bXyzLow->data);
		glNamedBufferSubData(ssXyzMed.handle, 4 * task.sparse_pointOffset, 4 * task.numPoints, task.bXyzMed->data);
		glNamedBufferSubData(ssXyzHig.handle, 4 * task.sparse_pointOffset, 4 * task.numPoints, task.bXyzHig->data);
		glNamedBufferSubData(ssColors.handle, 4 * task.sparse_pointOffset, 4 * task.numPoints, task.bColors->data);

		cout << "uploading, offset: " << formatNumber(4 * task.sparse_pointOffset) << ", size: " << formatNumber(4 * task.numPoints) << endl;

		this->numBatchesLoaded += task.numBatches;
		this->numPointsLoaded += task.numPoints;
		task.lasfile->numPointsLoaded += task.numPoints;

		cout << "numBatchesLoaded: " << numBatchesLoaded << endl;

	}

};