
#include <mutex>
#include <thread>

#include "compute/Resources.h"
#include "compute/ComputeLasLoader.h"
#include "Renderer.h"
#include "GLTimerQueries.h"

using namespace std;

mutex mtx_state;

void ComputeLasData::load(Renderer* renderer){

	cout << "ComputeLasData::load()" << endl;

	{
		lock_guard<mutex> lock(mtx_state);

		if(state != ResourceState::UNLOADED){
			return;
		}else{
			state = ResourceState::LOADING;
		}
	}

	{ // create buffers
		int numBatches = (this->numPoints / POINTS_PER_WORKGROUP) + 1;
		this->ssBatches = renderer->createBuffer(64 * numBatches);
		this->ssXyz_12b = renderer->createSparseBuffer(4 * this->numPoints);
		this->ssXyz_8b = renderer->createSparseBuffer(4 * this->numPoints);
		this->ssXyz_4b = renderer->createSparseBuffer(4 * this->numPoints);
		this->ssColors = renderer->createSparseBuffer(4 * this->numPoints);
		this->ssLoadBuffer = renderer->createBuffer(this->bytesPerPoint * MAX_POINTS_PER_BATCH);

		GLuint zero = 0;
		glClearNamedBufferData(this->ssBatches.handle, GL_R32UI, GL_RED, GL_UNSIGNED_INT, &zero);
		glClearNamedBufferData(this->ssXyz_12b.handle, GL_R32UI, GL_RED, GL_UNSIGNED_INT, &zero);
		glClearNamedBufferData(this->ssXyz_8b.handle, GL_R32UI, GL_RED, GL_UNSIGNED_INT, &zero);
		glClearNamedBufferData(this->ssXyz_4b.handle, GL_R32UI, GL_RED, GL_UNSIGNED_INT, &zero);
		glClearNamedBufferData(this->ssColors.handle, GL_R32UI, GL_RED, GL_UNSIGNED_INT, &zero);
	}


	// start loader thread
	ComputeLasData *ref = this;
	thread t([ref](){

		int pointsRemaining = ref->numPoints;
		int pointsRead = 0;
		while(pointsRemaining > 0){

			{ // abort loader thread if state is set to unloading
				lock_guard<mutex> lock(mtx_state);

				if(ref->state == ResourceState::UNLOADING){
					cout << "stopping loader thread for " << ref->path << endl;

					ref->state = ResourceState::UNLOADED;

					return;
				}
			}

			if(ref->task){
				this_thread::sleep_for(1ms);
				continue;
			}

			int pointsInBatch = std::min(pointsRemaining, MAX_POINTS_PER_BATCH);
			int64_t start = int64_t(ref->offsetToPointData) + int64_t(ref->bytesPerPoint) * int64_t(pointsRead);
			int64_t size = ref->bytesPerPoint * pointsInBatch;
			auto buffer = readBinaryFile(ref->path, start, size);

			auto task = make_shared<LoaderTask>();
			task->buffer = buffer;
			task->pointOffset = pointsRead;
			task->numPoints = pointsInBatch;

			ref->task = task;

			pointsRemaining -= pointsInBatch;
			pointsRead += pointsInBatch;

			Debug::set("numPointsLoaded", formatNumber(pointsRead));

		}

		cout << "finished loading " << formatNumber(pointsRead) << " points" << endl;

		{ // check if resource was marked as unloading in the meantime
			lock_guard<mutex> lock(mtx_state);

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

void ComputeLasData::unload(Renderer* renderer){

	cout << "ComputeLasData::unload()" << endl;

	numPointsLoaded = 0;

	// delete buffers
	glDeleteBuffers(1, &ssBatches.handle);
	glDeleteBuffers(1, &ssXyz_12b.handle);
	glDeleteBuffers(1, &ssXyz_8b.handle);
	glDeleteBuffers(1, &ssXyz_4b.handle);
	glDeleteBuffers(1, &ssColors.handle);
	glDeleteBuffers(1, &ssLoadBuffer.handle);
	//glDeleteBuffers(1, &ssLOD.handle);
	//glDeleteBuffers(1, &ssLODColor.handle);

	lock_guard<mutex> lock(mtx_state);

	if(state == ResourceState::LOADED){
		state = ResourceState::UNLOADED;
	}else if(state == ResourceState::LOADING){
		// if loader thread is still running, notify thread by marking resource as "unloading"
		state = ResourceState::UNLOADING;
	}
}

void ComputeLasData::process(Renderer* renderer){

	static Shader* csLoad = nullptr;

	if(csLoad == nullptr){
		csLoad = new Shader({ {"./modules/compute/computeLasLoader.cs", GL_COMPUTE_SHADER} });
	}

	static GLBuffer ssDebug = renderer->createBuffer(256);

	GLuint zero = 0;
	glClearNamedBufferData(ssDebug.handle, GL_R32UI, GL_RED, GL_UNSIGNED_INT, &zero);

	if(this->task){

		glNamedBufferSubData(this->ssLoadBuffer.handle, 0, this->task->buffer->size, this->task->buffer->data);

		//TODO now run las parse shader;
		if(csLoad->program != -1){

			static int batchCounter = 0;
			// string timestampLabel = "load-batch[" + std::to_string(batchCounter) + "]";
			// GLTimerQueries::timestampPrint(timestampLabel + "-start");
			batchCounter++;

			glUseProgram(csLoad->program);

			auto boxMin = this->boxMin;
			auto boxMax = this->boxMax;
			auto scale = this->scale;
			auto offset = this->offset;

			glUniform1i(11, POINTS_PER_THREAD);

			glUniform3f(20, boxMin.x, boxMin.y, boxMin.z);
			glUniform3f(21, boxMax.x, boxMax.y, boxMax.z);
			glUniform1i(22, this->task->numPoints);
			glUniform1i(23, this->numPoints);
			glUniform1i(24, this->pointFormat);
			glUniform1i(25, this->bytesPerPoint);
			glUniform3f(26, scale.x, scale.y, scale.z);
			glUniform3d(27, offset.x, offset.y, offset.z);

			int batchOffset = this->numPointsLoaded / POINTS_PER_WORKGROUP;
			glUniform1i(30, batchOffset);

			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, this->ssLoadBuffer.handle);

			{ // commit pages

				static int PAGE_SIZE = 0;
				if(PAGE_SIZE == 0){
					glGetIntegerv(GL_SPARSE_BUFFER_PAGE_SIZE_ARB, &PAGE_SIZE);
				}

				int64_t offset = 4 * this->numPointsLoaded;
				int64_t pageAlignedOffset = offset - (offset % PAGE_SIZE);

				int64_t size = 4 * this->task->numPoints;
				int64_t pageAlignedSize = size - (size % PAGE_SIZE) + PAGE_SIZE;
				pageAlignedSize = std::min(pageAlignedSize, this->ssXyz_4b.size);

				glBindBuffer(GL_SHADER_STORAGE_BUFFER, this->ssXyz_12b.handle);
				glBufferPageCommitmentARB(GL_SHADER_STORAGE_BUFFER, pageAlignedOffset, pageAlignedSize, GL_TRUE);
				glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

				glBindBuffer(GL_SHADER_STORAGE_BUFFER, this->ssXyz_8b.handle);
				glBufferPageCommitmentARB(GL_SHADER_STORAGE_BUFFER, pageAlignedOffset, pageAlignedSize, GL_TRUE);
				glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

				glBindBuffer(GL_SHADER_STORAGE_BUFFER, this->ssXyz_4b.handle);
				glBufferPageCommitmentARB(GL_SHADER_STORAGE_BUFFER, pageAlignedOffset, pageAlignedSize, GL_TRUE);
				glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

				glBindBuffer(GL_SHADER_STORAGE_BUFFER, this->ssColors.handle);
				glBufferPageCommitmentARB(GL_SHADER_STORAGE_BUFFER, pageAlignedOffset, pageAlignedSize, GL_TRUE);
				glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
				
				// cout << "page size: " << PAGE_SIZE << endl;
			}


			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 30, ssDebug.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 40, this->ssBatches.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 41, this->ssXyz_12b.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 42, this->ssXyz_8b.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 43, this->ssXyz_4b.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 44, this->ssColors.handle);
			//glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 45, this->ssLOD.handle);
			//glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 46, this->ssLODColor.handle);

			int numBatches = this->task->numPoints / POINTS_PER_WORKGROUP;
			glDispatchCompute(numBatches, 1, 1);

			glMemoryBarrier(GL_ALL_BARRIER_BITS);


			// READ DEBUG VALUES
			// if(Debug::enableShaderDebugValue)
			// {
			// 	glMemoryBarrier(GL_ALL_BARRIER_BITS);

			// 	struct DebugData{
			// 		uint32_t value = 0;
			// 		uint32_t index = 0;
			// 		float x = 0;
			// 		float y = 0;
			// 		float z = 0;
			// 		uint32_t X = 0;
			// 		uint32_t Y = 0;
			// 		uint32_t Z = 0;
			// 		float min_x = 0;
			// 		float min_y = 0;
			// 		float min_z = 0;
			// 		float size_x = 0;
			// 		float size_y = 0;
			// 		float size_z = 0;
			// 		uint32_t check = 0;
			// 	};

			// 	DebugData data;
			// 	glGetNamedBufferSubData(ssDebug.handle, 0, sizeof(DebugData), &data);

			// 	auto dbg = Debug::getInstance();

			// 	if(data.index == 2877987){
			// 		dbg->set("[dbg] index", formatNumber(data.index));
			// 		dbg->set("[dbg] x", formatNumber(data.x, 3));
			// 		dbg->set("[dbg] y", formatNumber(data.y, 3));
			// 		dbg->set("[dbg] z", formatNumber(data.z, 3));
			// 		dbg->set("[dbg] X", formatNumber(data.X));
			// 		dbg->set("[dbg] Y", formatNumber(data.Y));
			// 		dbg->set("[dbg] Z", formatNumber(data.Z));
			// 		dbg->set("[dbg]  min_x", formatNumber(data.min_x, 3));
			// 		dbg->set("[dbg]  min_y", formatNumber(data.min_y, 3));
			// 		dbg->set("[dbg]  min_z", formatNumber(data.min_z, 3));
			// 		dbg->set("[dbg]  siye_x", formatNumber(data.size_x, 3));
			// 		dbg->set("[dbg]  siye_y", formatNumber(data.size_y, 3));
			// 		dbg->set("[dbg]  siye_z", formatNumber(data.size_z, 3));
			// 		dbg->set("[dbg]  check", formatNumber(data.check));
			// 	}

			// 	glMemoryBarrier(GL_ALL_BARRIER_BITS);
			// }

			// GLTimerQueries::timestampPrint(timestampLabel + "-end");

		}

		this->numPointsLoaded += this->task->numPoints;
		this->task = nullptr;

	}

}
