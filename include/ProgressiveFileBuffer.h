
#pragma once

#include <string>
#include <queue>
#include <vector>
#include <mutex>
#include <thread>

#include "glm/common.hpp"
#include "glm/matrix.hpp"
#include <glm/gtx/transform.hpp>

#include "unsuck.hpp"
#include "Renderer.h"


// progressively load a file into an OpenGL buffer (or multiple, if large).
// main thread must call process() each frame to ensure that data loaded from file 
// by a separate thread gets sent to the GPU in the main thread.
struct ProgressiveFileBuffer{

	struct BufferTask{
		GLuint handle = -1;
		shared_ptr<Buffer> buffer = nullptr;
		int64_t byteOffset = 0;
		int64_t byteSize = 0;
		int64_t totalBytesRead = 0;
	};

	vector<GLuint> glBuffers;

	shared_ptr<BufferTask> task = nullptr;

	static constexpr int64_t MAX_BUFFER_SIZE = 2'000'000'000;
	static constexpr int64_t MAX_BATCH_SIZE = 20'000'000;

	// number of loaded bytes
	int64_t size = 0;
	int64_t fileSize = 0;

	ProgressiveFileBuffer(){

	}

	void process(){

		// should be thread safe if read and assignment are individually atomic... are they?
		if(this->task != nullptr){
			glNamedBufferSubData(task->handle, task->byteOffset, task->byteSize, task->buffer->data);

			this->size = max(this->size, task->totalBytesRead);
			this->task = nullptr;
		}
	}

	static shared_ptr<ProgressiveFileBuffer> load(string path){

		auto loader = make_shared<ProgressiveFileBuffer>();

		int64_t fileSize = fs::file_size(path);
		loader->fileSize = fileSize;

		{// initialize gpu memory
			int64_t remaining = fileSize;
			while(remaining > 0){
				int64_t bufferSize = std::min(remaining, MAX_BUFFER_SIZE);

				GLuint ssbo;
				glCreateBuffers(1, &ssbo);
				glNamedBufferStorage(ssbo, bufferSize, nullptr, GL_DYNAMIC_STORAGE_BIT);

				loader->glBuffers.push_back(ssbo);

				remaining -= bufferSize;
			}
		}

		// start thread that fills gpu memory
		thread t([loader, path, fileSize](){
			
			auto target = make_shared<Buffer>(MAX_BATCH_SIZE);

			int64_t remaining = fileSize;
			int64_t bytesRead = 0;
			while(remaining > 0){

				if(loader->task){
					this_thread::sleep_for(1ms);
					continue;
				}

				int64_t batchSize = std::min(remaining, MAX_BATCH_SIZE);

				int64_t byteOffset = bytesRead;
				readBinaryFile(path, byteOffset , batchSize, target->data);
				
				int targetBufferIndex = byteOffset / MAX_BUFFER_SIZE;
				GLuint targetBufferHandle = loader->glBuffers[targetBufferIndex];

				auto task = make_shared<BufferTask>();
				task->handle = targetBufferHandle;
				task->byteOffset = byteOffset % MAX_BUFFER_SIZE;
				task->byteSize = batchSize;
				task->buffer = target;
				task->totalBytesRead = bytesRead + batchSize;

				// is this simple assignment atomic (thread safe)?
				loader->task = task;

				bytesRead += batchSize;
				remaining -= batchSize;

				// loader->size = bytesRead;

				
			}
			
			cout << "fully loaded " << bytesRead << endl;

		});
		t.detach();

		return loader;
	}

};