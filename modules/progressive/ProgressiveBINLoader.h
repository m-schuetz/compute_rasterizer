
#pragma once

#include <string>
#include <vector>
#include <queue>
#include <fstream>
#include <chrono>
#include <thread>
#include <mutex>
#include <atomic>
#include <iostream>
#include <algorithm>
#include <sstream>

#include "GL\glew.h"
#include "GLFW\glfw3.h"

#include "ComputeShader.h"
#include "V8Helper.h"
#include "BArray.h"

#include <Windows.h>

using std::string;
using std::vector;
using std::queue;
using std::ifstream;
using std::ios;
using std::cout;
using std::endl;
using std::streamsize;
using std::thread;
using std::mutex;
using std::unique_lock;
using std::lock_guard;
using std::atomic;
using std::min;
using std::stringstream;

#define DISABLE_FILE_CACHE

class BINLoader {
public:

	string file;
	uint64_t numPoints = 0;

	mutex mtc_access_chunk;
	mutex mtx_binary_chunks;

	queue<BArray*> binaryChunks;
	atomic<uint64_t> numLoaded = 0;

	uint32_t defaultChunkSize = 500'000;

	BINLoader(string file) {
		this->file = file;

		{
			auto size = fs::file_size(file);
			numPoints = size / 16;
			//numPoints = numPoints > 400'000'000 ? 400'000'000 : numPoints;

			//numPoints = 400'000'000;
		}

		createBinaryLoaderThread();
	}

	bool allChunksServed() {
		lock_guard<mutex> lock(mtc_access_chunk);
		bool result = binaryChunks.size() == 0 && numLoaded >= numPoints;

		return result;
	}

	BArray* getNextChunk() {
		lock_guard<mutex> lock(mtc_access_chunk);

		BArray* chunk = nullptr;

		if (binaryChunks.size() > 0) {
			chunk = binaryChunks.front();
			binaryChunks.pop();
		}

		return chunk;
	}

	void createBinaryLoaderThread() {

		thread t([this]() {

			auto start = now();
			

#ifdef DISABLE_FILE_CACHE
			{ // disable windows file cache for benchmarking
				LPCTSTR lfile = file.c_str();

				auto hFile = CreateFile(lfile, GENERIC_READ,
					FILE_SHARE_READ,
					NULL, OPEN_EXISTING,
					FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN
					| FILE_FLAG_NO_BUFFERING, NULL);
			}

#endif

			FILE* in = fopen(file.c_str(), "rb");
			_fseeki64(in, 0, ios::beg);
			auto size = fs::file_size(file);

			uint64_t offset = 0;
			uint64_t pointsLoaded = 0;

			bool done = false;
			while (!done) {

				uint32_t chunkSizePoints = (uint32_t)min(uint64_t(defaultChunkSize), numPoints - pointsLoaded);
				uint32_t chunkSizeBytes = chunkSizePoints * 16;

				BArray* chunkBuffer = new BArray(chunkSizeBytes);
				auto bytesRead = fread(chunkBuffer->data, 1, chunkSizeBytes, in);

				done = bytesRead == 0;

				mtx_binary_chunks.lock();
				//binaryChunks.emplace_back(chunkBuffer);
				binaryChunks.emplace(chunkBuffer);
				mtx_binary_chunks.unlock();

				offset += chunkSizeBytes;
				pointsLoaded += chunkSizePoints;
				numLoaded = pointsLoaded;

				{
					double progress = double(pointsLoaded) / double(numPoints);
					string strProgress = std::to_string(int(progress * 100));
					V8Helper::instance()->debugValue["pointcloud_progress"] = strProgress + "%";
				}
				

				if (pointsLoaded >= numPoints) {
					break;
				}

			}

			auto end = now();
			auto duration = end - start;
			cout << "finished loading file: " << duration << "s" << endl;

			//V8Helper::instance()->debugValue["pointcloud_loaded"] = "true";
			V8Helper::instance()->debugValue["pointcloud_progress"] = "100%";


		});
		t.detach();
	}

};


class ProgressiveBINLoader {

	// see https://en.wikipedia.org/wiki/Primality_test
	bool isPrime(uint64_t n) {
		if (n <= 3) {
			return n > 1;
		} else if ((n % 2) == 0 || (n % 3) == 0) {
			return false;
		}

		uint64_t i = 5;
		while ((i * i) <= n) {
			if ((n % i) == 0 || (n % (i + 2)) == 0) {
				return false;
			}


			i = i + 6;
		}

		return true;
	}

	//
	// Primes where p = 3 mod 4 allow us to generate random numbers without duplicates in range [0, prime - 1]
	// https://preshing.com/20121224/how-to-generate-a-sequence-of-unique-random-integers/
	uint64_t previousPrimeCongruent3mod4(uint64_t start) {
		for (uint64_t i = start -1; true; i--) {
			if ((i % 4) == 3 && isPrime(i)) {
				return i;
			}
		}
	}

public:
	BINLoader* loader = nullptr;


	uint32_t prime = 0;
	
	vector<GLuint> ssVertexBuffers;
	GLuint ssChunk16B = -1;
	GLuint ssChunk4B = -1;
	GLuint ssDebug = -1;

	uint32_t pointsUploaded;
	int bytePerPoint = 16;
	vector<BArray*> chunks;
	ComputeShader* csDistribute = nullptr;
	ComputeShader* csDistributeAttributes = nullptr;

	int maxPointsPerBuffer = 134'000'000;

	ProgressiveBINLoader(string path) {

		loader = new BINLoader(path);
		prime = uint32_t(previousPrimeCongruent3mod4(loader->numPoints));

		uint64_t numBuffers = (loader->numPoints / maxPointsPerBuffer) + 1;
		if ((loader->numPoints % maxPointsPerBuffer) == 0) {
			numBuffers = numBuffers - 1;
		}

		GLbitfield usage = GL_DYNAMIC_DRAW;

		uint64_t pointsLeft = loader->numPoints;
		for (int i = 0; i < numBuffers; i++) {
			int numPointsInBuffer = pointsLeft > maxPointsPerBuffer ? maxPointsPerBuffer : pointsLeft;

			GLuint ssVertexBuffer;


			glCreateBuffers(1, &ssVertexBuffer);
			uint32_t size = numPointsInBuffer * bytePerPoint;
			
			//auto flags = GL_MAP_WRITE_BIT;
			//auto flags = GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT;
			//glNamedBufferStorage(ssVertexBuffer, size, 0, flags);

			//auto myPointer = glMapBufferRange(GL_ARRAY_BUFFER, 0, MY_BUFFER_SIZE, flags);

			glNamedBufferData(ssVertexBuffer, size, nullptr, usage);

			ssVertexBuffers.emplace_back(ssVertexBuffer);

			pointsLeft = pointsLeft - numPointsInBuffer;
		}
		
		{
			uint32_t chunkSize = loader->defaultChunkSize * 16;
			glCreateBuffers(1, &ssChunk16B);
			glNamedBufferData(ssChunk16B, chunkSize, nullptr, usage);
		}

		{
			uint32_t chunkSize = loader->defaultChunkSize * 4;
			glCreateBuffers(1, &ssChunk4B);
			glNamedBufferData(ssChunk4B, chunkSize, nullptr, usage);
		}

		//{
		//	glCreateBuffers(1, &ssDebug);
		//	glNamedBufferData(ssDebug, loader->numPoints * 4, nullptr, usage);
		//}

		string csPath = "../../modules/progressive/distribute.cs";
		csDistribute = new ComputeShader(csPath);
		monitorFile(csPath, [=]() {
			csDistribute = new ComputeShader(csPath);
		});

		string csDAPath = "../../modules/progressive/distribute_attribute.cs";
		csDistributeAttributes = new ComputeShader(csDAPath);
		monitorFile(csDAPath, [=]() {
			csDistributeAttributes = new ComputeShader(csDAPath);
		});

	}

	bool isDone() {
		return loader->allChunksServed();
	}

	///
	/// upload a chunk, if available, and return the number of uploaded points.
	/// returns 0, if no chunk was uploaded.
	///
	uint32_t uploadNextAvailableChunk() {
		BArray* chunk = loader->getNextChunk();

		if (chunk == nullptr) {
			return 0;
		}

		uint64_t chunkSize = chunk->size;
		uint64_t numPoints = chunkSize / 16;

		//{
		//	GLint64 max;
		//	glGetInteger64v(GL_MAX_SHADER_STORAGE_BLOCK_SIZE, &max);

		//	cout << "!!!!!!!!!!!!!  " << max << "   !!!!!!!!!!!!!!!!!" << endl;
		//}

		{// upload
			glNamedBufferSubData(ssChunk16B, 0, chunkSize, chunk->data);
			//glNamedBufferSubData(ssChunkIndices, 0, chunkSize * 4, chunk->shuffledOrder.data());

			// don't keep the data in RAM
			// only for benchmarking reasons, do not commit uncommented!?!
			//delete chunk;
		}

		{// distribute to shuffled location
			glUseProgram(csDistribute->program);
			
			GLuint ssInput = ssChunk16B;

			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssInput);

			//glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 15, ssDebug);

			//cout << "ssDebug: " << ssDebug << endl;

			for(int i = 0; i < ssVertexBuffers.size(); i++){
				glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2 + i, ssVertexBuffers[i]);
			}

			//glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, ssVertexBuffers[0]);
			//glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, ssVertexBuffers[1]);

			auto uLocation = csDistribute->uniformLocations["uNumPoints"];
			glUniform1i(uLocation, numPoints);

			auto uPrime = csDistribute->uniformLocations["uPrime"];
			glUniform1d(uPrime, double(prime));

			//cout << "offset: " << pointsUploaded << endl;
			auto uOffset = csDistribute->uniformLocations["uOffset"];
			glUniform1i(uOffset, pointsUploaded);

			if (pointsUploaded > 50'000'000) {
				int a = 10;
			}

			uint32_t groups = uint32_t(ceil(double(numPoints) / 32.0));
			glDispatchCompute(groups, 1, 1);

			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, 0);

			for (int i = 0; i < ssVertexBuffers.size(); i++) {
				glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2 + i, 0);
			}
			//glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, 0);
			//glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, 0);
			
			glUseProgram(0);

			glMemoryBarrier(GL_ALL_BARRIER_BITS);
		}

		chunks.emplace_back(chunk);

		pointsUploaded += numPoints;

		return numPoints;
	}

	void uploadChunk(void* data, int offset, int size) {

		int targetOffset = offset;
		int chunkSize = size;

		{// upload
			glNamedBufferSubData(ssChunk16B, 0, chunkSize * 16, data);
			//glNamedBufferSubData(ssChunkIndices, 0, chunkSize * 4, chunk->shuffledOrder.data());
		}

		{// distribute to shuffled location
			glUseProgram(csDistribute->program);

			int bufferIndex = pointsUploaded / maxPointsPerBuffer;

			GLuint ssInput = ssChunk16B;
			GLuint ssTarget = ssVertexBuffers[bufferIndex];

			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssInput);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, ssTarget);

			auto uLocation = csDistribute->uniformLocations["uNumPoints"];
			glUniform1i(uLocation, chunkSize);

			auto uPrime = csDistribute->uniformLocations["uPrime"];
			glUniform1d(uPrime, double(prime));

			//cout << "offset: " << targetOffset << endl;
			auto uOffset = csDistribute->uniformLocations["uOffset"];
			glUniform1i(uOffset, targetOffset);

			uint32_t groups = uint32_t(ceil(double(chunkSize) / 32.0));
			glMemoryBarrier(GL_ALL_BARRIER_BITS);
			glDispatchCompute(groups, 1, 1);
			glMemoryBarrier(GL_ALL_BARRIER_BITS);

			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, 0);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, 0);

			glUseProgram(0);

			glMemoryBarrier(GL_ALL_BARRIER_BITS);
		}

	}

	void uploadChunkAttribute(void* data, int offset, int size) {

		int targetOffset = offset;
		int chunkSize = size;

		{// upload
			glNamedBufferSubData(ssChunk4B, 0, chunkSize * 4, data);
			//glNamedBufferSubData(ssChunkIndices, 0, chunkSize * 4, chunk->shuffledOrder.data());
		}

		{// distribute to shuffled location
			glUseProgram(csDistributeAttributes->program);

			int bufferIndex = pointsUploaded / maxPointsPerBuffer;

			GLuint ssInput = ssChunk4B;
			GLuint ssTarget = ssVertexBuffers[bufferIndex];

			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssInput);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, ssTarget);

			auto uLocation = csDistribute->uniformLocations["uNumPoints"];
			glUniform1i(uLocation, chunkSize);

			auto uPrime = csDistribute->uniformLocations["uPrime"];
			glUniform1d(uPrime, double(prime));

			auto uOffset = csDistribute->uniformLocations["uOffset"];
			glUniform1i(uOffset, targetOffset);

			uint32_t groups = uint32_t(ceil(double(chunkSize) / 32.0));
			glDispatchCompute(groups, 1, 1);

			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, 0);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, 0);

			glUseProgram(0);

			glMemoryBarrier(GL_ALL_BARRIER_BITS);
		}

	}

	template<typename SourceType, typename TargetType>
	static void transformAttribute(void* sourceBuffer, void *targetBuffer, int numPoints, double scale, double offset, int targetByteOffset) {

		SourceType* source = reinterpret_cast<SourceType*>(sourceBuffer);
		int32_t* target = reinterpret_cast<int32_t*>(targetBuffer);

		for (int i = 0; i < numPoints; i++) {
			TargetType value = double(source[i]) * scale + offset;

			//target[i] = *reinterpret_cast<int32_t*>(&value);

			auto ptr = &reinterpret_cast<uint8_t*>(targetBuffer)[4 * i + targetByteOffset];
			reinterpret_cast<TargetType*>(ptr)[0] = value;
		}

	}

};

