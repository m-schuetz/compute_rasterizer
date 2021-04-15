
#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <chrono>
#include <thread>
#include <mutex>
#include <atomic>
#include <iostream>
#include <algorithm>
#include <sstream>
#include <type_traits>

#include "GL\glew.h"
#include "GLFW\glfw3.h"

#include "modules/CppUtils/CppUtils.h"
#include "modules/progressive/LASLoader.h"
#include "ComputeShader.h"
#include "GLTimerQueries.h"

using std::string;
using std::vector;
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
using LASLoaderThreaded::LASLoader;
using LASLoaderThreaded::Points;


class ProgressiveLoader {

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

	shared_ptr<LASLoader> loader = nullptr;
	uint32_t prime = 0;
	
	vector<GLuint> ssVertexBuffers;
	GLuint ssChunk16B = -1;
	GLuint ssChunk4B = -1;
	GLuint ssDebug = -1;

	uint32_t pointsUploaded;
	int bytePerPoint = 16;
	ComputeShader* csDistribute = nullptr;
	ComputeShader* csDistributeAttributes = nullptr;

	int maxPointsPerBuffer = 134'000'000;

	ProgressiveLoader(string path) {

		loader = make_shared<LASLoader>(path);
		prime = uint32_t(previousPrimeCongruent3mod4(loader->header.numPoints));

		uint64_t numBuffers = (loader->header.numPoints / maxPointsPerBuffer) + 1;
		if ((loader->header.numPoints % maxPointsPerBuffer) == 0) {
			numBuffers = numBuffers - 1;
		}

		//GLbitfield usage = GL_DYNAMIC_DRAW;
		GLbitfield usage = GL_STATIC_DRAW;

		uint64_t pointsLeft = loader->header.numPoints;
		for (uint64_t i = 0; i < numBuffers; i++) {
			int numPointsInBuffer = pointsLeft > maxPointsPerBuffer ? maxPointsPerBuffer : pointsLeft;

			GLuint ssVertexBuffer;


			glCreateBuffers(1, &ssVertexBuffer);
			uint32_t size = numPointsInBuffer * bytePerPoint;
			
			auto flags = GL_MAP_WRITE_BIT;
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

	~ProgressiveLoader() {
		
	}

	void dispose() {

		for (auto handle : ssVertexBuffers) {
			cout << "delete glbuffer " << handle << endl;
			glDeleteBuffers(1, &handle);
		}

		//glDeleteBuffers(ssVertexBuffers.size(), ssVertexBuffers.data());
		glDeleteBuffers(1, &ssChunk16B);
		glDeleteBuffers(1, &ssChunk4B);
	}

	bool isDone() {
		return loader->allChunksServed();
	}

	///
	/// upload a chunk, if available, and return the number of uploaded points.
	/// returns 0, if no chunk was uploaded.
	///
	uint32_t uploadNextAvailableChunk() {

		auto start = Utils::now();

		Points* chunk = loader->getNextChunk();

		if (chunk == nullptr) {
			return 0;
		}

		int chunkSize = chunk->size;

		GLTimerQueries::timestamp("loader.uploadChunk-start");

		{// upload
			glNamedBufferSubData(ssChunk16B, 0, chunkSize * 16, chunk->xyzrgba.data());
			//glNamedBufferSubData(ssChunkIndices, 0, chunkSize * 4, chunk->shuffledOrder.data());
		}

		{// distribute to shuffled location
			glUseProgram(csDistribute->program);
			
			GLuint ssInput = ssChunk16B;

			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssInput);

			for(int i = 0; i < ssVertexBuffers.size(); i++){
				glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2 + i, ssVertexBuffers[i]);
			}

			auto uLocation = csDistribute->uniformLocations["uNumPoints"];
			glUniform1i(uLocation, chunkSize);

			auto uPrime = csDistribute->uniformLocations["uPrime"];
			glUniform1d(uPrime, double(prime));

			//cout << "offset: " << pointsUploaded << endl;
			auto uOffset = csDistribute->uniformLocations["uOffset"];
			glUniform1i(uOffset, pointsUploaded);

			uint32_t groups = uint32_t(ceil(double(chunkSize) / 32.0));
			glDispatchCompute(groups, 1, 1);

glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, 0);

for (int i = 0; i < ssVertexBuffers.size(); i++) {
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2 + i, 0);
}

glUseProgram(0);

glMemoryBarrier(GL_ALL_BARRIER_BITS);
		}

		GLTimerQueries::timestamp("loader.uploadChunk-end");

		//chunks.emplace_back(chunk);

		pointsUploaded += chunk->size;

		auto duration = Utils::now() - start;

		auto size = chunk->size;
		delete chunk;

		return size;
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

			int bufferIndex = offset / maxPointsPerBuffer;

			GLuint ssInput = ssChunk4B;
			GLuint ssTarget = ssVertexBuffers[bufferIndex];

			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssInput);
			//glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, ssTarget);

			for (int i = 0; i < ssVertexBuffers.size(); i++) {
				glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2 + i, ssVertexBuffers[i]);
			}

			auto uLocation = csDistribute->uniformLocations["uNumPoints"];
			glUniform1i(uLocation, chunkSize);

			auto uPrime = csDistribute->uniformLocations["uPrime"];
			glUniform1d(uPrime, double(prime));

			auto uOffset = csDistribute->uniformLocations["uOffset"];
			glUniform1i(uOffset, targetOffset);

			uint32_t groups = uint32_t(ceil(double(chunkSize) / 32.0));
			glDispatchCompute(groups, 1, 1);

			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, 0);
			//glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, 0);

			for (int i = 0; i < ssVertexBuffers.size(); i++) {
				glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2 + i, 0);
			}

			glUseProgram(0);

			glMemoryBarrier(GL_ALL_BARRIER_BITS);
		}

	}

	template<typename SourceType, typename TargetType>
	static void transformAttribute(void* sourceBuffer, void* targetBuffer, int numPoints, double scale, double offset, int targetByteOffset) {

		SourceType* source = reinterpret_cast<SourceType*>(sourceBuffer);
		int32_t* target = reinterpret_cast<int32_t*>(targetBuffer);

		for (int i = 0; i < numPoints; i++) {
			double v = source[i];
			TargetType value = v * scale + offset;

			auto ptr = &reinterpret_cast<uint8_t*>(targetBuffer)[4 * i + targetByteOffset];
			reinterpret_cast<TargetType*>(ptr)[0] = value;
		}
	}

	template<typename SourceType, typename TargetType>
	static void transformAttributeRange(void* sourceBuffer, void* targetBuffer, int numPoints, double start, double end, int targetByteOffset) {

		SourceType* source = reinterpret_cast<SourceType*>(sourceBuffer);
		int32_t* target = reinterpret_cast<int32_t*>(targetBuffer);

		double rangeWidth = end - start;
		double rangeModifier = 1.0;

		if (std::is_floating_point_v<TargetType>){
			rangeModifier = 1.0;
		} else {
			rangeModifier = std::pow(256.0, double(sizeof(TargetType)));
		}
		rangeModifier = rangeModifier / rangeWidth;

		for (int i = 0; i < numPoints; i++) {
			double v = source[i];
			v = v > end ? end : v;
			v = v < start ? start : v;

			TargetType value = (v - start) * rangeModifier;

			auto ptr = &reinterpret_cast<uint8_t*>(targetBuffer)[4 * i + targetByteOffset];
			reinterpret_cast<TargetType*>(ptr)[0] = value;
		}
	}

};

