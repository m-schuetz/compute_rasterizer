
#pragma once

#include <vector>
#include <fstream>
#include <iostream>
#include <filesystem>

#include "libplatform/libplatform.h"
#include "v8.h"
#include "V8Helper.h"
#include <filesystem>

using std::vector;
using std::ifstream;
using std::ios;

namespace fs = std::filesystem;

class File {

	ifstream *stream = nullptr;
	string path = "";

public:

	File(string path) {
		this->path = path;
		stream = new ifstream(path, ios::in | ios::binary);
		bool ok = stream->good();

		int a = 10;
	}

	vector<char> readBytes(int numBytes) {
		vector<char> bytes;
		bytes.resize(numBytes);

		stream->read(bytes.data(), numBytes);
		
		auto actuallyRead = stream->gcount();

		if (actuallyRead < numBytes) {
			bytes.resize(actuallyRead);
		}

		return bytes;
	}

	uint64_t fileSize() {
		uint64_t size = fs::file_size(this->path);

		return size;
	}

	void setReadLocation(long long location) {
		stream->seekg(location);
	}

	void close() {
		stream->close();
		delete stream;
		stream = nullptr;
	}

};


Local<ObjectTemplate> createV8FileTemplate(v8::Isolate *isolate);

Local<Object> v8Object(File *shader);