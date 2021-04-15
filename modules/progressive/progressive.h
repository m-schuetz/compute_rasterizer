
#pragma once

#include <vector>

#include "v8.h"

#include "modules/progressive/ProgressiveLoader.h"
#include "modules/progressive/ProgressiveBINLoader.h"

using std::vector;

using v8::CopyablePersistentTraits;
using v8::Persistent;
using v8::Object;
using v8::Array;
using v8::Isolate;
using v8::ObjectTemplate;
using v8::Local;


class LoadData {
public:

	ProgressiveLoader* loader = nullptr;
	double tStartUpload = 0;
	double tEndUpload = 0;

	LoadData() {

	}

	~LoadData() {
		cout << "~LoadData()" << endl;
	}

};

struct SetAttributeDescriptor {
	string name = "";

	bool useScaleOffset = false;
	double scale = 1.0;
	double offset = 0.0;

	bool useRange = false;
	double rangeStart = 0.0;
	double rangeEnd = 1.0;
};

ProgressiveLoader* loadLasProgressive(string file);
