

#include "modules/progressive/progressive.h"

#include <iostream>
#include <memory>

using std::cout;
using std::endl;
using std::make_shared;

static bool loadingLAS = false;

void uploadHook(shared_ptr<LoadData> loadData) {
	auto start = now();

	loadData->loader->uploadNextAvailableChunk();
	loadData->loader->uploadNextAvailableChunk();
	loadData->loader->uploadNextAvailableChunk();
	loadData->loader->uploadNextAvailableChunk();
	loadData->loader->uploadNextAvailableChunk();

	schedule([loadData]() {

		if (!loadData->loader->isDone()) {
			uploadHook(loadData);
		} else {
			loadData->tEndUpload = now();
			double duration = loadData->tEndUpload - loadData->tStartUpload;
			cout << "upload duration: " << duration << "s" << endl;

			loadingLAS = false;
		}
	});

	auto duration = now() - start;
	//cout << "uploadHook(): " << duration << "s" << endl;
};

ProgressiveLoader* loadLasProgressive(string file) {

	auto loader = new ProgressiveLoader(file);
	shared_ptr<LoadData> load = make_shared<LoadData>();
	load->tStartUpload = now();
	load->loader = loader;

	uploadHook(load);

	return loader;
}


