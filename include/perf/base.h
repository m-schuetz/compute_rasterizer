
#pragma once

struct LasFile{
	string path = "";
	int format = 0;
	Box boundingBox;
	uint64_t offsetToPointData = 0;
	uint64_t numPoints = 0;

};

struct Metadata{

	vector<LasFile> files;
	Box boundingBox;

};