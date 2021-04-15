#pragma once

#include <string>
#include <memory>
#include <unordered_map>
#include <vector>

#include "unsuck.hpp"

using std::string;
using std::shared_ptr;
using std::unordered_map;
using std::vector;

namespace LasLoader {

	struct LAS {

		string path;
		int64_t numPoints;

		double minX = 0.0;
		double minY = 0.0;
		double minZ = 0.0;

		double maxX = 0.0;
		double maxY = 0.0;
		double maxZ = 0.0;
		
		unordered_map<string, shared_ptr<Buffer>> buffers;

	};

	shared_ptr<LAS> loadLas(string file);

}