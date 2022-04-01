
#pragma once

#include "glm/common.hpp"
#include "glm/matrix.hpp"
#include <glm/gtx/transform.hpp>

using glm::vec3;
using glm::dvec3;
using glm::ivec3;

struct Box{
	dvec3 min = { Infinity, Infinity, Infinity };
	dvec3 max = { -Infinity, -Infinity, -Infinity };
	ivec3 color;

	Box() {

	}

	Box(vec3 min, vec3 max) {
		this->min = min;
		this->max = max;
	}

	dvec3 center(){
		return (min + max) / 2.0;
	}

	dvec3 size(){
		return max - min;
	}
	
	Box cube() {
		auto size = this->size();
		double cubeSize = std::max(std::max(size.x, size.y), size.z);

		Box cubic;
		cubic.min = this->min;
		cubic.max = this->min + cubeSize;

		return cubic;
	}

	void expand(Box box) {

		this->min.x = std::min(this->min.x, box.min.x);
		this->min.y = std::min(this->min.y, box.min.y);
		this->min.z = std::min(this->min.z, box.min.z);

		this->max.x = std::max(this->max.x, box.max.x);
		this->max.y = std::max(this->max.y, box.max.y);
		this->max.z = std::max(this->max.z, box.max.z);

	}

	void expand(dvec3 point){
		this->min.x = std::min(this->min.x, point.x);
		this->min.y = std::min(this->min.y, point.y);
		this->min.z = std::min(this->min.z, point.z);

		this->max.x = std::max(this->max.x, point.x);
		this->max.y = std::max(this->max.y, point.y);
		this->max.z = std::max(this->max.z, point.z);
	}
};