
#pragma once

#include "glm/common.hpp"
#include "glm/matrix.hpp"
#include <glm/gtx/transform.hpp>

using namespace std;
using glm::dmat4;
using glm::mat4;
using glm::dvec3;
using glm::dvec4;

class Plane{

public:

	dvec3 normal = {0.0, 0.0, 0.0};
	double constant = 0.0;

	Plane(){
	
	}

	Plane(dvec3 normal, double constant){
		this->normal = normal;
		this->constant = constant;
	}

	Plane* set(double x, double y, double z, double constant){
		this->normal = {x, y, z};
		this->constant = constant;

		return this;
	}

	double distanceTo(dvec3 point){
		double distance = glm::dot(normal, point) + constant;

		return distance;
	}

	Plane* normalize(){
		double length = glm::length(normal);

		normal = normal / length;
		constant = constant / length;

		return this;
	}

};