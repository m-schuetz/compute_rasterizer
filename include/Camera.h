
#pragma once

#include "glm/common.hpp"
#include "glm/matrix.hpp"
#include <glm/gtx/transform.hpp>

struct Camera{

	glm::dvec3 position;
	glm::dmat4 rotation;

	glm::dmat4 world;
	glm::dmat4 view;
	glm::dmat4 proj;

	double aspect = 1.0;
	double fovy = 60.0;
	double near = 0.1;
	double far = 200'000.0;
	int width = 128;
	int height = 128;

	Camera(){

	}

	void setSize(int width, int height){
		this->width = width;
		this->height = height;
		this->aspect = double(width) / double(height);
	}

	void update(){
		view =  glm::inverse(world);

		double pi = glm::pi<double>();
		proj = glm::perspective(pi * fovy / 180.0, aspect, near, far);
	}


};