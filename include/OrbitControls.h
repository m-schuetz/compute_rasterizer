
#pragma once

#include <iostream>

#include "glm/common.hpp"
#include "glm/matrix.hpp"
#include <glm/gtx/transform.hpp>

using namespace std;

struct OrbitControls{

	double yaw = 0.0;
	double pitch = 0.0;
	double radius = 2;
	glm::dvec3 target = {0.0, 0.0, 0.0};
	glm::dmat4 world;

	bool isLeftDown = false;
	bool isRightDown = false;

	glm::dvec2 mousePos;

	OrbitControls(){
	
	}

	glm::dvec3 getDirection(){
		auto rotation = getRotation();

		auto dir = rotation * glm::dvec4(0, 1, 0, 1.0);

		return dir;
	}

	glm::dvec3 getPosition(){
		auto dir = getDirection();

		auto pos = target - (radius * dir);

		//cout << 
		//	"[" << target.x << ", " << target.y << ", " << target.z << "] - " <<
		//	"[" << pos.x << ", " << pos.y << ", " << pos.z << "] - " <<
		//	"[" << dir.x << ", " << dir.y << ", " << dir.z << "]" << endl;

		return pos;
	}

	glm::dmat4 getRotation(){
		glm::dvec3 up    = {0, 0, 1};
		glm::dvec3 right = {1, 0, 0};

		auto rotYaw = glm::rotate(yaw, up);
		auto rotPitch = glm::rotate(pitch, right);

		//auto rotation = rotYaw;
		//auto rotation = rotPitch * rotYaw;
		auto rotation = rotPitch * rotYaw;

		return rotation;
	}

	void translate_local(double x, double y, double z){
		auto _pos = glm::dvec3(0, 0, 0);
		auto _right = glm::dvec3(1, 0, 0);
		auto _forward = glm::dvec3(0, 1, 0);
		auto _up = glm::dvec3(0, 0, 1);

		_pos = world * glm::dvec4(_pos, 1);
		_right = world * glm::dvec4(_right, 1);
		_forward = world * glm::dvec4(_forward, 1);
		_up = world * glm::dvec4(_up, 1);

		_right = glm::normalize(_right - _pos) * x;
		_forward = glm::normalize(_forward - _pos) * y;
		_up = glm::normalize(_up - _pos) * (-z);

		this->target = this->target + + _right + _forward + _up;
	}

	void onMouseButton(int button, int action, int mods){
		//cout << "button: " << button << ", action: " << action << ", mods: " << mods << endl;

		if(button == 0 && action == 1){
			isLeftDown = true;
		}else if(action == 0){
			isLeftDown = false;
		}

		if(button == 1 && action == 1){
			isRightDown = true;
		}else if(action == 0){
			isRightDown = false;
		}
	}

	void onMouseMove(double xpos, double ypos);

	void onMouseScroll(double xoffset, double yoffset){
		//cout << xoffset << ", " << yoffset << endl;

		// +1: zoom in
		// -1: zoom out

		if(yoffset < 0.0){
			radius = radius * 1.1;
		}else{
			radius = radius / 1.1;
		}

		//cout << radius << endl;

	}

	void update(){
		glm::dvec3 up    = {0, 0, 1};
		glm::dvec3 right = {1, 0, 0};

		auto translateRadius = glm::translate(
			glm::dmat4(), 
			glm::dvec3(0.0, 0.0, radius));
		auto translateTarget = glm::translate(glm::dmat4(), target);
		auto rotYaw = glm::rotate(yaw, up);
		auto rotPitch = glm::rotate(pitch, right);

		auto flip = glm::dmat4(
			1.0, 0.0, 0.0, 0.0,
			0.0, 0.0, 1.0, 0.0,
			0.0, -1.0, 0.0, 0.0,
			0.0, 0.0, 0.0, 1.0
		);

		world = translateTarget * rotYaw * rotPitch * flip * translateRadius;
	}

};