
#pragma once

#include <vector>

#include "glm/common.hpp"
#include "glm/matrix.hpp"
#include <glm/gtx/transform.hpp>

#include "Plane.h"
#include "Box.h"

using namespace std;
using glm::dmat4;
using glm::mat4;
using glm::dvec3;
using glm::dvec4;

struct Frustum{

	vector<Plane> planes;

	Frustum(){
		planes.resize(6);
	}

	void set(dmat4 me){
		auto me0   = me[0][0],  me1 = me[0][1],  me2 = me[0][2],  me3 = me[0][3];
		auto me4   = me[1][0],  me5 = me[1][1],  me6 = me[1][2],  me7 = me[1][3];
		auto me8   = me[2][0],  me9 = me[2][1], me10 = me[2][2], me11 = me[2][3];
		auto me12  = me[3][0], me13 = me[3][1], me14 = me[3][2], me15 = me[3][3];

		planes[0].set( me3 - me0, me7 - me4, me11 -  me8, me15 - me12 )->normalize();
		planes[1].set( me3 + me0, me7 + me4, me11 +  me8, me15 + me12 )->normalize();
		planes[2].set( me3 + me1, me7 + me5, me11 +  me9, me15 + me13 )->normalize();
		planes[3].set( me3 - me1, me7 - me5, me11 -  me9, me15 - me13 )->normalize();
		planes[4].set( me3 - me2, me7 - me6, me11 - me10, me15 - me14 )->normalize();
		planes[5].set( me3 + me2, me7 + me6, me11 + me10, me15 + me14 )->normalize();
	}

	bool intersectsBox(Box box){
		
		for(int i = 0; i < 6; i++){
			Plane& plane = planes[i];

			dvec3 vec = {
				plane.normal.x > 0.0 ? box.max.x : box.min.x,
				plane.normal.y > 0.0 ? box.max.y : box.min.y,
				plane.normal.z > 0.0 ? box.max.z : box.min.z
			};

			if(plane.distanceTo(vec) < 0.0){
				return false;
			}
		}

		return true;
	}

};