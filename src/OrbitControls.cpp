
#include "OrbitControls.h"
#include "Runtime.h"

void OrbitControls::onMouseMove(double xpos, double ypos){

	bool selectActive = Runtime::keyStates[342] > 0;
	if(selectActive){
		return;
	}

	
	glm::dvec2 newMousePos = {xpos, ypos};
	glm::dvec2 diff = newMousePos - mousePos;

	if(isLeftDown){
		yaw -= double(diff.x) / 400.0;
		pitch -= double(diff.y) / 400.0;
	}else if(isRightDown){
		auto ux = diff.x / 1000.0;
		auto uy = diff.y / 1000.0;

		translate_local(-ux * radius, uy * radius, 0);
	}
	
	mousePos = newMousePos;
}