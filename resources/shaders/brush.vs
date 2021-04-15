#version 450

layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aPivot;
layout(location = 2) in vec4 aColor;
layout(location = 3) in float aSize;
layout(location = 4) in float aTime;
layout(location = 5) in float aRandom;

out vec4 vColor;
out float vSize;
out float vPivotDistance;
out float vRandom;
out float vTime;

layout(std140, binding = 4) uniform shader_data{
	mat4 transform;
	mat4 world;
	mat4 view;
	mat4 proj;

	float time;
	vec2 screenSize;

} ssArgs;

void main() {

	vec4 pos;
	//if(gl_VertexID > 7400){
	vec3 animatedPos = aPosition;
	animatedPos.y += 0.02 * sin(1.5 + 10 * ssArgs.time * aRandom);
	animatedPos.x += 0.02 * cos(1.5 + 10 * ssArgs.time * aRandom);
	pos = ssArgs.transform * vec4(animatedPos, 1.0);
	//}else{
	//pos = ssArgs.transform * vec4(aPosition, 1.0);
	//}

	gl_Position = pos;

	vColor = aColor;
	vSize = aSize;
	vPivotDistance = length(aPosition - aPivot);
	vRandom = aRandom;
	vTime = aTime;

	gl_PointSize = aSize * 0.8;
	gl_PointSize = 0.5 * aSize + 5 * sin(10 * ssArgs.time * aRandom);
}