#version 450

layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec4 aColor;
layout(location = 2) in float aRandom;

layout(location = 0) uniform int uNodeIndex;
layout(location = 1) uniform mat4 uTransform;


out vec3 vColor;

void main() {

	vec4 pos = uTransform * vec4(aPosition, 1.0);

	gl_Position = pos;

	vColor = aColor.rgb;

	// MOSTLY RED
	//vColor = aColor * 0.01 + vec3(1.0, 0.0, 0.0);

	// COLOR BY INDEX
	//float w = float(gl_VertexID) / float(node.numPoints);
	//vColor = vec3(w, 0, 0);
}