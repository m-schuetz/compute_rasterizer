#version 450

layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec2 aUV;

out vec2 vUV;

void main() {
	vec4 pos = vec4(aPosition, 1.0);

	gl_Position = pos;

	vUV = aUV;
}