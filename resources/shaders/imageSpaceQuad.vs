#version 450

layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec2 aUV;

layout(location = 0) uniform vec2 uPos;
layout(location = 1) uniform vec2 uSize;

out vec2 vUV;

void main() {
	vec4 pos = vec4(aPosition, 1.0);

	pos.x = (pos.x / 1280) * uSize.x;
	pos.y = (pos.y / 720) * uSize.y;

	pos.xy = pos.xy + uPos;

	gl_Position = pos;

	vUV = aUV;
}