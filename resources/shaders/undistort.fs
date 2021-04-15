#version 450

layout(location = 0) out vec4 out_color;

in vec2 vUV;

layout(binding = 0) uniform sampler2D uDistorted;
layout(binding = 1) uniform sampler2D uMappingRed;
layout(binding = 2) uniform sampler2D uMappingGreen;
layout(binding = 3) uniform sampler2D uMappingBlue;


void main() {

	vec2 uvRed = texture(uMappingRed, vUV * vec2(1, 1)).xy;
	vec2 uvGreen = texture(uMappingGreen, vUV * vec2(1, 1)).xy;
	vec2 uvBlue = texture(uMappingBlue, vUV * vec2(1, 1)).xy;

	vec4 red = texture(uDistorted, uvRed);
	vec4 green = texture(uDistorted, uvGreen);
	vec4 blue = texture(uDistorted, uvBlue);

	vec4 color = vec4(red.r, green.g, blue.b, 1.0);
	
	out_color = color;
	
}

