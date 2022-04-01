#version 450

layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec4 aColor;

layout(std140, binding = 4) uniform shader_data{
	mat4 transform;
	mat4 world;
	mat4 view;
	mat4 proj;
	float time;
	vec2 screenSize;
} ssArgs;

out vec3 vColor;

void main() {
	vec4 pos = ssArgs.transform * vec4(aPosition, 1.0);

	gl_Position = pos;

	// gl_Position = vec4(
	// 	gl_VertexID / 100.0,
	// 	0.0, 0.0, 1.0
	// );

	// gl_Position = vec4(
	// 	0.5 * sin(ssArgs.time),
	// 	0.0, 0.0, 1.0
	// );

	// if(aColor.xyz.x == 0 && aColor.xyz.y == 0){
	// 	gl_Position = vec4(10, 10, 10, 1);
	// }

	// gl_Position = vec4(0.0, 0.0, 0.0, 1.0);

	vColor = aColor.xyz;

	// gl_PointSize = 5.0;
}