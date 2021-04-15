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
	// vec4 pos = ssArgs.transform * vec4(0, 0, 0, 1);

	// pos = vec4(0, 0, 0, 1);
	gl_Position = pos;

	if(aColor.xyz.x == 0 && aColor.xyz.y == 0){
		gl_Position = vec4(10, 10, 10, 1);
	}
	// gl_Position = vec4(0, 0, 0, 1);

	// vColor = getColorFromV3();
	// vColor = vec3(1, 0, 0);
	vColor = aColor.xyz;

	// gl_PointSize = 1.0;

}