#version 450

layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aUV;

layout(std140, binding = 4) uniform shader_data{
	mat4 transform;
	mat4 world;
	mat4 view;
	mat4 proj;

	float time;
	vec2 screenSize;

} ssArgs;

out vec2 vUV;
out vec3 vNormal;

void main() {
	vec4 pos = ssArgs.transform * vec4(aPosition, 1.0);

	gl_Position = pos;

	// if(gl_VertexID == 0){
	// 	gl_Position = vec4(0, 0, 0, 1);
	// }else if(gl_VertexID == 1){
	// 	gl_Position = vec4(1, 0, 0, 1);
	// }else if(gl_VertexID == 2){
	// 	gl_Position = vec4(1, 1, 0, 1);
	// }

	vUV = aUV;
	vNormal = aNormal;
}