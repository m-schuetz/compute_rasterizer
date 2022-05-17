
#version 450

layout(location = 0) uniform mat4 uView;
layout(location = 1) uniform mat4 uProj;
// layout(location = 5) uniform vec3 uPos;
// layout(location = 6) uniform vec3 uScale;
// layout(location = 7) uniform ivec3 uColor;

struct Box{
	vec4 position;   //   16   0
	vec4 size;       //   16  16
	uint color;      //    4  32
	                 // size: 48
};

layout(std430, binding = 0) buffer data_1 {
	uint count;
	uint instanceCount;
	uint first;
	uint baseInstance;
	uint pad0; uint pad1; uint pad2; uint pad3;
	uint pad4; uint pad5; uint pad6; uint pad7;
	// offset: 48
	Box boxes[];
};

vec3 BOX[24] = vec3[24](
	// BOTTOM
	vec3(-0.5, -0.5, -0.5), vec3( 0.5, -0.5, -0.5),
	vec3( 0.5, -0.5, -0.5), vec3( 0.5,  0.5, -0.5),
	vec3( 0.5,  0.5, -0.5), vec3(-0.5,  0.5, -0.5),
	vec3(-0.5,  0.5, -0.5), vec3(-0.5, -0.5, -0.5),

	// TOP
	vec3(-0.5, -0.5,  0.5), vec3( 0.5, -0.5,  0.5),
	vec3( 0.5, -0.5,  0.5), vec3( 0.5,  0.5,  0.5),
	vec3( 0.5,  0.5,  0.5), vec3(-0.5,  0.5,  0.5),
	vec3(-0.5,  0.5,  0.5), vec3(-0.5, -0.5,  0.5),

	// CONNECT
	vec3(-0.5, -0.5, -0.5), vec3(-0.5, -0.5,  0.5),
	vec3( 0.5, -0.5, -0.5), vec3( 0.5, -0.5,  0.5),
	vec3( 0.5,  0.5, -0.5), vec3( 0.5,  0.5,  0.5),
	vec3(-0.5,  0.5, -0.5), vec3(-0.5,  0.5,  0.5)
);

out vec3 vColor;

void main() {

	//int boxID = gl_InstanceID;
	int boxID = gl_VertexID / 24 + gl_InstanceID;
	int localVertexID = gl_VertexID % 24;
	Box box = boxes[boxID];

	vec3 pos = BOX[localVertexID];
	pos = pos * box.size.xyz + box.position.xyz;

	gl_Position = uProj * uView * vec4(pos, 1.0);
	// gl_Position = vec4(float(gl_VertexID) / 10.0, 0.0, 0.5, 1.0);

	gl_PointSize = 3.0;

	uint uColor = box.color;
	
	vColor = vec3(
		float((uColor >>  0) & 0xFF) / 255.0,
		float((uColor >>  8) & 0xFF) / 255.0,
		float((uColor >> 16) & 0xFF) / 255.0
	);

}