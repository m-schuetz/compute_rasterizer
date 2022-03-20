
#version 450

layout(location = 0) uniform mat4 uView;
layout(location = 1) uniform mat4 uProj;

struct Box{
	vec4 position;   //   16   0
	vec4 size;       //   16  16
	uint color;      //    4  32
	                 // size: 48
};

layout(std430, binding = 0) buffer data_1 {
	Box boxes[];
};

#define NUM_VERTICES_PER_BOX 36

vec3 BOX[NUM_VERTICES_PER_BOX] = vec3[NUM_VERTICES_PER_BOX](
	// BOTTOM
	vec3(-0.5, -0.5, -0.5), vec3( 0.5, -0.5, -0.5), vec3( 0.5,  0.5, -0.5),
	vec3(-0.5, -0.5, -0.5), vec3( 0.5,  0.5, -0.5), vec3(-0.5,  0.5, -0.5),

	// TOP
	vec3(-0.5, -0.5,  0.5), vec3( 0.5, -0.5,  0.5), vec3( 0.5,  0.5,  0.5),
	vec3(-0.5, -0.5,  0.5), vec3( 0.5,  0.5,  0.5), vec3(-0.5,  0.5,  0.5),

	// FRONT
	vec3(-0.5, -0.5, -0.5), vec3( 0.5, -0.5, -0.5), vec3( 0.5, -0.5,  0.5),
	vec3(-0.5, -0.5, -0.5), vec3( 0.5, -0.5,  0.5), vec3(-0.5, -0.5,  0.5),

	// BACK
	vec3(-0.5,  0.5, -0.5), vec3( 0.5,  0.5, -0.5), vec3( 0.5,  0.5,  0.5),
	vec3(-0.5,  0.5, -0.5), vec3( 0.5,  0.5,  0.5), vec3(-0.5,  0.5,  0.5),

	// LEFT
	vec3(-0.5,  0.5, -0.5), vec3(-0.5, -0.5, -0.5), vec3(-0.5, -0.5,  0.5),
	vec3(-0.5,  0.5, -0.5), vec3(-0.5, -0.5,  0.5), vec3(-0.5,  0.5,  0.5),

	// RIGHT
	vec3( 0.5,  0.5, -0.5), vec3( 0.5, -0.5, -0.5), vec3( 0.5, -0.5,  0.5),
	vec3( 0.5,  0.5, -0.5), vec3( 0.5, -0.5,  0.5), vec3( 0.5,  0.5,  0.5)
);

out vec3 vColor;

void main() {

	int boxID = gl_VertexID / NUM_VERTICES_PER_BOX;
	int localVertexID = gl_VertexID % NUM_VERTICES_PER_BOX;
	Box box = boxes[0];

	vec3 pos = BOX[localVertexID];
	pos = pos * box.size.xyz + box.position.xyz;

	gl_Position = uProj * uView * vec4(pos, 1.0);

	gl_PointSize = 20.0;

	uint uColor = box.color;
	
	vColor = vec3(
		float((uColor >>  0) & 0xFF) / 255.0,
		float((uColor >>  8) & 0xFF) / 255.0,
		float((uColor >> 16) & 0xFF) / 255.0
	);

	// vColor.r = 255;

	// vColor = vec3(0.0, 1.0, 0.0);

	// vColor = vec3(1.0, 0.0, 0.0);
	// vColor = vec3(uColor) / 255.0;

}