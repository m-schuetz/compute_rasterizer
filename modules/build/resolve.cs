#version 450

#extension GL_NV_gpu_shader5 : enable


layout(local_size_x = 128, local_size_y = 1) in;

struct Vertex{
	float x;
	float y;
	float z;
	uint colors;
};

layout(std430, binding = 3) buffer ssAverageR{
	uint ssRed[];
};

layout(std430, binding = 4) buffer ssAverageG{
	uint ssGreen[];
};

layout(std430, binding = 5) buffer ssAverageB{
	uint ssBlue[];
};

layout(std430, binding = 6) buffer ssAverageA{
	uint ssWeight[];
};

layout(location = 0) uniform vec3 uMin;
layout(location = 1) uniform vec3 uMax;

layout(location = 1) uniform vec3 uCellMin;
layout(location = 2) uniform vec3 uCellMax;

layout(location = 3) uniform float uGridSize;
layout(location = 4) uniform float uAvgGridSize;

void main(){

	uint globalID = gl_GlobalInvocationID.x;
	uint cellIndex = globalID;
	
	uint w = ssWeight[cellIndex];

	if(w == 0){
		return;
	}

	uint r = ssRed[cellIndex];
	uint g = ssGreen[cellIndex];
	uint b = ssBlue[cellIndex];

	r = (r / w) & 0xFF;
	g = (g / w) & 0xFF;
	b = (b / w) & 0xFF;

	uint colors = 0;
	colors = colors | (r <<  0);
	colors = colors | (g <<  8);
	colors = colors | (b << 16);
	colors = colors | (w << 24);

	Vertex v;
	v.colors = colors;

	
}

