#version 450

#extension GL_NV_gpu_shader5 : enable


layout(local_size_x = 128, local_size_y = 1) in;

struct Vertex{
	float x;
	float y;
	float z;
	uint color;
};

layout(std430, binding = 0) buffer ssInputBuffer{
	Vertex vertices[];
};

layout (std430, binding = 1) buffer grid_data {
	int ssCounters[];
};

layout (std430, binding = 2) buffer offsets_data {
	int ssOffsets[];
};

layout(std430, binding = 3) buffer ssTargetBuffer{
	Vertex sorted[];
};

layout(location = 0) uniform vec3 uMin;
layout(location = 1) uniform vec3 uMax;
layout(location = 2) uniform float uGridSize;

void main(){

	uint globalID = gl_GlobalInvocationID.x;
	//globalID = 0;
	Vertex v = vertices[globalID];

	vec3 normalized = vec3(v.x, v.y, v.z) / (uMax - uMin);

	ivec3 cellIndices = ivec3(
		int(min(normalized.x * uGridSize, uGridSize - 1)),
		int(min(normalized.y * uGridSize, uGridSize - 1)),
		int(min(normalized.z * uGridSize, uGridSize - 1))
	);

	int cellIndex = int(cellIndices.x 
		+ cellIndices.y * uGridSize 
		+ cellIndices.z * uGridSize * uGridSize);

	int targetIndex = atomicAdd(ssOffsets[cellIndex], 1);

	sorted[targetIndex] = v;
}

