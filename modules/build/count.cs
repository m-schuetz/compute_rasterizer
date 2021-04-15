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

layout(location = 0) uniform vec3 uMin;
layout(location = 1) uniform vec3 uMax;
layout(location = 2) uniform float uGridSize;

void main(){

	uint globalID = gl_GlobalInvocationID.x;
	//globalID = 0;
	Vertex v = vertices[globalID];

	//vec3 normalized = (vec3(v.x, v.y, v.z) - uMin) / (uMax - uMin);
	vec3 normalized = vec3(v.x, v.y, v.z) / (uMax - uMin);

	ivec3 cellIndices = ivec3(
		int(min(normalized.x * uGridSize, uGridSize - 1)),
		int(min(normalized.y * uGridSize, uGridSize - 1)),
		int(min(normalized.z * uGridSize, uGridSize - 1))
	);

	int cellIndex = int(cellIndices.x 
		+ cellIndices.y * uGridSize 
		+ cellIndices.z * uGridSize * uGridSize);

	atomicAdd(ssCounters[cellIndex], 1);



	// DEBUG
	//ssCounters[0] = cellIndices.x;

	// ssCounters[0] = int(uMin.x * 1000);
	// ssCounters[1] = int(uMax.x * 1000);

	// ssCounters[3] = int(v.x * 1000);
	// ssCounters[4] = int(v.y * 1000);
	// ssCounters[5] = int(v.z * 1000);

	// ssCounters[7] = int(cellIndices.x * 1000);
	// ssCounters[8] = int(cellIndices.y * 1000);
	// ssCounters[9] = int(cellIndices.z * 1000);

	// ssCounters[11] = int(cellIndex);

}

