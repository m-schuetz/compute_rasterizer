#version 450

#extension GL_NV_gpu_shader5 : enable


layout(local_size_x = 128, local_size_y = 1) in;

struct Vertex{
	float x;
	float y;
	float z;
	uint colors;
};

struct Box3{
	vec3 min;
	vec3 max;
};

layout(std430, binding = 0) buffer ssPartitions{
	Vertex vertices[];
};

// layout (std430, binding = 1) buffer grid_data {
// 	int ssCounters[];
// };

layout (std430, binding = 2) buffer offsets_data {
	int ssOffsets[];
};

layout(std430, binding = 3) buffer ssAverageR{
	uint red[];
};

layout(std430, binding = 4) buffer ssAverageG{
	uint green[];
};

layout(std430, binding = 5) buffer ssAverageB{
	uint blue[];
};

layout(std430, binding = 6) buffer ssAverageA{
	uint weight[];
};

layout(std430, binding = 7) buffer ssAverageRG{
	uint64_t rg[];
};

layout(location = 0) uniform vec3 uMin;
layout(location = 1) uniform vec3 uMax;
layout(location = 2) uniform float uGridSize;
layout(location = 3) uniform float uAvgGridSize;
layout(location = 4) uniform int uOffset;
layout(location = 5) uniform int uBatchSize;

int getCellIndex(Vertex v){
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

	return cellIndex;
}

Box3 getCellBoundingBox(Vertex v){
	vec3 normalized = vec3(v.x, v.y, v.z) / (uMax - uMin);

	vec3 cellIndices = vec3(
		floor(min(normalized.x * uGridSize, uGridSize - 1)),
		floor(min(normalized.y * uGridSize, uGridSize - 1)),
		floor(min(normalized.z * uGridSize, uGridSize - 1))
	);

	vec3 dimensions = uMax - uMin;

	vec3 cellMin = (dimensions * cellIndices) / uGridSize;
	vec3 cellMax = (dimensions * (cellIndices + 1)) / uGridSize;

	Box3 box;
	box.min = cellMin;
	box.max = cellMax;

	return box;
}

int getAvgCellIndex(Vertex v){

	Box3 box = getCellBoundingBox(v);

	vec3 normalized = (vec3(v.x, v.y, v.z) - box.min) / (box.max - box.min);

	ivec3 cellIndices = ivec3(
		int(min(normalized.x * uAvgGridSize, uAvgGridSize - 1)),
		int(min(normalized.y * uAvgGridSize, uAvgGridSize - 1)),
		int(min(normalized.z * uAvgGridSize, uAvgGridSize - 1))
	);

	int cellIndex = int(cellIndices.x 
		+ cellIndices.y * uAvgGridSize 
		+ cellIndices.z * uAvgGridSize * uAvgGridSize);

	return cellIndex;
}

void main(){

	uint globalID = gl_GlobalInvocationID.x;

	if(globalID >= uBatchSize){
		return;
	}

	Vertex v = vertices[globalID + uOffset];

	//int cellIndex = getCellIndex(v);
	int avgCellIndex = getAvgCellIndex(v);

	uint b = (v.colors >> 16) & 0xFF;
	uint g = (v.colors >> 8) & 0xFF;
	uint r = (v.colors >> 0) & 0xFF;

	atomicAdd(red[avgCellIndex], r);
	atomicAdd(green[avgCellIndex], g);
	atomicAdd(blue[avgCellIndex], b);
	atomicAdd(weight[avgCellIndex], 1);

	//weight[avgCellIndex] = 2;

}

