#version 450

#extension GL_NV_gpu_shader5 : enable

layout(local_size_x = 128, local_size_y = 1) in;

layout (std430, binding = 0) buffer partition_data {
	int ssPointCount;
	int ssCellCount;
};

layout (std430, binding = 1) buffer grid_data {
	int ssCounters[];
};

layout (std430, binding = 2) buffer offsets_data {
	int ssOffsets[];
};

layout (std430, binding = 3) buffer cell_ids_data {
	uint ssCellIDs[];
};

layout (std430, binding = 4) buffer cell_num_point_data {
	uint ssCellNumPoints[];
};

void main(){

	uint globalID = gl_GlobalInvocationID.x;
	uint cellIndex = globalID;
	
	int numPointsInCell = ssCounters[cellIndex];

	if(numPointsInCell > 0){

		int offset = atomicAdd(ssPointCount, numPointsInCell);
		int cellCount = atomicAdd(ssCellCount, 1);

		ssOffsets[cellIndex] = offset;
		ssCellIDs[cellCount] = cellIndex;
		ssCellNumPoints[cellCount] = numPointsInCell;
	}
}

