#version 450

#extension GL_NV_shader_atomic_int64 : enable
#extension GL_NV_gpu_shader5 : enable
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_NV_shader_atomic_float : require
// #extension GL_EXT_shader_atomic_float2 : require

#define Infinity (1.0 / 0.0)
#define MAX_BUFFER_SIZE 2000000000l

#define SCALE 0.001
#define FACTOR_20BIT 1048576.0
#define FACTOR_19BIT 524288.0
#define SCALE_20BIT (1.0 / FACTOR_20BIT)
#define SCALE_19BIT (1.0 / FACTOR_19BIT)

#define STEPS_30BIT 1073741824
#define MASK_30BIT 1073741823
#define STEPS_20BIT 1048576
#define MASK_20BIT 1048575
#define STEPS_10BIT 1024
#define MASK_10BIT 1023

struct Batch{
	int state;
	float min_x;
	float min_y;
	float min_z;
	float max_x;
	float max_y;
	float max_z;
	int numPoints;

	int lod_offset;
	int lod_numPoints;
	int padding3;
	int padding4;
	int padding5;
	int padding6;
	int padding7;
	int padding8;
};

struct BoundingBox{
	vec4 position;   //   16   0
	vec4 size;       //   16  16
	uint color;      //    4  32
	                 // size: 48
};

struct Point{
	vec3 position;
	uint32_t color;
};

layout(local_size_x = 128, local_size_y = 1) in;

layout (std430, binding = 0) buffer data_source { uint32_t ssSource[]; };

layout (std430, binding = 30) buffer data_debug {
	uint32_t value;
	uint32_t index;
	float x;
	float y;
	float z;
	uint32_t X;
	uint32_t Y;
	uint32_t Z;
	float min_x;
	float min_y;
	float min_z;
	float size_x;
	float size_y;
	float size_z;
	uint32_t check;
} debug;

layout (std430, binding = 40) buffer data_batches { Batch ssBatches[]; };
layout (std430, binding = 41) buffer data_xyz_12b { float ssXyz_12b[]; };
layout (std430, binding = 42) buffer data_xyz_8b { uint32_t ssXyz_8b[]; };
layout (std430, binding = 43) buffer data_xyz_4b { uint32_t ssXyz_4b[]; };
layout (std430, binding = 44) buffer data_rgba { uint32_t ssRGBA[]; };

layout (std430, binding = 45) buffer data_lod {
	int numPoints;
	uint32_t ssLOD[];
} lod;

layout (std430, binding = 46) buffer data_lodColor {
	uint32_t data[];
} lodColor;


layout(location = 11) uniform int uPointsPerThread;

layout(location = 20) uniform vec3 uBoxMin;
layout(location = 21) uniform vec3 uBoxMax;
layout(location = 22) uniform int uNumPoints;
layout(location = 23) uniform int uNumTotalPoints;
layout(location = 24) uniform int uPointFormat;
layout(location = 25) uniform int uBytesPerPoint;
layout(location = 26) uniform vec3 uScale;
layout(location = 27) uniform dvec3 uOffset;

layout(location = 30) uniform int uBatchOffset;

shared vec3 sgMin[4];
shared vec3 sgMax[4];

#define SAMPLE_GRID_SIZE 4096
shared int wgSampleGrid[SAMPLE_GRID_SIZE];
shared int lodSampleCounter;
shared int lodWritePos;

uint32_t readUint8(uint offset){
	uint ipos = offset / 4;
	uint32_t val_u32 = ssSource[ipos];
	uint32_t shift = 8 * (offset % 4);
	uint32_t val_u8 = (val_u32 >> shift) & 0xFFu;

	return val_u8;
}

uint32_t readUint16(uint offset){
	uint32_t d0 = readUint8(offset + 0);
	uint32_t d1 = readUint8(offset + 1);

	uint32_t value = d0 | (d1 <<  8u);

	return value;
}

int32_t readInt32(uint offset){
	uint32_t d0 = readUint8(offset + 0);
	uint32_t d1 = readUint8(offset + 1);
	uint32_t d2 = readUint8(offset + 2);
	uint32_t d3 = readUint8(offset + 3);

	uint32_t value = d0
		| (d1 <<  8u)
		| (d2 << 16u)
		| (d3 << 24u);

	return int32_t(value);
}

Point getPoint(uint index){

	uint byteOffset = uBytesPerPoint * index;

	int offset_rgb = 0;
	if(uPointFormat == 2){
		offset_rgb = 20;
	}else if(uPointFormat == 3){
		offset_rgb = 28;
	}else if(uPointFormat == 7){
		offset_rgb = 30;
	}else if(uPointFormat == 8){
		offset_rgb = 30;
	}

	int32_t X = readInt32(byteOffset + 0);
	int32_t Y = readInt32(byteOffset + 4);
	int32_t Z = readInt32(byteOffset + 8);
	uint32_t R = readUint16(byteOffset + offset_rgb + 0);
	uint32_t G = readUint16(byteOffset + offset_rgb + 2);
	uint32_t B = readUint16(byteOffset + offset_rgb + 4);
	uint32_t r = R > 255 ? R / 256 : R;
	uint32_t g = G > 255 ? G / 256 : G;
	uint32_t b = B > 255 ? B / 256 : B;

	vec3 size = uBoxMax - uBoxMin;

	// float x = float(X) * uScale.x + float(uOffset.x);
	// float y = float(Y) * uScale.y + float(uOffset.y);
	// float z = float(Z) * uScale.z + float(uOffset.z);

	float x = float(double(X) * uScale.x + uOffset.x - double(uBoxMin.x)); 
	float y = float(double(Y) * uScale.y + uOffset.y - double(uBoxMin.y)); 
	float z = float(double(Z) * uScale.z + uOffset.z - double(uBoxMin.z)); 

	Point point;
	point.position.x = x;
	point.position.y = y;
	point.position.z = z;
	point.color = r | (g << 8) | (b << 16);

	return point;
}

void computeBoundingBox(){

	// initialize bounding boxes for thread and for subgroups
	sgMin[gl_SubgroupID] = vec3(Infinity, Infinity, Infinity);
	sgMax[gl_SubgroupID] = vec3(-Infinity, -Infinity, -Infinity);
	vec3 threadMin = vec3(Infinity, Infinity, Infinity);
	vec3 threadMax = vec3(-Infinity, -Infinity, -Infinity);

	uint localBatchIndex = gl_WorkGroupID.x;
	uint globalBatchIndex = gl_WorkGroupID.x + uBatchOffset;
	uint numPointsPerBatch = uPointsPerThread * gl_WorkGroupSize.x;
	// uint wgFirstPoint = batchIndex * numPointsPerBatch;

	barrier();

	// compute bounding box for current thread
	for(int i = 0; i < uPointsPerThread; i++){

		uint localIndex = localBatchIndex * numPointsPerBatch + i * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
		// uint globalIndex = globalBatchIndex * numPointsPerBatch + i * gl_WorkGroupSize.x + gl_LocalInvocationID.x;

		// if(index > uNumPoints){
		// 	break;
		// }

		Point point = getPoint(localIndex);

		threadMin.x = min(threadMin.x, point.position.x);
		threadMin.y = min(threadMin.y, point.position.y);
		threadMin.z = min(threadMin.z, point.position.z);
		threadMax.x = max(threadMax.x, point.position.x);
		threadMax.y = max(threadMax.y, point.position.y);
		threadMax.z = max(threadMax.z, point.position.z);
	}

	barrier();

	// compute the subgroup bounding box
	sgMin[gl_SubgroupID].x = subgroupMin(threadMin.x);
	sgMin[gl_SubgroupID].y = subgroupMin(threadMin.y);
	sgMin[gl_SubgroupID].z = subgroupMin(threadMin.z);
	sgMax[gl_SubgroupID].x = subgroupMax(threadMax.x);
	sgMax[gl_SubgroupID].y = subgroupMax(threadMax.y);
	sgMax[gl_SubgroupID].z = subgroupMax(threadMax.z);

	barrier();

	// compute the workgroup bounding box
	vec3 wgMin = min(min(sgMin[0], sgMin[1]), min(sgMin[2], sgMin[3]));
	vec3 wgMax = max(max(sgMax[0], sgMax[1]), max(sgMax[2], sgMax[3]));
	// vec3 wgSize = wgMax - wgMin;
	// wgMin = wgMin - 1.0;
	// wgMax = wgMax + 1.0;

	ssBatches[globalBatchIndex].min_x = wgMin.x;
	ssBatches[globalBatchIndex].min_y = wgMin.y;
	ssBatches[globalBatchIndex].min_z = wgMin.z;
	ssBatches[globalBatchIndex].max_x = wgMax.x;
	ssBatches[globalBatchIndex].max_y = wgMax.y;
	ssBatches[globalBatchIndex].max_z = wgMax.z;

}

void processPoints(){

	uint localBatchIndex = gl_WorkGroupID.x;
	uint globalBatchIndex = gl_WorkGroupID.x + uBatchOffset;
	uint numPointsPerBatch = uPointsPerThread * gl_WorkGroupSize.x;

	vec3 wgMin;
	wgMin.x = ssBatches[globalBatchIndex].min_x;
	wgMin.y = ssBatches[globalBatchIndex].min_y;
	wgMin.z = ssBatches[globalBatchIndex].min_z;
	
	vec3 wgMax;
	wgMax.x = ssBatches[globalBatchIndex].max_x;
	wgMax.y = ssBatches[globalBatchIndex].max_y;
	wgMax.z = ssBatches[globalBatchIndex].max_z;


	for(int i = 0; i < uPointsPerThread; i++){

		uint localIndex = localBatchIndex * numPointsPerBatch + i * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
		uint globalIndex = globalBatchIndex * numPointsPerBatch + i * gl_WorkGroupSize.x + gl_LocalInvocationID.x;

		if(localIndex > uNumPoints){
			break;
		}

		Point point = getPoint(localIndex);

		ssRGBA[globalIndex] = point.color;

		vec3 boxSize = wgMax - wgMin;

		uint32_t X = uint32_t(((point.position.x - wgMin.x) / boxSize.x) * STEPS_30BIT);
		uint32_t Y = uint32_t(((point.position.y - wgMin.y) / boxSize.y) * STEPS_30BIT);
		uint32_t Z = uint32_t(((point.position.z - wgMin.z) / boxSize.z) * STEPS_30BIT);

		X = min(X, STEPS_30BIT - 1);
		Y = min(Y, STEPS_30BIT - 1);
		Z = min(Z, STEPS_30BIT - 1);

		{ // 4 byte

			uint32_t X_4b = (X >> 20) & MASK_10BIT;
			uint32_t Y_4b = (Y >> 20) & MASK_10BIT;
			uint32_t Z_4b = (Z >> 20) & MASK_10BIT;

			uint32_t encoded = X_4b | (Y_4b << 10) | (Z_4b << 20);

			ssXyz_4b[globalIndex] = encoded;
		}

		{ // 8 byte

			uint32_t X_8b = (X >> 10) & MASK_10BIT;
			uint32_t Y_8b = (Y >> 10) & MASK_10BIT;
			uint32_t Z_8b = (Z >> 10) & MASK_10BIT;

			uint32_t encoded = X_8b | (Y_8b << 10) | (Z_8b << 20);

			ssXyz_8b[globalIndex] = encoded;
		}
		
		{ // 12 byte

			uint32_t X_12b = (X >> 10) & MASK_10BIT;
			uint32_t Y_12b = (Y >> 10) & MASK_10BIT;
			uint32_t Z_12b = (Z >> 10) & MASK_10BIT;

			uint32_t encoded = X_12b | (Y_12b << 10) | (Z_12b << 20);

			ssXyz_12b[globalIndex] = encoded;
		}



	}

}


// void computeLOD(){
// 	barrier();

// 	uint localBatchIndex = gl_WorkGroupID.x;
// 	uint globalBatchIndex = gl_WorkGroupID.x + uBatchOffset;
// 	uint numPointsPerBatch = uPointsPerThread * gl_WorkGroupSize.x;

// 	vec3 wgMin = vec3(
// 		ssBatches[globalBatchIndex].min_x,
// 		ssBatches[globalBatchIndex].min_y,
// 		ssBatches[globalBatchIndex].min_z
// 	);
	
// 	vec3 wgMax = vec3(
// 		ssBatches[globalBatchIndex].max_x,
// 		ssBatches[globalBatchIndex].max_y,
// 		ssBatches[globalBatchIndex].max_z
// 	);
	
// 	// reset sample grid
// 	int loopSize = SAMPLE_GRID_SIZE / int(gl_WorkGroupSize.x);
// 	for(int i = 0; i < loopSize; i++){
// 		uint index = i * gl_WorkGroupSize.x + gl_LocalInvocationID.x;

// 		wgSampleGrid[index] = -1;
// 	}

// 	barrier();

// 	// iterate over points and store IDs of first point per cell
// 	float TARGET_GRID_SIZE = 16.0;
// 	vec3 wgSize = wgMax - wgMin;
// 	float longest = max(max(wgSize.x, wgSize.y), wgSize.z);
// 	vec3 gridSizes = ceil(TARGET_GRID_SIZE * (wgSize / longest));

// 	if(gl_LocalInvocationID.x == 0){
// 		lodSampleCounter = 0;
// 	}

// 	barrier();

// 	for(int i = 0; i < uPointsPerThread; i++){

// 		uint localIndex = localBatchIndex * numPointsPerBatch + i * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
// 		uint globalIndex = globalBatchIndex * numPointsPerBatch + i * gl_WorkGroupSize.x + gl_LocalInvocationID.x;

// 		if(localIndex > uNumPoints){
// 			break;
// 		}

// 		Point point = getPoint(localIndex);

// 		int X = int(gridSizes.x * (point.position.x - wgMin.x) / wgSize.x);
// 		int Y = int(gridSizes.y * (point.position.y - wgMin.y) / wgSize.y);
// 		int Z = int(gridSizes.z * (point.position.z - wgMin.z) / wgSize.z);

// 		int gridIndex = int(X + Y * gridSizes.x + Z * gridSizes.x * gridSizes.y);

// 		if(gridIndex < SAMPLE_GRID_SIZE){
// 			int old = atomicMax(wgSampleGrid[gridIndex], int(localIndex));
// 			if(old == -1){
// 				atomicAdd(lodSampleCounter, 1);
// 			}
// 		}else{
// 			// TODO - should not happen, but does
// 		}

// 		barrier();
// 	}

// 	barrier();

// 	if(gl_LocalInvocationID.x == 0){
// 		int lodOffset = atomicAdd(lod.numPoints, lodSampleCounter);

// 		ssBatches[globalBatchIndex].lod_offset = lodOffset;
// 		ssBatches[globalBatchIndex].lod_numPoints = lodSampleCounter;
// 	}

// 	// if(lodSampleCounter > 20000){
// 	// 	debug.value = lodSampleCounter;
// 	// }

// 	barrier();

// 	int lodOffset = ssBatches[globalBatchIndex].lod_offset;

// 	// iterate over sample grid and transfer sampled points to buffer
// 	for(int i = 0; i < loopSize; i++){
// 		uint index = i * gl_WorkGroupSize.x + gl_LocalInvocationID.x;

// 		int pointID = wgSampleGrid[index];

// 		if(pointID != -1){
// 			Point point = getPoint(pointID);

// 			int targetOffset = atomicAdd(lodWritePos, 1);
// 			int targetIndex = lodOffset + targetOffset;

// 			{ // 4 byte
// 				vec3 boxSize = wgMax - wgMin;

// 				uint32_t X = uint32_t(((point.position.x - wgMin.x) / boxSize.x) * STEPS_10BIT) & MASK_10BIT;
// 				uint32_t Y = uint32_t(((point.position.y - wgMin.y) / boxSize.y) * STEPS_10BIT) & MASK_10BIT;
// 				uint32_t Z = uint32_t(((point.position.z - wgMin.z) / boxSize.z) * STEPS_10BIT) & MASK_10BIT;

// 				uint32_t encoded = X | (Y << 10) | (Z << 20);

// 				lod.ssLOD[16 + targetIndex] = encoded;
// 				lodColor.data[targetIndex] = point.color;
// 			}

			

// 		}
// 	}

// }

void main(){

	uint localBatchIndex = gl_WorkGroupID.x;
	uint globalBatchIndex = gl_WorkGroupID.x + uBatchOffset;

	lodSampleCounter = 0;
	lodWritePos = 0;

	Batch batch;
	batch.state = 0;
	batch.numPoints = 0;
	batch.lod_numPoints = 0;
	batch.lod_offset = 0;
	ssBatches[globalBatchIndex] = batch;

	barrier();

	computeBoundingBox();

	barrier();

	processPoints();

	// barrier();

	// computeLOD();

}