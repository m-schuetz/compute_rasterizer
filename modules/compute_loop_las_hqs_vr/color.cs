#version 450

#extension GL_NV_shader_atomic_int64 : enable
#extension GL_NV_gpu_shader5 : enable
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_NV_shader_atomic_float : require
// #extension GL_EXT_shader_atomic_float2 : require
#extension GL_KHR_shader_subgroup_vote : enable
#extension GL_NV_shader_subgroup_partitioned : require
#extension GL_KHR_shader_subgroup_ballot : enable
#extension GL_NV_gpu_shader_fp64 : enable

#define STATE_UNINITIALIZED 0
#define STATE_INITIALIZED 1

#define BUFFER_POSITION 0
#define BUFFER_COLOR 1

#define VIEW_LEFT_INDEX 0
#define VIEW_RIGHT_INDEX 0

#define SCALE 0.001
#define FACTOR_20BIT 1048576.0
#define FACTOR_19BIT 524288.0
#define FACTOR_9BIT 512.0
#define SCALE_20BIT (1.0 / FACTOR_20BIT)
#define SCALE_19BIT (1.0 / FACTOR_19BIT)
#define SCALE_9BIT (1.0 / FACTOR_9BIT)

#define STEPS_30BIT 1073741824
#define MASK_30BIT 1073741823
#define STEPS_20BIT 1048576
#define MASK_20BIT 1048575
#define STEPS_10BIT 1024
#define MASK_10BIT 1023

#define Infinity (1.0 / 0.0)

struct Batch{
	int state;
	float min_x;
	float min_y;
	float min_z;
	float max_x;
	float max_y;
	float max_z;
	
	int numPoints;
	int padding1;
	int padding2;
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

struct Brush{
	vec3 position;
	float radius;
};

layout(local_size_x = 128, local_size_y = 1) in;

layout (std430, binding = 1) buffer framebuffer_left_depth {
	uint32_t ssLeft_depth[];
};

layout (std430, binding = 2) buffer framebuffer_left_color {
	uint64_t ssLeft_rgba[];
};

layout (std430, binding = 3) buffer framebuffer_right_depth {
	uint32_t ssRight_depth[];
};

layout (std430, binding = 4) buffer framebuffer_right_color {
	uint64_t ssRight_rgba[];
};

layout (std430, binding = 30) buffer dbg_buffer {
	uint32_t ssDebug[];
};

layout (std430, binding = 40) buffer data_batches {
	Batch ssBatches[];
};

layout (std430, binding = 41) buffer data_xyz_12b {
	uint32_t ssXyz_12b[];
};

layout (std430, binding = 42) buffer data_xyz_compressed_8b {
	uint32_t ssXyz_8b[];
};

layout (std430, binding = 43) buffer data_xyz_compressed_4b {
	uint32_t ssXyz_4b[];
};

layout (std430, binding = 44) buffer data_rgba {
	uint32_t ssRGBA[];
};

layout (std430, binding = 50) buffer data_bb {
	uint count;
	uint instanceCount;
	uint first;
	uint baseInstance;
	// 16
	uint pad0;
	uint pad1;
	uint pad2;
	uint pad3;
	// 32
	uint pad4;
	uint pad5;
	uint pad6;
	uint pad7;
	// 48
	BoundingBox ssBoxes[];
} boundingBoxes;

layout (std430, binding = 51) buffer data_selection {
	uint32_t ssSelection[];
};

layout(location = 0) uniform mat4 uLeftTransform;
layout(location = 1) uniform mat4 uLeftTransformFrustum;
layout(location = 2) uniform mat4 uLeftWorldView;
layout(location = 3) uniform mat4 uLeftProj;

layout(location = 4) uniform mat4 uRightTransform;
layout(location = 5) uniform mat4 uRightTransformFrustum;
layout(location = 6) uniform mat4 uRightWorldView;
layout(location = 7) uniform mat4 uRightProj;

layout(location =  9) uniform vec3 uCamPos;
layout(location = 10) uniform ivec2 uImageSize;
layout(location = 11) uniform int uPointsPerThread;
layout(location = 12) uniform int uEnableFrustumCulling;
layout(location = 13) uniform int uShowBoundingBox;

layout(location = 20) uniform vec3 uBoxMin;
layout(location = 21) uniform vec3 uBoxMax;
layout(location = 22) uniform int uNumPoints;
layout(location = 23) uniform int64_t uOffsetToPointData;
layout(location = 24) uniform int uPointFormat;
layout(location = 25) uniform int64_t uBytesPerPoint;
layout(location = 26) uniform vec3 uScale;
layout(location = 27) uniform dvec3 uOffset;

layout(location = 30) uniform vec3 uBrushPos;
layout(location = 31) uniform float uBrushSize;
layout(location = 32) uniform int uBrushIsActive;


uint SPECTRAL[5] = {
	0x00ba832b,
	0x00a4ddab,
	0x00bfffff,
	0x0061aefd,
	0x001c19d7
};

bool isInsideFrustum(int viewIndex, vec3 point){

	mat4 transform;
	if(viewIndex == 0){
		transform = uLeftTransformFrustum;
	}else{
		transform = uRightTransformFrustum;
	}

	vec4 pos = transform * vec4(point, 1.0);
	pos.xyz = pos.xyz / pos.w;

	if(pos.w <= 0.0 || pos.x < -1.4 || pos.x > 1.2 || pos.y < -1.2 || pos.y > 1.2){
		return false;
	}else{
		return true;
	}

}


void main(){
	
	uint batchIndex = gl_WorkGroupID.x;
	uint numPointsPerBatch = uPointsPerThread * gl_WorkGroupSize.x;
	uint wgFirstPoint = batchIndex * numPointsPerBatch;

	vec3 bbSize = uBoxMax - uBoxMin;

	Batch batch = ssBatches[batchIndex];

	int level = 0;

	// Frustum and LOD culling
	// if(false)
	{ 
		vec3 wgMin = vec3(batch.min_x, batch.min_y, batch.min_z);
		vec3 wgMax = vec3(batch.max_x, batch.max_y, batch.max_z);

		// FRUSTUM CULLING
		if((uEnableFrustumCulling != 0) && (!isInsideFrustum(VIEW_LEFT_INDEX, wgMin) && !isInsideFrustum(VIEW_LEFT_INDEX, wgMax))){
			return;
		}


		// LOD CULLING
		vec3 wgCenter = (wgMin + wgMax) / 2.0;
		float wgRadius = distance((uLeftWorldView * vec4(wgMin, 1.0)).xyz, (uLeftWorldView * vec4(wgMax, 1.0)).xyz);

		vec4 viewCenter = uLeftWorldView * vec4(wgCenter, 1.0);
		vec4 viewEdge = viewCenter + vec4(wgRadius, 0.0, 0.0, 0.0);

		vec4 projCenter = uLeftProj * viewCenter;
		vec4 projEdge = uLeftProj * viewEdge;

		projCenter.xy = projCenter.xy / projCenter.w;
		projEdge.xy = projEdge.xy / projEdge.w;

		float w_depth = distance(projCenter.xy, projEdge.xy);

		float d_screen = length(projCenter.xy);
		float w_screen = exp(- (d_screen * d_screen) / 1.0);

		float w = w_depth * w_screen;

		if(w < 0.01){
			level = 4;
		}else if(w < 0.02){
			level = 3;
		}else if(w < 0.05){
			level = 2;
		}else if(w < 0.1){
		// if((uEnableLOD != 0) && (w < uLOD * 0.01)){
			// return;
			level = 1;
		}

	}

	int loopSize = uPointsPerThread;

	if((uShowBoundingBox != 0) && gl_LocalInvocationID.x == 0)
	// if((uShowBoundingBox != 0) && gl_LocalInvocationID.x == 0)
	// if(gl_LocalInvocationID.x == 0 && (batchIndex % 10 == 0))
	{ // bounding boxes
		uint boxIndex = atomicAdd(boundingBoxes.instanceCount, 1);
		// boundingBoxes.instanceCount = 1;

		// uint boxIndex = 0;
		boundingBoxes.count = 24;
		boundingBoxes.first = 0;
		boundingBoxes.baseInstance = 0;

		vec3 wgMin = vec3(batch.min_x, batch.min_y, batch.min_z);
		vec3 wgMax = vec3(batch.max_x, batch.max_y, batch.max_z);
		vec3 wgPos = (wgMin + wgMax) / 2.0;
		vec3 wgSize = wgMax - wgMin;

		uint color = 0x0000FF00;
		if(level > 0){
			color = 0x000000FF;
		}

		color = SPECTRAL[level];

		BoundingBox box;
		box.position = vec4(wgPos, 0.0);
		box.size = vec4(wgSize, 0.0);
		box.color = color;


		boundingBoxes.ssBoxes[boxIndex] = box;
	}


	for(int i = 0; i < loopSize; i++){

		uint index = wgFirstPoint + i * gl_WorkGroupSize.x + gl_LocalInvocationID.x;

		if(index > uNumPoints){
			return;
		}

		vec3 wgMin = vec3(batch.min_x, batch.min_y, batch.min_z);
		vec3 wgMax = vec3(batch.max_x, batch.max_y, batch.max_z);
		vec3 boxSize = wgMax - wgMin;

		vec3 point;
		
		// - to reduce memory bandwidth, we load lower precision coordinates for smaller screen-size bounding boxes
		// - coordinates are stored in up to three 4 byte values
		// - A single 4 byte value holds 3 x 10 bit integer components (x, y, z)
		// - if more precision is required, we can load another 4 byte with additional 3 x 10 bit precision
		// - 10 bits grants us 1024 individual values => we can present 1 meter in millimeter precision
		//   20 bits: 1km in mm precsision
		//   30 bits: 1000km in mm precision (but not really because we are loosing precision due to float conversions)
		if(level == 0){
			// 12 byte ( 30 bit per axis)
			uint32_t b4 = ssXyz_4b[index];
			uint32_t b8 = ssXyz_8b[index];
			uint32_t b12 = ssXyz_12b[index];

			uint32_t X_4 = (b4 >>  0) & MASK_10BIT;
			uint32_t Y_4 = (b4 >> 10) & MASK_10BIT;
			uint32_t Z_4 = (b4 >> 20) & MASK_10BIT;

			uint32_t X_8 = (b8 >>  0) & MASK_10BIT;
			uint32_t Y_8 = (b8 >> 10) & MASK_10BIT;
			uint32_t Z_8 = (b8 >> 20) & MASK_10BIT;

			uint32_t X_12 = (b12 >>  0) & MASK_10BIT;
			uint32_t Y_12 = (b12 >> 10) & MASK_10BIT;
			uint32_t Z_12 = (b12 >> 20) & MASK_10BIT;

			uint32_t X = (X_4 << 20) | (X_8 << 10) | X_12;
			uint32_t Y = (Y_4 << 20) | (Y_8 << 10) | X_12;
			uint32_t Z = (Z_4 << 20) | (Z_8 << 10) | X_12;

			float x = float(X) * (boxSize.x / STEPS_30BIT) + wgMin.x;
			float y = float(Y) * (boxSize.y / STEPS_30BIT) + wgMin.y;
			float z = float(Z) * (boxSize.z / STEPS_30BIT) + wgMin.z;

			point = vec3(x, y, z);
		}else if(level == 1){ 
			// 8 byte (20 bits per axis)

			uint32_t b4 = ssXyz_4b[index];
			uint32_t b8 = ssXyz_8b[index];

			uint32_t X_4 = (b4 >>  0) & MASK_10BIT;
			uint32_t Y_4 = (b4 >> 10) & MASK_10BIT;
			uint32_t Z_4 = (b4 >> 20) & MASK_10BIT;

			uint32_t X_8 = (b8 >>  0) & MASK_10BIT;
			uint32_t Y_8 = (b8 >> 10) & MASK_10BIT;
			uint32_t Z_8 = (b8 >> 20) & MASK_10BIT;

			uint32_t X = (X_4 << 20) | (X_8 << 10);
			uint32_t Y = (Y_4 << 20) | (Y_8 << 10);
			uint32_t Z = (Z_4 << 20) | (Z_8 << 10);

			float x = float(X) * (boxSize.x / STEPS_30BIT) + wgMin.x;
			float y = float(Y) * (boxSize.y / STEPS_30BIT) + wgMin.y;
			float z = float(Z) * (boxSize.z / STEPS_30BIT) + wgMin.z;

			point = vec3(x, y, z);
		}else{ 
			// 4 byte (10 bits per axis)

			uint32_t encoded = ssXyz_4b[index];

			uint32_t X = (encoded >>  0) & MASK_10BIT;
			uint32_t Y = (encoded >> 10) & MASK_10BIT;
			uint32_t Z = (encoded >> 20) & MASK_10BIT;

			float x = float(X) * (boxSize.x / STEPS_10BIT) + wgMin.x;
			float y = float(Y) * (boxSize.y / STEPS_10BIT) + wgMin.y;
			float z = float(Z) * (boxSize.z / STEPS_10BIT) + wgMin.z;

			point = vec3(x, y, z);
		}
		
		// now project to screen
		vec4 pos = vec4(point, 1.0);

		bool isInsideBrush = false;
		{
			vec3 diff = uBrushPos - point;
			if(length(diff) < 1.0 * uBrushSize){
				isInsideBrush = true;
			}

			if(isInsideBrush && (uBrushIsActive != 0)){
				ssSelection[index] = 1;
			}
		}

		bool isSelected = (ssSelection[index] == 1);
		// bool isSelected = false;

		if(isSelected){
			continue;
		}

		vec4 posLeft = uLeftTransform * pos;
		vec4 posRight = uRightTransform * pos;

		posLeft.xyz = posLeft.xyz / posLeft.w;
		posRight.xyz = posRight.xyz / posRight.w;

		bool leftInFrustum = !(posLeft.w <= 0.0 || posLeft.x < -1.0 || posLeft.x > 1.0 || posLeft.y < -1.0 || posLeft.y > 1.0);
		bool rightInFrustum = !(posRight.w <= 0.0 || posRight.x < -1.0 || posRight.x > 1.0 || posRight.y < -1.0 || posRight.y > 1.0);

		barrier();

		if(leftInFrustum){
			vec2 imgPos = (posLeft.xy * 0.5 + 0.5) * uImageSize;
			ivec2 pixelCoords = ivec2(imgPos);
			int pixelID = pixelCoords.x + pixelCoords.y * uImageSize.x;

			float depth = posLeft.w;
			float bufferDepth = uintBitsToFloat(ssLeft_depth[pixelID]);

			// average points within 1% distance
			bool visible = (depth <= bufferDepth * 1.01);

			if(visible){

				uvec4 ballot = subgroupPartitionNV(pixelID);
				uint32_t ballotLeaderID = subgroupPartitionedMinNV(gl_SubgroupInvocationID.x, ballot);
				bool isBallotLeader = (ballotLeaderID == gl_SubgroupInvocationID.x);

				uint32_t color = ssRGBA[index];
				uint32_t R = (color >>  0) & 0xFF;
				uint32_t G = (color >>  8) & 0xFF;
				uint32_t B = (color >> 16) & 0xFF;

				if(isInsideBrush){
					R = min((R * 4) / 3, 255);
					G = (G * 2) / 4;
					B = (B * 2) / 4;
				}

				uvec4 rgbc = uvec4(R, G, B, 1);

				uvec4 ballot_rgbc = subgroupPartitionedAddNV(rgbc, ballot);
				uint32_t ballot_R = ballot_rgbc.x;
				uint32_t ballot_G = ballot_rgbc.y;
				uint32_t ballot_B = ballot_rgbc.z;
				uint32_t ballot_count = ballot_rgbc.w;

				if(isBallotLeader){
					int64_t RG = (int64_t(ballot_R) << 0) | (int64_t(ballot_G) << 32);
					int64_t BA = (int64_t(ballot_B) << 0) | (int64_t(ballot_count) << 32);

					atomicAdd(ssLeft_rgba[2 * pixelID + 0], RG);
					atomicAdd(ssLeft_rgba[2 * pixelID + 1], BA);
				}

			}
		}

		barrier();

		if(rightInFrustum){
			vec2 imgPos = (posRight.xy * 0.5 + 0.5) * uImageSize;
			ivec2 pixelCoords = ivec2(imgPos);
			int pixelID = pixelCoords.x + pixelCoords.y * uImageSize.x;

			float depth = posRight.w;
			float bufferDepth = uintBitsToFloat(ssRight_depth[pixelID]);

			// average points within 1% distance
			bool visible = (depth <= bufferDepth * 1.01);
			// visible = true;

			if(visible){

				uvec4 ballot = subgroupPartitionNV(pixelID);
				uint32_t ballotLeaderID = subgroupPartitionedMinNV(gl_SubgroupInvocationID.x, ballot);
				bool isBallotLeader = (ballotLeaderID == gl_SubgroupInvocationID.x);

				uint32_t color = ssRGBA[index];
				uint32_t R = (color >>  0) & 0xFF;
				uint32_t G = (color >>  8) & 0xFF;
				uint32_t B = (color >> 16) & 0xFF;

				if(isInsideBrush){
					R = min((R * 4) / 3, 255);
					G = (G * 2) / 4;
					B = (B * 2) / 4;
				}

				uvec4 rgbc = uvec4(R, G, B, 1);

				uvec4 ballot_rgbc = subgroupPartitionedAddNV(rgbc, ballot);
				uint32_t ballot_R = ballot_rgbc.x;
				uint32_t ballot_G = ballot_rgbc.y;
				uint32_t ballot_B = ballot_rgbc.z;
				uint32_t ballot_count = ballot_rgbc.w;

				// ballot_R = 200;
				// ballot_G =   0;
				// ballot_B =   0;
				// ballot_count = 1;

				if(isBallotLeader){
					int64_t RG = (int64_t(ballot_R) << 0) | (int64_t(ballot_G) << 32);
					int64_t BA = (int64_t(ballot_B) << 0) | (int64_t(ballot_count) << 32);

					atomicAdd(ssRight_rgba[2 * pixelID + 0], RG);
					atomicAdd(ssRight_rgba[2 * pixelID + 1], BA);
				}

			}
		}

		// barrier();
	}

}