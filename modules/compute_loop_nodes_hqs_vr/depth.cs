#version 450

// Frustum culling code adapted from three.js
// see https://github.com/mrdoob/three.js/blob/c7d06c02e302ab9c20fe8b33eade4b61c6712654/src/math/Frustum.js
// Three.js license: MIT
// see https://github.com/mrdoob/three.js/blob/a65c32328f7ac4c602079ca51078e3e4fee3123e/LICENSE

#extension GL_ARB_gpu_shader_int64 : enable
#extension GL_NV_shader_atomic_int64 : enable
#extension GL_NV_gpu_shader5 : enable
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_KHR_shader_subgroup_vote : require
#extension GL_KHR_shader_subgroup_clustered : require

#define STEPS_30BIT 1073741824
#define MASK_30BIT 1073741823
#define STEPS_20BIT 1048576
#define MASK_20BIT 1048575
#define STEPS_10BIT 1024
#define MASK_10BIT 1023

#define Infinity (1.0 / 0.0)
layout(local_size_x = 128, local_size_y = 1) in;

struct Batch{
	int state;
	float min_x;
	float min_y;
	float min_z;
	float max_x;
	float max_y;
	float max_z;
	int numPoints;

	int pointOffset;
	int level;
	int padding1;
	int padding2;
	int padding3;
	int padding4;
	int padding5;
	int padding6;
};

struct Point{
	float x;
	float y;
	float z;
	uint32_t color;
};

struct BoundingBox{
	vec4 position;   //   16   0
	vec4 size;       //   16  16
	uint color;      //    4  32
	                 // size: 48
};

layout (std430, binding =  1) buffer abc_0 { uint32_t ssLeft_depth[]; };
layout (std430, binding =  2) buffer abc_1 { uint32_t ssLeft_rgba[]; };
layout (std430, binding =  3) buffer abc_2 { uint32_t ssRight_depth[]; };
layout (std430, binding =  4) buffer abc_3 { uint32_t ssRight_rgba[]; };
layout (std430, binding = 11) buffer abc_4 { Batch ssBatches[]; };
layout (std430, binding = 41) buffer abc_6 { uint32_t ssXyz_12b[]; };
layout (std430, binding = 42) buffer abc_7 { uint32_t ssXyz_8b[]; };
layout (std430, binding = 43) buffer abc_8 { uint32_t ssXyz_4b[]; };
layout (std430, binding = 44) buffer abc_9 { uint32_t ssRGBA[]; };

layout (std430, binding = 30) buffer abc_10 { 
	uint32_t value;
	bool enabled;
	uint32_t depth_numPointsProcessed;
	uint32_t depth_numNodesProcessed;
	uint32_t depth_numPointsRendered;
	uint32_t depth_numNodesRendered;
	uint32_t color_numPointsProcessed;
	uint32_t color_numNodesProcessed;
	uint32_t color_numPointsRendered;
	uint32_t color_numNodesRendered;
	uint32_t numPointsVisible;
} debug;

layout (std430, binding = 50) buffer data_bb {
	uint count;
	uint instanceCount;
	uint first;
	uint baseInstance;
	uint pad0; uint pad1; uint pad2; uint pad3;
	uint pad4; uint pad5; uint pad6; uint pad7;
	// 48
	BoundingBox ssBoxes[];
} boundingBoxes;

layout(std140, binding = 31) uniform UniformData{
	mat4 left_world;
	mat4 left_view;
	mat4 left_proj;
	mat4 left_transform;
	mat4 left_transformFrustum;
	mat4 right_world;
	mat4 right_view;
	mat4 right_proj;
	mat4 right_transform;
	mat4 right_transformFrustum;
	int pointsPerThread;
	int enableFrustumCulling;
	int showBoundingBox;
	int numPoints;
	ivec2 imageSize;
} uniforms;

uint SPECTRAL[5] = {
	0x00ba832b,
	0x00a4ddab,
	0x00bfffff,
	0x0061aefd,
	0x001c19d7
};

struct Plane{
	vec3 normal;
	float constant;
};

float t(int index){

	int a = index % 4;
	int b = index / 4;

	return uniforms.left_transformFrustum[b][a];
}

float t_right(int index){

	int a = index % 4;
	int b = index / 4;

	return uniforms.right_transformFrustum[b][a];
}


float distanceToPoint(vec3 point, Plane plane){
	return dot(plane.normal, point) + plane.constant;
}

Plane createPlane(float x, float y, float z, float w){

	float nLength = length(vec3(x, y, z));

	Plane plane;
	plane.normal = vec3(x, y, z) / nLength;
	plane.constant = w / nLength;

	return plane;
}

Plane[6] frustumPlanes(){
	Plane planes[6] = {
		createPlane(t_right( 3) - t_right(0), t_right( 7) - t_right(4), t_right(11) - t_right( 8), t_right(15) - t_right(12)),
		createPlane(t( 3) + t(0), t( 7) + t(4), t(11) + t( 8), t(15) + t(12)),
		createPlane(t( 3) + t(1), t( 7) + t(5), t(11) + t( 9), t(15) + t(13)),
		createPlane(t( 3) - t(1), t( 7) - t(5), t(11) - t( 9), t(15) - t(13)),
		createPlane(t( 3) - t(2), t( 7) - t(6), t(11) - t(10), t(15) - t(14)),
		createPlane(t( 3) + t(2), t( 7) + t(6), t(11) + t(10), t(15) + t(14)),
	};

	return planes;
}

bool intersectsFrustum(vec3 wgMin, vec3 wgMax){

	Plane[] planes = frustumPlanes();
	
	for(int i = 0; i < 6; i++){

		Plane plane = planes[i];

		vec3 vector;
		vector.x = plane.normal.x > 0.0 ? wgMax.x : wgMin.x;
		vector.y = plane.normal.y > 0.0 ? wgMax.y : wgMin.y;
		vector.z = plane.normal.z > 0.0 ? wgMax.z : wgMin.z;

		float d = distanceToPoint(vector, plane);

		if(d < 0){
			return false;
		}
	}

	return true;
}

float getWeight(vec3 pos){
	mat4 worldView = uniforms.left_view * uniforms.left_world;

	vec4 viewCenter_l = uniforms.left_view * uniforms.left_world * vec4(pos, 1.0);
	vec4 viewCenter_r = uniforms.right_view * uniforms.right_world * vec4(pos, 1.0);
	vec4 viewCenter = (viewCenter_l + viewCenter_r) / 2.0;
	viewCenter = viewCenter_r;

	vec4 projCenter_l = uniforms.left_proj * viewCenter;
	vec4 projCenter_r = uniforms.right_proj * viewCenter;
	vec4 projCenter = (projCenter_l + projCenter_r) / 2.0;

	projCenter.xy = projCenter.xy / projCenter.w;

	float d_screen = length(projCenter.xy);
	float w_screen = exp(- (d_screen * d_screen) / 0.20);
	w_screen = w_screen + 0.1;
	w_screen = clamp(w_screen * 0.8, 0.0, 1.0);

	return w_screen;
}

int getPrecisionLevel(vec3 wgMin, vec3 wgMax){
	
	mat4 worldView = uniforms.left_view * uniforms.left_world;
	vec3 wgCenter = (wgMin + wgMax) / 2.0;
	float wgRadius = distance((worldView * vec4(wgMin, 1.0)).xyz, (worldView * vec4(wgMax, 1.0)).xyz);

	vec4 viewCenter_l = uniforms.left_view * uniforms.left_world * vec4(wgCenter, 1.0);
	vec4 viewCenter_r = uniforms.right_view * uniforms.left_world * vec4(wgCenter, 1.0);
	vec4 viewCenter = (viewCenter_l + viewCenter_r) / 2.0;
	vec4 viewEdge = viewCenter + vec4(wgRadius, 0.0, 0.0, 0.0);

	vec4 projCenter = uniforms.left_proj * viewCenter;
	vec4 projEdge = uniforms.left_proj * viewEdge;

	projCenter.xy = projCenter.xy / projCenter.w;
	projEdge.xy = projEdge.xy / projEdge.w;

	float w_depth = distance(projCenter.xy, projEdge.xy);
	float d_screen = length(projCenter.xy);
	float w_screen = exp(- (d_screen * d_screen) / 1.0);
	w_screen = getWeight((wgMin + wgMax) / 2.0);

	float w = w_depth * w_screen;

	int level = 0;
	if(w < 0.01){
		level = 4;
	}else if(w < 0.02){
		level = 3;
	}else if(w < 0.05){
		level = 2;
	}else if(w < 0.1){
		level = 1;
	}

	return level;
}

void main(){


	uint batchIndex = gl_WorkGroupID.x;
	uint numPointsPerBatch = uniforms.pointsPerThread * gl_WorkGroupSize.x;

	Batch batch = ssBatches[batchIndex];

	if(debug.enabled && gl_LocalInvocationID.x == 0){
		atomicAdd(debug.depth_numNodesProcessed, 1);
	}

	uint wgFirstPoint = batch.pointOffset;

	vec3 wgMin = vec3(batch.min_x, batch.min_y, batch.min_z);
	vec3 wgMax = vec3(batch.max_x, batch.max_y, batch.max_z);
	vec3 boxSize = wgMax - wgMin;

	// FRUSTUM CULLING
	if((uniforms.enableFrustumCulling != 0) && !intersectsFrustum(wgMin, wgMax)){
		return;
	}

	int level = getPrecisionLevel(wgMin, wgMax);

	if(level >= 2){
		return;
	}

	if(debug.enabled && gl_LocalInvocationID.x == 0){
		atomicAdd(debug.depth_numNodesRendered, 1);
	}

	uint batchSize = batch.numPoints;
	uint loopSize = uint(ceil(float(batchSize) / float(gl_WorkGroupSize.x)));
	loopSize = min(loopSize, 500);

	for(int i = 0; i < loopSize; i++){

		uint index = wgFirstPoint + i * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
		uint localIndex = i * gl_WorkGroupSize.x + gl_LocalInvocationID.x;

		if(localIndex >= batch.numPoints){
			return;
		}

		if(debug.enabled){
			atomicAdd(debug.depth_numPointsProcessed, 1);
		}

		vec3 point;

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
		}else if(level > 1){ 
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

		vec4 pos = vec4(point.x, point.y, point.z, 1.0);

		vec4 posLeft = uniforms.left_transform * pos;
		vec4 posRight = uniforms.right_transform * pos;

		posLeft.xyz = posLeft.xyz / posLeft.w;
		posRight.xyz = posRight.xyz / posRight.w;

		bool leftInFrustum = !(posLeft.w <= 0.0 || posLeft.x < -1.0 || posLeft.x > 1.0 || posLeft.y < -1.0 || posLeft.y > 1.0);
		bool rightInFrustum = !(posRight.w <= 0.0 || posRight.x < -1.0 || posRight.x > 1.0 || posRight.y < -1.0 || posRight.y > 1.0);

		if(leftInFrustum){
			vec2 imgPos = (posLeft.xy * 0.5 + 0.5) * uniforms.imageSize;
			ivec2 pixelCoords = ivec2(imgPos);
			int pixelID = pixelCoords.x + pixelCoords.y * uniforms.imageSize.x;

			uint32_t newDepth = floatBitsToInt(posLeft.w);
			uint32_t oldDepth = ssLeft_depth[pixelID];

			if(newDepth < oldDepth){
				atomicMin(ssLeft_depth[pixelID], newDepth);

				if(debug.enabled){
					atomicAdd(debug.depth_numPointsRendered, 1);
				}
			}
		}

		if(rightInFrustum)
		{
			vec2 imgPos = (posRight.xy * 0.5 + 0.5) * uniforms.imageSize;
			ivec2 pixelCoords = ivec2(imgPos);
			int pixelID = pixelCoords.x + pixelCoords.y * uniforms.imageSize.x;

			uint32_t newDepth = floatBitsToInt(posRight.w);
			uint32_t oldDepth = ssRight_depth[pixelID];

			if(newDepth < oldDepth){
				atomicMin(ssRight_depth[pixelID], newDepth);

				if(debug.enabled){
					atomicAdd(debug.depth_numPointsRendered, 1);
				}
			}
		}

	}

}