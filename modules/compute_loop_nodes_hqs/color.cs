#version 450

// Frustum culling code adapted from three.js
// see https://github.com/mrdoob/three.js/blob/c7d06c02e302ab9c20fe8b33eade4b61c6712654/src/math/Frustum.js
// Three.js license: MIT
// see https://github.com/mrdoob/three.js/blob/a65c32328f7ac4c602079ca51078e3e4fee3123e/LICENSE

#extension GL_NV_shader_atomic_int64 : enable
#extension GL_NV_gpu_shader5 : enable
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_NV_shader_atomic_float : require
#extension GL_KHR_shader_subgroup_vote : enable
#extension GL_NV_shader_subgroup_partitioned : require
#extension GL_KHR_shader_subgroup_ballot : enable
#extension GL_NV_gpu_shader_fp64 : enable

#define STEPS_30BIT 1073741824
#define MASK_30BIT 1073741823
#define STEPS_20BIT 1048576
#define MASK_20BIT 1048575
#define STEPS_10BIT 1024
#define MASK_10BIT 1023

#define Infinity (1.0 / 0.0)

#define SELECTION_NONE 0
#define SELECTION_INTERSECTS 1
#define SELECTION_INSIDE 2

#define SELECT_INACTIVE 0
#define SELECT_ACTIVE 1

#define SELECTION_RADIUS 50

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
	int selection;
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

layout (std430, binding =  1) buffer abc_0 { uint32_t ssDepth[];     };
layout (std430, binding =  2) buffer abc_1 { uint64_t ssColor[];     };
layout (std430, binding = 11) buffer abc_2 { Batch ssBatches[];      };
layout (std430, binding = 41) buffer abc_4 { uint32_t ssXyz_12b[];   };
layout (std430, binding = 42) buffer abc_5 { uint32_t ssXyz_8b[];    };
layout (std430, binding = 43) buffer abc_6 { uint32_t ssXyz_4b[];    };
layout (std430, binding = 44) buffer abc_7 { uint32_t ssRGBA[];      };
layout (std430, binding = 45) buffer abc_8 { uint32_t ssSelection[]; };

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

layout (std430, binding = 50) buffer abc_9 {
	uint count;
	uint instanceCount;
	uint first;
	uint baseInstance;
	uint pad0; uint pad1; uint pad2; uint pad3;
	uint pad4; uint pad5; uint pad6; uint pad7;
	// 48
	BoundingBox ssBoxes[];
} boundingBoxes;

layout(location = 11) uniform ivec2 uMousePosition;
layout(location = 12) uniform int uSelectState;
layout(location = 13) uniform int uMouseButtons;
layout(location = 14) uniform float uFovY;

layout(std140, binding = 31) uniform UniformData{
	mat4 world;
	mat4 view;
	mat4 proj;
	mat4 transform;
	mat4 transformFrustum;
	int pointsPerThread;
	int enableFrustumCulling;
	int showBoundingBox;
	int numPoints;
	ivec2 imageSize;
	bool colorizeChunks;
	bool colorizeOverdraw;
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

	return uniforms.transformFrustum[b][a];
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
		createPlane(t( 3) - t(0), t( 7) - t(4), t(11) - t( 8), t(15) - t(12)),
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

int getPrecisionLevel(vec3 wgMin, vec3 wgMax){
	
	vec3 wgCenter = (wgMin + wgMax) / 2.0;
	float wgRadius = distance(wgMin, wgMax);

	vec4 viewCenter = uniforms.view * uniforms.world * vec4(wgCenter, 1.0);
	vec4 viewEdge = viewCenter + vec4(wgRadius, 0.0, 0.0, 0.0);

	vec4 projCenter = uniforms.proj * viewCenter;
	vec4 projEdge = uniforms.proj * viewEdge;

	projCenter.xy = projCenter.xy / projCenter.w;
	projEdge.xy = projEdge.xy / projEdge.w;

	vec2 screenCenter = uniforms.imageSize.xy * (projCenter.xy + 1.0) / 2.0;
	vec2 screenEdge = uniforms.imageSize.xy * (projEdge.xy + 1.0) / 2.0;
	float pixelSize = distance(screenEdge, screenCenter);

	int level = 0;
	if(pixelSize < 80){
		level = 4;
	}else if(pixelSize < 200){
		level = 3;
	}else if(pixelSize < 500){
		level = 2;
	}else if(pixelSize < 10000){
		level = 1;
	}else{
		level = 0;
	}

	return level;
}

struct Box{
	vec3 min;
	vec3 max;
};

struct Ray{
	vec3 orig;
	vec3 dir;
};

bool intersect(Box box, Ray r){

	vec3 min = box.min;
	vec3 max = box.max;

	float tmin = (min.x - r.orig.x) / r.dir.x; 
	float tmax = (max.x - r.orig.x) / r.dir.x; 

	if (tmin > tmax){
		float tmp = tmin;
		tmin = tmax;
		tmax = tmp;
	}

	float tymin = (min.y - r.orig.y) / r.dir.y; 
	float tymax = (max.y - r.orig.y) / r.dir.y; 

	if (tymin > tymax){
		float tmp = tymin;
		tymin = tymax;
		tymax = tmp;
	}

	if ((tmin > tymax) || (tymin > tmax)) 
		return false; 

	if (tymin > tmin) 
		tmin = tymin; 

	if (tymax < tmax) 
		tmax = tymax; 

	float tzmin = (min.z - r.orig.z) / r.dir.z; 
	float tzmax = (max.z - r.orig.z) / r.dir.z; 

	if (tzmin > tzmax){ 
		float tmp = tzmin;
		tzmin = tzmax;
		tzmax = tmp;
	}

	if ((tmin > tzmax) || (tzmin > tmax)) 
		return false; 

	if (tzmin > tmin) 
		tmin = tzmin; 

	if (tzmax < tmax) 
		tmax = tzmax; 

	return true; 
}

int intersectMouse(vec3 point, float dist){
	vec4 projected = uniforms.transform * vec4(point, 1.0);
	vec3 ndc = projected.xyz / projected.w;
	vec2 imgPos = (ndc.xy * 0.5 + 0.5) * uniforms.imageSize;
	ivec2 pixelCoords = ivec2(imgPos);
	pixelCoords.y = uniforms.imageSize.y - pixelCoords.y;
	float d = distance(pixelCoords, uMousePosition);

	return (d < dist) ? 1 : 0;
}

bool intersect(vec3 point, vec3 spherePos, float sphereRadius){
	return length(point - spherePos) < sphereRadius;
}

void main(){
	
	uint batchIndex = gl_WorkGroupID.x;
	uint numPointsPerBatch = uniforms.pointsPerThread * gl_WorkGroupSize.x;

	Batch batch = ssBatches[batchIndex];

	if(debug.enabled && gl_LocalInvocationID.x == 0){
		atomicAdd(debug.color_numNodesProcessed, 1);
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

	if(level >= 4){
		return;
	}

	// POPULATE BOUNDING BOX BUFFER, if enabled
	if((uniforms.showBoundingBox != 0) && gl_LocalInvocationID.x == 0){ 
		uint boxIndex = atomicAdd(boundingBoxes.instanceCount, 1);

		boundingBoxes.count = 24;
		boundingBoxes.first = 0;
		boundingBoxes.baseInstance = 0;

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

	bool doHighlighting = (uSelectState == SELECT_ACTIVE);
	bool doSelection = (uSelectState == SELECT_ACTIVE && uMouseButtons == 1);
	float selectionDepth = 100000000.0;
	uvec4 dbg = uvec4(0, 0, 0, 0);
	int selectionState = SELECTION_NONE;

	if(doHighlighting){
		// compute reference depth: closest depth within pixel window around mouse

		int window = 5;

		for(int ox = -window; ox <= window; ox++){
		for(int oy = -window; oy <= window; oy++){

			ivec2 mousePos = ivec2(
				uMousePosition.x,
				uniforms.imageSize.y - uMousePosition.y
			);
			int pixelID = (mousePos.x + ox) + (mousePos.y + oy) * uniforms.imageSize.x;

			float bufferDepth = uintBitsToFloat(ssDepth[pixelID]);

			if(bufferDepth > 0.0){
				selectionDepth = min(bufferDepth, selectionDepth);
			}

		}
		}
	}

	{
		ivec2 pixelCoords = ivec2(
			uMousePosition.x,
			uniforms.imageSize.y - uMousePosition.y
		);
		int pixelID = pixelCoords.x + pixelCoords.y * uniforms.imageSize.x;
		float bufferDepth = uintBitsToFloat(ssDepth[pixelID]);

		float aspect = float(uniforms.imageSize.x) / float(uniforms.imageSize.y);
		float PI = 3.1415926;
		float fovRadians = PI * uFovY / 180.0;
		float top = tan(fovRadians / 2.0);
		float height = 2.0 * top;
		float width = aspect * height;
		float left = -0.5 * width;

		vec2 uv = vec2(uMousePosition) / vec2(uniforms.imageSize);
		uv.y = 1.0 - uv.y;
		vec3 dir = vec3(
			uv.x * width - width / 2.0,
			uv.y * height - height / 2.0,
			-1.0
		);

		vec3 camPos = (inverse(uniforms.view * uniforms.world) * vec4(0.0, 0.0, 0.0, 1.0)).xyz;
		vec3 camTarget = (inverse(uniforms.view * uniforms.world) * vec4(dir, 1.0)).xyz;
		vec3 camDir = camTarget - camPos;
		vec3 referencePos = camPos + camDir * selectionDepth;

		float radius = SELECTION_RADIUS * selectionDepth * 0.001;


		vec3 min = vec3(batch.min_x, batch.min_y, batch.min_z);
		vec3 max = vec3(batch.max_x, batch.max_y, batch.max_z);
		vec3 center = (min + max) / 2.0;

		bool isIntersecting = false;
		isIntersecting = isIntersecting || intersect(center, referencePos, radius);
		isIntersecting = isIntersecting || intersect(vec3(min.x, min.y, min.z), referencePos, radius);
		isIntersecting = isIntersecting || intersect(vec3(min.x, min.y, max.z), referencePos, radius);
		isIntersecting = isIntersecting || intersect(vec3(min.x, max.y, min.z), referencePos, radius);
		isIntersecting = isIntersecting || intersect(vec3(min.x, max.y, max.z), referencePos, radius);
		isIntersecting = isIntersecting || intersect(vec3(max.x, min.y, min.z), referencePos, radius);
		isIntersecting = isIntersecting || intersect(vec3(max.x, min.y, max.z), referencePos, radius);
		isIntersecting = isIntersecting || intersect(vec3(max.x, max.y, min.z), referencePos, radius);
		isIntersecting = isIntersecting || intersect(vec3(max.x, max.y, max.z), referencePos, radius);

		if(isIntersecting){
			selectionState = SELECTION_INTERSECTS;
		}
		
	}

	if(false)
	// if(doHighlighting)
	{
		Box box;
		box.min = vec3(batch.min_x, batch.min_y, batch.min_z);
		box.max = vec3(batch.max_x, batch.max_y, batch.max_z);

		vec3 center = (box.min + box.max) / 2.0;

		int numIntersections = 0;
		numIntersections += intersectMouse(center, SELECTION_RADIUS);
		numIntersections += intersectMouse(vec3(box.min.x, box.min.y, box.min.z), SELECTION_RADIUS);
		numIntersections += intersectMouse(vec3(box.min.x, box.min.y, box.max.z), SELECTION_RADIUS);
		numIntersections += intersectMouse(vec3(box.min.x, box.max.y, box.min.z), SELECTION_RADIUS);
		numIntersections += intersectMouse(vec3(box.min.x, box.max.y, box.max.z), SELECTION_RADIUS);
		numIntersections += intersectMouse(vec3(box.max.x, box.min.y, box.min.z), SELECTION_RADIUS);
		numIntersections += intersectMouse(vec3(box.max.x, box.min.y, box.max.z), SELECTION_RADIUS);
		numIntersections += intersectMouse(vec3(box.max.x, box.max.y, box.min.z), SELECTION_RADIUS);
		numIntersections += intersectMouse(vec3(box.max.x, box.max.y, box.max.z), SELECTION_RADIUS);

		if(numIntersections == 9){
			dbg.r = 255;
			selectionState = SELECTION_INTERSECTS;
		}else if(numIntersections > 0){
			dbg.b = 255;
			selectionState = SELECTION_INTERSECTS;
		}


	}


	// // if(false)
	// {
	// 	vec3 wgMin = vec3(batch.min_x, batch.min_y, batch.min_z);
	// 	vec3 wgMax = vec3(batch.max_x, batch.max_y, batch.max_z);

	// 	// FRUSTUM CULLING
	// 	if((!isInsideFrustum(wgMin) && !isInsideFrustum(wgMax))){
	// 		return;
	// 	}

	// 	// LOD CULLING
	// 	vec3 wgCenter = (wgMin + wgMax) / 2.0;
	// 	float wgRadius = distance(wgMin, wgMax);

	// 	vec4 viewCenter = uniforms.view * uniforms.world * vec4(wgCenter, 1.0);
	// 	vec4 viewEdge = viewCenter + vec4(wgRadius, 0.0, 0.0, 0.0);

	// 	vec4 projCenter = uniforms.proj * viewCenter;
	// 	vec4 projEdge = uniforms.proj * viewEdge;

	// 	projCenter.xy = projCenter.xy / projCenter.w;
	// 	projEdge.xy = projEdge.xy / projEdge.w;

	// 	float w_depth = distance(projCenter.xy, projEdge.xy);

	// 	float d_screen = length(projCenter.xy);
	// 	float w_screen = exp(- (d_screen * d_screen) / 1.0);
	// 	w_screen = 1.0;
	// 	// w_screen = ((w_screen - 1.0) * 0.7) + 1.0;

	// 	float w = w_depth * w_screen;

	// 	if(w < 0.01){
	// 		level = 4;
	// 	}else if(w < 0.02){
	// 		level = 3;
	// 	}else if(w < 0.05){
	// 		level = 2;
	// 	}else if(w < 0.1){
	// 		level = 1;
	// 	}else{
	// 		level = 0;
	// 	}

	// 	bool doCullLOD = w < 0.05;

	// 	if(selectionState == SELECTION_NONE && doCullLOD){
	// 		// regular case, but not important enough -> cull

	// 		return;
	// 	}else if(selectionState == SELECTION_INTERSECTS && doCullLOD){
	// 		// keep rendering to enable point-wise selection

	// 		// return;
	// 	}else if(selectionState == SELECTION_INSIDE && doCullLOD){
	// 		// TODO mark whole batch as selected and stop rendering

	// 		if(uSelectState == SELECT_ACTIVE){
	// 			ssBatchBuffer[batchIndex].selection = 1;
	// 		}

	// 		return;
	// 	}



	// 	// if(w < 1.03){
	// 	// 	return;
	// 	// }
	// }

	if(debug.enabled && gl_LocalInvocationID.x == 0){
		atomicAdd(debug.color_numNodesRendered, 1);
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
			atomicAdd(debug.color_numPointsProcessed, 1);
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

		pos = uniforms.transform * pos;
		pos.xyz = pos.xyz / pos.w;

		// bool selected = ssSelection[index]

		bool isInsideFrustum = !(pos.w <= 0.0 || pos.x < -1.0 || pos.x > 1.0 || pos.y < -1.0 || pos.y > 1.0);

		if(isInsideFrustum){
			vec2 imgPos = (pos.xy * 0.5 + 0.5) * uniforms.imageSize;
			ivec2 pixelCoords = ivec2(imgPos);
			int pixelID = pixelCoords.x + pixelCoords.y * uniforms.imageSize.x;

			float depth = pos.w;
			float bufferDepth = uintBitsToFloat(ssDepth[pixelID]);

			// average points within 1% distance
			bool visible = (depth <= bufferDepth * 1.01);

			if((ssRGBA[index] >> 24) == 0xFF){
				// visible = true;
			}

			if(visible){

				bool highlighted = false;
				{

					
					ivec2 pixelCoords = ivec2(
						uMousePosition.x,
						uniforms.imageSize.y - uMousePosition.y
					);
					int pixelID = pixelCoords.x + pixelCoords.y * uniforms.imageSize.x;
					float bufferDepth = uintBitsToFloat(ssDepth[pixelID]);

					float aspect = float(uniforms.imageSize.x) / float(uniforms.imageSize.y);
					float PI = 3.1415926;
					float fovRadians = PI * uFovY / 180.0;
					float top = tan(fovRadians / 2.0);
					float height = 2.0 * top;
					float width = aspect * height;
					float left = -0.5 * width;

					vec2 uv = vec2(uMousePosition) / vec2(uniforms.imageSize);
					uv.y = 1.0 - uv.y;
					vec3 dir = vec3(
						uv.x * width - width / 2.0,
						uv.y * height - height / 2.0,
						-1.0
					);

					vec3 camPos = (inverse(uniforms.view * uniforms.world) * vec4(0.0, 0.0, 0.0, 1.0)).xyz;
					vec3 camTarget = (inverse(uniforms.view * uniforms.world) * vec4(dir, 1.0)).xyz;
					vec3 camDir = camTarget - camPos;
					vec3 referencePos = camPos + camDir * selectionDepth;
					vec3 pointPos = vec3(point.x, point.y, point.z);
					float d = distance(referencePos, pointPos);

					if(d < SELECTION_RADIUS * selectionDepth * 0.001){
						highlighted = true;
					}

					
					if(highlighted && uSelectState == SELECT_ACTIVE && uMouseButtons == 1){
						ssRGBA[index] = ssRGBA[index] | 0xFF000000;
					}else if(highlighted && uSelectState == SELECT_ACTIVE && uMouseButtons == 2){
						ssRGBA[index] = ssRGBA[index] & 0x00FFFFFF;
					}
				}

				uvec4 ballot = subgroupPartitionNV(pixelID);
				uint32_t ballotLeaderID = subgroupPartitionedMinNV(gl_SubgroupInvocationID.x, ballot);
				bool isBallotLeader = (ballotLeaderID == gl_SubgroupInvocationID.x);

				uint32_t color = ssRGBA[index];

				if(uniforms.colorizeChunks){
					color = batchIndex * 45;
				}

				// color = 0x00010101;

				uint32_t R = (color >>  0) & 0xFF;
				uint32_t G = (color >>  8) & 0xFF;
				uint32_t B = (color >> 16) & 0xFF;
				uint32_t A = (color >> 24) & 0xFF;
				uvec4 rgbc = uvec4(R, G, B, 1);
				

				bool selected = (A != 0) || batch.selection != 0;

				if(highlighted){
					rgbc.g = 255;
				}
				if(selected){
					rgbc.r = 255;
					// continue;
				}

				uvec4 ballot_rgbc = subgroupPartitionedAddNV(rgbc, ballot);
				uint32_t ballot_R = ballot_rgbc.x;
				uint32_t ballot_G = ballot_rgbc.y;
				uint32_t ballot_B = ballot_rgbc.z;
				uint32_t ballot_count = ballot_rgbc.w;

				if(isBallotLeader){
					int64_t RG = (int64_t(ballot_R) << 0) | (int64_t(ballot_G) << 32);
					int64_t BA = (int64_t(ballot_B) << 0) | (int64_t(ballot_count) << 32);

					atomicAdd(ssColor[2 * pixelID + 0], RG);
					atomicAdd(ssColor[2 * pixelID + 1], BA);

					if(debug.enabled){
						atomicAdd(debug.color_numPointsRendered, 1);
					}
				}

			}
		}

		barrier();
	}

}