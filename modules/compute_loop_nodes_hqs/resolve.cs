#version 450

#extension GL_ARB_gpu_shader_int64 : enable
#extension GL_NV_shader_atomic_int64 : enable
#extension GL_NV_gpu_shader5 : enable
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_KHR_shader_subgroup_vote : require
#extension GL_KHR_shader_subgroup_clustered : require

layout(local_size_x = 16, local_size_y = 16) in;

layout(r32ui, binding = 0) coherent uniform uimage2D uOutput;
layout (std430, binding = 1) buffer abc_0 { uint32_t ssDepth[]; };
layout (std430, binding = 2) buffer abc_1 { uint32_t ssRGBA[]; };

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
} uniforms;

void main(){

	uvec2 id = gl_LocalInvocationID.xy;
	id.x += gl_WorkGroupSize.x * gl_WorkGroupID.x;
	id.y += gl_WorkGroupSize.y * gl_WorkGroupID.y;

	ivec2 imgSize = uniforms.imageSize;

	// { // 1 pixel
	// 	ivec2 pixelCoords = ivec2(id);
	// 	ivec2 sourceCoords = ivec2(id);
	// 	int pixelID = sourceCoords.x + sourceCoords.y * imgSize.x;

	// 	uint32_t R = ssRGBA[4 * pixelID + 0];
	// 	uint32_t G = ssRGBA[4 * pixelID + 1];
	// 	uint32_t B = ssRGBA[4 * pixelID + 2];
	// 	uint32_t count = ssRGBA[4 * pixelID + 3];

	// 	uint32_t r = R / count;
	// 	uint32_t g = G / count;
	// 	uint32_t b = B / count;

		// if(count == 0){
		// 	r = 0;
		// 	g = 0;
		// 	b = 0;
		// }else{
		// 	if(debug.enabled){
		// 		atomicAdd(debug.numPointsVisible, 1);
		// 	}
		// }

	// 	uint32_t color = r | (g << 8) | (b << 16);

	// 	imageAtomicExchange(uOutput, pixelCoords, color);
	// }

	{ // n x n pixel
		ivec2 pixelCoords = ivec2(id);

		float R = 0;
		float G = 0;
		float B = 0;
		float count = 0;

		int window = 0;
		float depth = 1000000.0;
		for(int ox = -window; ox <= window; ox++){
		for(int oy = -window; oy <= window; oy++){

			int pixelID = (pixelCoords.x + ox) + (pixelCoords.y + oy) * imgSize.x;

			float pixelDepth = uintBitsToFloat(ssDepth[pixelID]);
			if(pixelDepth >= 0.0){
				depth = min(depth, pixelDepth);
			}
		}
		}

		for(int ox = -window; ox <= window; ox++){
		for(int oy = -window; oy <= window; oy++){

			int pixelID = (pixelCoords.x + ox) + (pixelCoords.y + oy) * imgSize.x;
			float pixelDepth = uintBitsToFloat(ssDepth[pixelID]);

			float w = 1.0;
			if(ox == 0 && oy == 0){
				w = 100;
			}else if(ox <= 1 && oy <= 1){
				w = 2;
			}else{
				w = 1;
			}

			if(pixelDepth > depth * 1.01){
				w = 0;
			}

			R += float(ssRGBA[4 * pixelID + 0]) * w;
			G += float(ssRGBA[4 * pixelID + 1]) * w;
			B += float(ssRGBA[4 * pixelID + 2]) * w;
			count += float(ssRGBA[4 * pixelID + 3]) * w;
		}
		}

		uint32_t r = uint32_t(R / count);
		uint32_t g = uint32_t(G / count);
		uint32_t b = uint32_t(B / count);

		if(count == 0){
			r = 0;
			g = 0;
			b = 0;
		}else{
			if(debug.enabled){
				atomicAdd(debug.numPointsVisible, 1);
			}
		}


		uint32_t color = r | (g << 8) | (b << 16);
		// color = uint32_t(depth);
		if(depth < 0.0){
			color = 0x0000FF00;
		}
		imageAtomicExchange(uOutput, pixelCoords, color);
	}
}