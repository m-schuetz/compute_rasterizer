#version 450

#extension GL_ARB_gpu_shader_int64 : enable
#extension GL_NV_shader_atomic_int64 : enable
#extension GL_NV_gpu_shader5 : enable


#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_KHR_shader_subgroup_vote : require
#extension GL_KHR_shader_subgroup_clustered : require

// Maybe on the 3090??
// #extension GL_EXT_shader_realtime_clock : require
#extension GL_ARB_shader_clock : require


layout(local_size_x = 16, local_size_y = 16) in;

layout(r32ui, binding = 0) coherent uniform uimage2D uOutput;
layout(std430, binding = 1) buffer abc_2 { uint32_t ssDepth[]; };
layout(std430, binding = 2) buffer abc_3 { uint32_t ssRGBA[]; };

layout (std430, binding = 30) buffer abc_1 { 
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

void main(){

	uvec2 id = gl_LocalInvocationID.xy;
	id.x += gl_WorkGroupSize.x * gl_WorkGroupID.x;
	id.y += gl_WorkGroupSize.y * gl_WorkGroupID.y;

	ivec2 imgSize = uniforms.imageSize;

	ivec2 pixelCoords = ivec2(id);
	ivec2 sourceCoords = ivec2(id);
	int pixelID = sourceCoords.x + sourceCoords.y * imgSize.x;

	uint32_t R = ssRGBA[4 * pixelID + 0];
	uint32_t G = ssRGBA[4 * pixelID + 1];
	uint32_t B = ssRGBA[4 * pixelID + 2];
	uint32_t count = ssRGBA[4 * pixelID + 3];

	if(uniforms.colorizeChunks){
		// color = batchIndex * 1234;
	}else if(uniforms.colorizeOverdraw){
		// count = 1;

		int spectralIndex = 0;

		if(count < 10){
			spectralIndex = 0;
		}else if(count < 250){
			spectralIndex = 1;
		}else if(count < 1000){
			spectralIndex = 2;
		}else if(count < 4000){
			spectralIndex = 3;
		}else{
			spectralIndex = 4;
		}

		R = (SPECTRAL[spectralIndex] >>  0) & 0xFF;
		G = (SPECTRAL[spectralIndex] >>  8) & 0xFF;
		B = (SPECTRAL[spectralIndex] >> 16) & 0xFF;
		
		count = 1;
	}

	uint32_t r = R / count;
	uint32_t g = G / count;
	uint32_t b = B / count;

	// r = 20 * uint(log(float(count)) / log(2));
	// g = 20 * uint(log(float(count)) / log(2));
	// b = 20 * uint(log(float(count)) / log(2));

	if(count == 0){
		r = 0;
		g = 0;
		b = 0;
	}

	uint32_t color = r | (g << 8) | (b << 16);
	
	if(count == 0){
		color = 0x00443322;
	}else{
		if(debug.enabled){
			atomicAdd(debug.numPointsVisible, 1);
		}
	}

	imageAtomicExchange(uOutput, pixelCoords, color);
}