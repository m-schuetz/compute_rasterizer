#version 460

// #extension GL_ARB_gpu_shader_int64 : enable
#extension GL_NV_shader_atomic_int64 : enable
#extension GL_NV_gpu_shader5 : enable

// #extension GL_KHR_shader_subgroup_basic : require
// #extension GL_KHR_shader_subgroup_arithmetic : require
// #extension GL_KHR_shader_subgroup_ballot : require
// #extension GL_KHR_shader_subgroup_vote : require
// #extension GL_KHR_shader_subgroup_clustered : require

layout(local_size_x = 128, local_size_y = 1) in;

struct Vertex{
	float x;
	float y;
	float z;
	uint colors;
};

layout (std430, binding = 5) buffer point_data {
	Vertex vertices[];
};

layout (std430, binding = 1) buffer framebuffer_data {
	// use this if fragment counts are computed instead of colors
	//int64_t ssFramebuffer[];
	uint64_t ssFramebuffer[];
};

layout(location = 0) uniform mat4 uTransform;
layout(location = 1) uniform ivec2 uImageSize;

void main(){

	uint globalID = uint(gl_WorkGroupID.x * gl_WorkGroupSize.x);
	globalID = globalID + gl_LocalInvocationID.x;

	Vertex v = vertices[globalID];

	vec4 pos = uTransform * vec4(v.x, v.y, v.z, 1.0);
	pos.xyz = pos.xyz / pos.w;

	if(pos.w <= 0.0 || pos.x < -1.0 || pos.x > 1.0 || pos.y < -1.0 || pos.y > 1.0){
		return;
	}

	vec2 imgPos = (pos.xy * 0.5 + 0.5) * uImageSize;
	ivec2 pixelCoords = ivec2(imgPos);
	int pixelID = pixelCoords.x + pixelCoords.y * uImageSize.x;

	int64_t u64Depth = floatBitsToInt(pos.w);

	int64_t val64 = (u64Depth << 32) | int64_t(v.colors);

	// 1 pixel
	uint64_t old = ssFramebuffer[pixelID];
	uint64_t oldDepth = (old >> 32);

	if(u64Depth < oldDepth){
		atomicMin(ssFramebuffer[pixelID], val64);
	}

	// 4 pixels
	// atomicMin(ssFramebuffer[pixelID], val64);
	// atomicMin(ssFramebuffer[pixelID + 1], val64);
	// atomicMin(ssFramebuffer[pixelID + uImageSize.x], val64);
	// atomicMin(ssFramebuffer[pixelID + uImageSize.x + 1], val64);
	
	// count fragments
	//atomicAdd(ssFramebuffer[pixelID], 1l);

}
