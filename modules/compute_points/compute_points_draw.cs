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


layout(local_size_x = 128, local_size_y = 1) in;

layout(location = 0) uniform dmat4 uTransform;
layout(location = 1) uniform ivec2 uImageSize;
layout(r32ui, binding = 0) coherent uniform uimage2D uOutput;

layout (std430, binding = 3) buffer depth_data {
	uint32_t fbo32[];
};

layout (std430, binding = 4) buffer ss_d_rgba {
	uint64_t framebuffer[];
};

void main(){
	
	uint globalID = gl_GlobalInvocationID.x;

		
	vec2 imgPos = vec2(globalID, globalID);
	ivec2 pixelCoords = ivec2(imgPos);
	int pixelID = pixelCoords.x + pixelCoords.y * uImageSize.x;

	uint data = 0xFF0000FF;

	// imageAtomicExchange(uOutput, pixelCoords, data);

	fbo32[pixelID] = data;
}