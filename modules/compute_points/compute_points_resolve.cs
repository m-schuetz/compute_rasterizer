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

	uvec2 id = gl_LocalInvocationID.xy;
	id.x += gl_WorkGroupSize.x * gl_WorkGroupID.x;
	id.y += gl_WorkGroupSize.y * gl_WorkGroupID.y;

	// ivec2 imgSize = imageSize(uOutput);
	ivec2 imgSize = uImageSize;

	ivec2 pixelCoords = ivec2(id);
	ivec2 sourceCoords = ivec2(id);
	int pixelID = sourceCoords.x + sourceCoords.y * imgSize.x;


	uint data = fbo32[pixelID];

	// for(int i = 0; i < 10; i++){
		// pixelCoords = ivec2(i, i);
		// data = 0xFFFF00FF;

		imageAtomicExchange(uOutput, pixelCoords, data);
	// }

	// imageStore(uOutput, pixelCoords, icolor);

}