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

// layout(location = 0) uniform dmat4 uTransform;
layout(location = 1) uniform ivec2 uImageSize;
layout(r32ui, binding = 0) coherent uniform uimage2D uOutput;

layout (std430, binding=1) buffer framebuffer_data {
	uint64_t ssFramebuffer[];
};

layout (std430, binding = 50) buffer data_colors_0 {
	uint32_t ssColors_0[];
};

layout (std430, binding = 51) buffer data_colors_1 {
	uint32_t ssColors_1[];
};


void main(){

	uvec2 id = gl_LocalInvocationID.xy;
	id.x += gl_WorkGroupSize.x * gl_WorkGroupID.x;
	id.y += gl_WorkGroupSize.y * gl_WorkGroupID.y;

	ivec2 imgSize = uImageSize;

	ivec2 pixelCoords = ivec2(id);
	ivec2 sourceCoords = ivec2(id);
	int pixelID = sourceCoords.x + sourceCoords.y * imgSize.x;

	uint64_t data = ssFramebuffer[pixelID];
	uint32_t pointID = uint32_t(data & 0xFFFFFFFFul);

	uint32_t color = 0xFF0000FF;
	uint colorID = pointID % 500000000;
	if(pointID < 500000000){
		color = ssColors_0[colorID];
	}else{
		color = ssColors_1[colorID];
	}


	if(pointID >= 0x7FFFFFFF){
		color = 0x00000000;
	}

	uint value = uint(data & 0xFFFFFFFFul);
	imageAtomicExchange(uOutput, pixelCoords, color);
}