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
layout(r32ui, binding = 0) coherent uniform uimage2D uFboLeft;
layout(r32ui, binding = 1) coherent uniform uimage2D uFboRight;

layout (std430, binding = 1) buffer framebuffer_depth {
	uint32_t ssLeft_depth[];
};

layout (std430, binding = 2) buffer framebuffer_left_color {
	uint32_t ssLeft_rgba[];
};

layout (std430, binding = 3) buffer framebuffer_right_depth {
	uint32_t ssRight_depth[];
};

layout (std430, binding = 4) buffer framebuffer_right_color {
	uint32_t ssRight_rgba[];
};

#define MAX_BUFFER_SIZE 2000000000l

void main(){

	uvec2 id = gl_LocalInvocationID.xy;
	id.x += gl_WorkGroupSize.x * gl_WorkGroupID.x;
	id.y += gl_WorkGroupSize.y * gl_WorkGroupID.y;

	ivec2 imgSize = uImageSize;

	ivec2 pixelCoords = ivec2(id);
	ivec2 sourceCoords = ivec2(id);
	int pixelID = sourceCoords.x + sourceCoords.y * imgSize.x;

	{ // LEFT
		uint32_t R = ssLeft_rgba[4 * pixelID + 0];
		uint32_t G = ssLeft_rgba[4 * pixelID + 1];
		uint32_t B = ssLeft_rgba[4 * pixelID + 2];
		uint32_t count = ssLeft_rgba[4 * pixelID + 3];

		uint32_t r = R / count;
		uint32_t g = G / count;
		uint32_t b = B / count;

		if(count == 0){
			r = 0;
			g = 0;
			b = 0;
		}

		uint32_t color = r | (g << 8) | (b << 16);

		imageAtomicExchange(uFboLeft, pixelCoords, color);
		// imageAtomicExchange(uFboRight, pixelCoords, color);
	}

	{ // RIGHT
		uint32_t R = ssRight_rgba[4 * pixelID + 0];
		uint32_t G = ssRight_rgba[4 * pixelID + 1];
		uint32_t B = ssRight_rgba[4 * pixelID + 2];
		uint32_t count = ssRight_rgba[4 * pixelID + 3];

		uint32_t r = R / count;
		uint32_t g = G / count;
		uint32_t b = B / count;

		if(count == 0){
			r = 0;
			g = 0;
			b = 0;
		}

		uint32_t color = r | (g << 8) | (b << 16);

		imageAtomicExchange(uFboRight, pixelCoords, color);
	}
}