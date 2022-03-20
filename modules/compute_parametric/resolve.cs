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


void main(){

	uvec2 id = gl_LocalInvocationID.xy;
	id.x += gl_WorkGroupSize.x * gl_WorkGroupID.x;
	id.y += gl_WorkGroupSize.y * gl_WorkGroupID.y;

	ivec2 pixelCoords = ivec2(id);
	ivec2 sourceCoords = ivec2(id);
	int pixelID = sourceCoords.x + sourceCoords.y * uImageSize.x;

	{ // n x n pixel
		ivec2 pixelCoords = ivec2(id);

		float R = 0;
		float G = 0;
		float B = 0;
		float count = 0;

		int window = 1;
		float closestDepth = 1000000.0;
		uint32_t closestPointColor = 0;

		for(int ox = -window; ox <= window; ox++){
		for(int oy = -window; oy <= window; oy++){

			int pixelID = (pixelCoords.x + ox) + (pixelCoords.y + oy) * uImageSize.x;

			uint64_t data = ssFramebuffer[pixelID];
			uint32_t uDepth = uint32_t(data >> 32l);
			uint32_t pointColor = uint32_t(data & 0xffffffffl);
			float depth = uintBitsToFloat(uDepth);

			if(depth > 0.0 && depth < closestDepth){
				closestDepth = depth;
				closestPointColor = pointColor;
			}
			
		}
		}


		uint32_t color = closestPointColor;

		if(closestPointColor == 0){
			color = 0x00443322;
		}

		imageAtomicExchange(uOutput, pixelCoords, color);
	}
}