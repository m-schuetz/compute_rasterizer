#version 450

#extension GL_ARB_gpu_shader_int64 : enable
#extension GL_NV_shader_atomic_int64 : enable
#extension GL_NV_gpu_shader5 : enable
#extension GL_NV_shader_thread_group : require
#extension GL_NV_shader_thread_shuffle : require
#extension GL_ARB_shader_group_vote : require
#extension GL_ARB_shader_ballot : require

// 
// This compute shader renders point clouds with 1 to 4 pixels per point.
// It is essentially vertex-shader, rasterizer and fragment-shader in one.
// Rendering happens by encoding depth values in the most significant bits, 
// and color values in the least significant bits of a 64 bit integer.
// atomicMin then writes the fragment with the smallest depth value into an SSBO.
// 
// The SSBO is essentially our framebuffer with color and depth attachment combined.
// A second compute shader, resolve.cs, then transfers the values from the SSBO to an 
// actual OpenGL texture.
// 

layout(local_size_x = 128, local_size_y = 1) in;

struct Vertex{
	float x;
	float y;
	float z;
	uint colors;
};

layout(location = 0) uniform mat4 uTransform;

// depth colum/row of the worldViewProjection matrix in double precision
layout(location = 1) uniform dvec4 uDepthLine;

layout (std430, binding = 0) buffer point_data {
	Vertex vertices[];
};

layout (std430, binding = 1) buffer framebuffer_data {
	uint64_t ssFramebuffer[];
};

uniform ivec2 uImageSize;

layout(binding = 0) uniform sampler2D uGradient;

bool neighborActive(uint index){

	uint mask = activeThreadsNV();
	uint sourceThreadId = gl_ThreadEqMaskNV ^ index;

	return (mask & (1 << sourceThreadId)) != 0;
}

uint64_t encode(int32_t depth, uint color){
	return (int64_t(depth) << 24) | int64_t(color);
}

void main(){

	uint globalID = gl_GlobalInvocationID.x;

	Vertex v = vertices[globalID];

	vec4 pos = uTransform * vec4(v.x, v.y, v.z, 1.0);
	pos.xyz = pos.xyz / pos.w;

	if(pos.w <= 0.0 || pos.x < -1.0 || pos.x > 1.0 || pos.y < -1.0 || pos.y > 1.0){
		return;
	}

	vec2 imgPos = (pos.xy * 0.5 + 0.5) * uImageSize;
	ivec2 pixelCoords = ivec2(imgPos);
	int pixelID = pixelCoords.x + pixelCoords.y * uImageSize.x;





	int32_t depth = floatBitsToInt(pos.w);
	uint color = v.colors;
	int64_t val64 = (int64_t(depth) << 24) | int64_t(color);


	int32_t neighborPixelID = shuffleXorNV(pixelID, 1, 32);
	int32_t neighborDepth = shuffleXorNV(depth, 1, 32);
	uint neighborColor = shuffleXorNV(color, 1, 32);

	if(pixelID != neighborPixelID){
		atomicMin(ssFramebuffer[pixelID], val64);

		return;
	}else{

		if(neighborActive(1) && neighborDepth < depth){
			depth = neighborDepth;
			color = neighborColor;
		}

	}

	if((gl_ThreadEqMaskNV & 1u) == 0u){

		if(neighborActive(2)){
			int32_t neighborPixelID = shuffleXorNV(pixelID, 2, 32);
			int32_t neighborDepth = shuffleXorNV(depth, 2, 32);
			uint neighborColor = shuffleXorNV(color, 2, 32);

			if(pixelID != neighborPixelID){
				atomicMin(ssFramebuffer[pixelID], encode(depth, color));
				// atomicMin(ssFramebuffer[pixelID], (int64_t(depth) << 24) | 0x000000fffful);

				return;
			}else if(neighborDepth < depth){
				depth = neighborDepth;
				color = neighborColor;
			}
		}
		
		// atomicMin(ssFramebuffer[pixelID], encode(depth, color));
		// atomicMin(ssFramebuffer[pixelID], (int64_t(depth) << 24) | 0x00000000fful);
	}

	if((gl_ThreadEqMaskNV & 2u) == 0u){

		if(neighborActive(4)){
			int32_t neighborPixelID = shuffleXorNV(pixelID, 4, 32);
			int32_t neighborDepth = shuffleXorNV(depth, 4, 32);
			uint neighborColor = shuffleXorNV(color, 4, 32);

			if(pixelID != neighborPixelID){
				// atomicMin(ssFramebuffer[pixelID], encode(depth, color));
				// atomicMin(ssFramebuffer[pixelID], (int64_t(depth) << 24) | 0x0000ff00fful);

				return;
			}else if(neighborDepth < depth){
				depth = neighborDepth;
				color = neighborColor;
			}
		}
		
		// atomicMin(ssFramebuffer[pixelID], encode(depth, color));
		// atomicMin(ssFramebuffer[pixelID], (int64_t(depth) << 24) | 0x00000000fful);
	}



}