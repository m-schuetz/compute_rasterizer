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
	//Vertex vertices[];
	uint64_t vertices[];
};

layout (std430, binding = 1) buffer framebuffer_data {
	// use this if fragment counts are computed instead of colors
	//int64_t ssFramebuffer[];
	uint64_t ssFramebuffer[];
};

uniform ivec2 uImageSize;

layout(binding = 0) uniform sampler2D uGradient;

Vertex readVertex(uint index){

	// 0x1fff = 0b1'1111'1111'1111ul = 13 x 1 bits

	uint64_t data = vertices[index];

	uint X = uint((data >>  0ul) & 0x1ffful);
	uint Y = uint((data >> 13ul) & 0x1ffful);
	uint Z = uint((data >> 26ul) & 0x1ffful);
	uint r = uint((data >> 40ul) & 0xFFul);
	uint g = uint((data >> 48ul) & 0xFFul);
	uint b = uint((data >> 56ul) & 0xFFul);

	uint colors = r | (g << 8) | (b << 16);

	Vertex vertex;
	vertex.x = float(X) / 300.0;
	vertex.y = float(Y) / 300.0;
	vertex.z = float(Z) / 300.0;
	vertex.colors = colors;

	return vertex;
}

void main(){

	uint globalID = gl_GlobalInvocationID.x;

	Vertex v = readVertex(globalID);

	vec4 pos = uTransform * vec4(v.x, v.y, v.z, 1.0);
	pos.xyz = pos.xyz / pos.w;

	if(pos.w <= 0.0 || pos.x < -1.0 || pos.x > 1.0 || pos.y < -1.0 || pos.y > 1.0){
		return;
	}

	vec2 imgPos = (pos.xy * 0.5 + 0.5) * uImageSize;
	ivec2 pixelCoords = ivec2(imgPos);
	int pixelID = pixelCoords.x + pixelCoords.y * uImageSize.x;

	// convert depth from single-precision float to a 64 bit fixed-precision integer.
	int32_t depth = floatBitsToInt(pos.w);
	int32_t minD = depth;

	int leaderID = shuffleNV(pixelID, 0, 32);
	uint haveLeaderID = ballotThreadNV(leaderID == pixelID);
	bool fastPath = (haveLeaderID == 0xffffffff);

	if(fastPath) // fast path for min
	{		
		minD = min(minD, shuffleXorNV(minD, 16,32));
		minD = min(minD, shuffleXorNV(minD, 8, 32));
		minD = min(minD, shuffleXorNV(minD, 4, 32));
		minD = min(minD, shuffleXorNV(minD, 2, 32));
		minD = min(minD, shuffleXorNV(minD, 1, 32));
	}

	// no fast path or thread was depth comp winner
	if (minD == depth){
		int64_t val64 = (int64_t(depth) << 24) | int64_t(v.colors);

		bool lateStageReduction = false;
		if(lateStageReduction && !fastPath)
		{
			// last chance for reduction
			uint neighborPixelID = shuffleXorNV(pixelID,  1, 32);
			uvec2 neighborVal = shuffleXorNV(uvec2(depth, v.colors), 1, 32);
			if(gl_LocalInvocationID.x%2 == 0 && neighborPixelID == pixelID)
			{
				int64_t neighborVal64 = (int64_t(neighborVal.x) << 24) | int64_t(neighborVal.y);
				val64 = min(val64, neighborVal64);
				uint64_t old = ssFramebuffer[pixelID];
				if(val64 < old)
				{
					atomicMin(ssFramebuffer[pixelID], val64);
				}
				return;
			}
		}

		uint64_t old = ssFramebuffer[pixelID];
		if(val64 < old)
		{
			atomicMin(ssFramebuffer[pixelID], val64);
		}

		// 1 pixel


		// 4 pixels
		// atomicMin(ssFramebuffer[pixelID], val64);
		// atomicMin(ssFramebuffer[pixelID + 1], val64);
		// atomicMin(ssFramebuffer[pixelID + uImageSize.x], val64);
		// atomicMin(ssFramebuffer[pixelID + uImageSize.x + 1], val64);
		
		// count fragments
		//atomicAdd(ssFramebuffer[pixelID], 1l);
	}

}