#version 450

#extension GL_ARB_gpu_shader_int64 : enable
#extension GL_NV_shader_atomic_int64 : enable
#extension GL_NV_gpu_shader5 : enable


#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_KHR_shader_subgroup_vote : require
#extension GL_KHR_shader_subgroup_clustered : require

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

layout (std430, binding = 0) buffer point_data {
	Vertex vertices[];
};

layout (std430, binding = 1) buffer framebuffer_data {
	uint64_t ssFramebuffer[];
};

uniform ivec2 uImageSize;

layout(binding = 0) uniform sampler2D uGradient;

void writeVal(int64_t val64, int pixelID)
{
	atomicMin(ssFramebuffer[pixelID], val64);
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
	int32_t minD = depth;

	bool fastPath = subgroupAllEqual(pixelID);
	if(fastPath) // fast path for min
	{		
		minD = subgroupMin(minD);
	}

	// no fast path or thread was depth comp winner
	if (minD == depth){
		int64_t val64 = (int64_t(depth) << 24) | int64_t(v.colors);

		// last chance for reduction
		if (fastPath
		|| subgroupClusteredXor(pixelID, 2) != 0 
		|| subgroupClusteredMin(depth, 2) == depth)
		{
			writeVal(val64, pixelID);
		}
	}
}