#version 450

#extension GL_ARB_gpu_shader_int64 : enable
#extension GL_NV_shader_atomic_int64 : enable
#extension GL_NV_gpu_shader5 : enable

#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_KHR_shader_subgroup_vote : require
#extension GL_NV_shader_subgroup_partitioned : require

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

layout (std430, binding = 2) buffer debug_data {
	uint32_t dbgCountAtomicMin;
	uint32_t dbgCountDistinctPixels;
};

uniform ivec2 uImageSize;

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

	// get mask of threads with same pixelID
	uvec4 mask = subgroupPartitionNV(pixelID);
	// minDepth of threads with same pixelID
	int32_t minDepth = subgroupPartitionedMinNV(depth, mask);


	// { // count distinct pixels per subgroup
	// 	uint32_t firstID = subgroupPartitionedMinNV(gl_SubgroupInvocationID, mask);

	// 	if(firstID == gl_SubgroupInvocationID){
	// 		atomicAdd(dbgCountDistinctPixels, 1);
	// 	}
	// }

	// eliminate duplicates, except points with identical pixelID and depth
	bool isClosestThread = (depth == minDepth);

	// Also eliminate duplicates from points with same pixelID and same depth:
	// bool isClosestThread = false;
	// {
	// 	uvec4 mask_minD = subgroupBallot(depth == minDepth);
	// 	mask.x = mask.x & mask_minD.x;

	// 	uint32_t firstID = subgroupPartitionedMinNV(gl_SubgroupInvocationID, mask);

	// 	if(firstID == gl_SubgroupInvocationID){
	// 		isClosestThread = true;
	// 	}
	// }


	if(isClosestThread){
		// thread has smallest depth for a given pixel ID
		
		// count atomicMin
		// atomicAdd(dbgCountAtomicMin, 1);

		int64_t val64 = (int64_t(depth) << 24) | int64_t(v.colors);

		uint64_t old = ssFramebuffer[pixelID];
		if(val64 < old){
			atomicMin(ssFramebuffer[pixelID], val64);
		}
	}

}
