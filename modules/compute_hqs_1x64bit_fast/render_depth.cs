
#version 450

#extension GL_ARB_gpu_shader_int64 : enable
#extension GL_NV_shader_atomic_int64 : enable
#extension GL_NV_shader_thread_group : require
#extension GL_NV_shader_thread_shuffle : require
#extension GL_ARB_shader_group_vote : require
#extension GL_ARB_shader_ballot : require

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

layout (std430, binding = 1) buffer depthbuffer_data {
	uint ssDepthbuffer[];
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

	uint depth = floatBitsToUint(pos.w);
	uint minD = depth;

	int leaderID = shuffleNV(pixelID, 0, 32);
	uint haveLeaderID = ballotThreadNV(leaderID == pixelID);
	
	if(haveLeaderID == 0xffffffff) // fast path for min
	{		
		minD = min(minD, shuffleXorNV(minD, 16,32));
		minD = min(minD, shuffleXorNV(minD, 8, 32));
		minD = min(minD, shuffleXorNV(minD, 4, 32));
		minD = min(minD, shuffleXorNV(minD, 2, 32));
		minD = min(minD, shuffleXorNV(minD, 1, 32));
	}

	if (minD == depth) // no fast path or thread was depth comp winner
	{
		uint old = ssDepthbuffer[pixelID];
		if(depth < old){
			atomicMin(ssDepthbuffer[pixelID], depth);
		}
	}
}

