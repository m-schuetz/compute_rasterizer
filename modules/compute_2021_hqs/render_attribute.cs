
#version 450

#extension GL_ARB_gpu_shader_int64 : enable
#extension GL_NV_shader_atomic_int64 : enable
#extension GL_NV_gpu_shader5 : enable
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

layout (std430, binding = 2) buffer depthbuffer_data {
	uint ssDepthbuffer[];
};

layout (std430, binding = 3) buffer rgba_data {
	int64_t ssRGBA[];
};

uniform ivec2 uImageSize;

layout(binding = 0) uniform sampler2D uGradient;


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

	

	float depth = pos.w;
	float bufferedDepth = uintBitsToFloat(ssDepthbuffer[pixelID]);

	// {
	// 	// int pixelID = int(globalID) * 1234;
	// 	int64_t value = 0x00ffff0000ffff00l;
	// 	ssRGBA[2 * pixelID + 1] = int64_t(bufferedDepth);
	// }

	if(depth <= bufferedDepth * 1.01){

		int64_t b = int64_t((v.colors >> 16) & 0xFF);
		int64_t g = int64_t((v.colors >> 8) & 0xFF);
		int64_t r = int64_t((v.colors >> 0) & 0xFF);
		uvec4 sum = uvec4(r, g, b, 1);

		bool fastPathEnabled = true;
		int leaderID = shuffleNV(pixelID, 0, 32);
		uint haveLeaderID = ballotThreadNV(leaderID == pixelID);
		bool isLeader = gl_ThreadInWarpNV == 0;
		if(haveLeaderID == 0xffffffff && fastPathEnabled){

			sum = sum + shuffleXorNV(sum, 16, 32);
			sum = sum + shuffleXorNV(sum,  8, 32);
			sum = sum + shuffleXorNV(sum,  4, 32);
			sum = sum + shuffleXorNV(sum,  2, 32);
			sum = sum + shuffleXorNV(sum,  1, 32);

			if(isLeader){
				r = int64_t(sum.r / sum.a);
				g = int64_t(sum.g / sum.a);
				b = int64_t(sum.b / sum.a);

				int64_t rg = (r << 32) | g;
				int64_t ba = (b << 32) | 1;

				atomicAdd(ssRGBA[2 * pixelID + 0], rg);
				atomicAdd(ssRGBA[2 * pixelID + 1], ba);
			}
		}else{
			int64_t rg = (r << 32) | g;
			int64_t ba = (b << 32) | 1;

			atomicAdd(ssRGBA[2 * pixelID + 0], rg);
			atomicAdd(ssRGBA[2 * pixelID + 1], ba);
		}

	}


}
