
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

layout (std430, binding = 1) buffer framebuffer_data {
	int64_t ssFramebuffer[];
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
	float depthInBuffer = uintBitsToFloat(ssDepthbuffer[pixelID]);

	// r18-g18-b18-a10 bit counters

	bool depthFits = depth <= depthInBuffer * 1.01;

	// uint colors = v.colors;
	uint r = (v.colors >>  0) & 0xFF;
	uint g = (v.colors >>  8) & 0xFF;
	uint b = (v.colors >> 16) & 0xFF;

	uvec2 sum = uvec2(r, b << 16 | g);
	//uvec4 sum = uvec4(r, g, b, 1);

	bool fastPathEnabled = true;
	int leaderID = shuffleNV(pixelID, 0, 32);
	uint participate = ballotThreadNV(leaderID == pixelID && depthFits);
	bool isLeader = gl_ThreadInWarpNV == 0;
	if(participate == 0xffffffff && fastPathEnabled){

		sum = sum + shuffleDownNV(sum, 16, 32);
		sum = sum + shuffleDownNV(sum,  8, 32);
		sum = sum + shuffleDownNV(sum,  4, 32);
		sum = sum + shuffleDownNV(sum,  2, 32);
		sum = sum + shuffleDownNV(sum,  1, 32);

		if(isLeader){

			r = sum.x / 32;
			g = (sum.y / 32) & 0xff;
			b = (sum.y >> 16) / 32;

			int64_t rgba = (int64_t(r) << 46) | (int64_t(g) << 28) | (int64_t(b) << 10) | 1;
			atomicAdd(ssRGBA[pixelID], rgba);
		}
	}else if(depthFits) {
		int64_t rgba = (int64_t(r) << 46) | (int64_t(g) << 28) | (int64_t(b) << 10) | 1;
		atomicAdd(ssRGBA[pixelID], rgba);
	}



		// int64_t old = ssRGBA[pixelID];
		// uint oldFragmentCounter = uint(old & 0x3ffUL);

		// if(oldFragmentCounter < 100){
			// int64_t b = int64_t((v.colors >> 16) & 0xFF);
			// int64_t g = int64_t((v.colors >> 8) & 0xFF);
			// int64_t r = int64_t((v.colors >> 0) & 0xFF);
			// int64_t a = 1;

			// int64_t rgba = (r << 46) | (g << 28) | (b << 10) | a;

			// atomicAdd(ssRGBA[pixelID], rgba);
		// }
	//}


}