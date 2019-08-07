
#version 450

#extension GL_ARB_gpu_shader_int64 : enable
#extension GL_NV_shader_atomic_int64 : enable

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
	uint64_t ssDepthbuffer[];
};

layout (std430, binding = 3) buffer rg_data {
	int64_t ssRG[];
};

layout (std430, binding = 4) buffer ba_data {
	int64_t ssBA[];
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

	// if(v.x < 130 || v.x > 165){
	// 	return;
	// }
	// if(v.y < 151 || v.y > 180){
	// 	return;
	// }

	vec2 imgPos = (pos.xy * 0.5 + 0.5) * uImageSize;
	ivec2 pixelCoords = ivec2(imgPos);
	int pixelID = pixelCoords.x + pixelCoords.y * uImageSize.x;

	float depth = pos.w;
	uint64_t u64Depth = uint64_t(depth * 1000000.0);

	uint64_t u64Colors = v.colors;
	uint64_t val64 = (u64Depth << 24) | v.colors;

	uint64_t depthInBuffer = ssDepthbuffer[pixelID];

	if(u64Depth <= double(depthInBuffer) * 1.01){
	//if(u64Depth == depthInBuffer){
	//{

		int64_t b = int64_t((v.colors >> 16) & 0xFF);
		int64_t g = int64_t((v.colors >> 8) & 0xFF);
		int64_t r = int64_t((v.colors >> 0) & 0xFF);
		int64_t a = 1;

		int64_t rg = (r << 32) | g;
		int64_t ba = (b << 32) | a;

		atomicAdd(ssRG[pixelID], rg);
		atomicAdd(ssBA[pixelID], ba);

		//atomicAdd(ssRG[pixelID + 100], rg);
		//atomicAdd(ssBA[pixelID + 100], ba);

	}


}

