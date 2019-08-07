#version 450

#extension GL_ARB_gpu_shader_int64 : enable
#extension GL_NV_shader_atomic_int64 : enable

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
	// use this if fragment counts are computed instead of colors
	//int64_t ssFramebuffer[];
	uint64_t ssFramebuffer[];
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

	// compute depth in double precision all the way
	// only needed for maximum depth precision, but slows down around 5-10%
	//dvec4 pos64 = dvec4(v.x, v.y, v.z, 1.0);
	//double depth64 = dot(uDepthLine, pos64);
	//double depth = depth64;

	// convert depth from single-precision float to a 64 bit fixed-precision integer.
	double depth = pos.w;
	int64_t u64Depth = int64_t(depth * 1000000.0lf);

	int64_t val64 = (u64Depth << 24) | int64_t(v.colors);


	// 1 pixel
	atomicMin(ssFramebuffer[pixelID], val64);

	// 4 pixels
	//atomicMin(ssFramebuffer[pixelID], val64);
	//atomicMin(ssFramebuffer[pixelID + 1], val64);
	//atomicMin(ssFramebuffer[pixelID + uImageSize.x], val64);
	//atomicMin(ssFramebuffer[pixelID + uImageSize.x + 1], val64);
	
	// count fragments
	//atomicAdd(ssFramebuffer[pixelID], 1l);

}

