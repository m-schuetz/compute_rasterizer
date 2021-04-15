
#version 450

#extension GL_ARB_gpu_shader_int64 : enable
#extension GL_NV_shader_atomic_int64 : enable

layout(local_size_x = 16, local_size_y = 16) in;

struct Vertex{
	float x;
	float y;
	float z;
	uint colors;
};

layout (std430, binding=0) buffer point_data {
	Vertex vertices[];
};

layout (std430, binding=2) buffer depthbuffer_data {
	uint ssDepthbuffer[];
};

layout (std430, binding=3) buffer rgba_data {
	// uint64_t ssRGBA[];
	uint ssRGBA[];
};

layout (std430, binding=4) buffer rgba_data_slow {
	// uint64_t ssRGBA[];
	uvec4 ssRGBA_slow[];
};

layout(rgba8ui, binding = 0) uniform uimage2D uOutput;

layout(binding = 1) uniform sampler2D uGradient;

#define ROBUST
#ifdef ROBUST
uvec4 rgbAt(int pixelID){
	// decoding the colors is a whole lot faster when you interpret the 
	// ssRGBA buffer as 2x32 bit uints, rather than 1x64 bit uint, on the GTX 1060.
	// 3.0ms vs 0.4ms

	uvec4 color = uvec4(0,0,0,0);
	
	uint first = ssRGBA[2 * pixelID + 0];
	uint second = ssRGBA[2 * pixelID + 1];
	if((first & 0xFFFF) <= 255) // can trust this
	{
		color.a = first & 0xFFFF;
		color.b = first >> 16;
		color.g = second & 0xFFFF;
		color.r = second >> 16;
	}
	
	color += ssRGBA_slow[pixelID];
	color.rgb /= color.a;

	if(color.a == 0){
		return uvec4(30, 50, 60, 255);
		// return uvec4(255, 255, 255, 255);
	}

	if(color.a == 0xffffffff){
		color = uvec4(255, 255, 255, 255);
	}

	return color;
}
#else
uvec4 rgbAt(int pixelID){

	// decoding the colors is a whole lot faster when you interpret the 
	// ssRGBA buffer as 2x32 bit uints, rather than 1x64 bit uint, on the GTX 1060.
	// 3.0ms vs 0.4ms
	uint first = ssRGBA[2 * pixelID + 0];
	uint second = ssRGBA[2 * pixelID + 1];

	uint a = first & 1023;
	uint r = (second >> 14) / a;
	uint g_least = (first >> 28) & 15;
	uint g_most = second & 16383;
	uint g = ((g_most << 4) | g_least) / a;
	uint b = ((first >> 10) & 262143) / a;

	// The uint64 version
	// uint64_t rgba = ssRGBA[pixelID];

	// uint64_t a = uint(rgba & 0x3ffUL);

	// //r18-g18-b18-a10 bit counters
	// uint r = uint(((rgba >> 46ul) & 0x3ffffUL) / a);
	// uint g = uint(((rgba >> 28ul) & 0x3ffffUL) / a);
	// uint b = uint(((rgba >> 10ul) & 0x3ffffUL) / a);

	if(a == 0){
		return uvec4(30, 50, 60, 255);
	}

	uvec4 icolor = uvec4(r, g, b, a);

	if(a == 0xffffffff){
		icolor = uvec4(255, 255, 255, 255);
	}

	return icolor;
}
#endif

uvec4 depthAt(int pixelID){
	float depth = uintBitsToFloat(ssDepthbuffer[pixelID]);
	
	depth = (depth - 3000) / 4000;
	depth = pow(depth, 0.7);

	uint c = uint(255 * depth);

	uvec4 icolor = uvec4(c, c, c, 255);

	return icolor;
}

void main(){
	uvec2 id = gl_LocalInvocationID.xy;
	id.x += gl_WorkGroupSize.x * gl_WorkGroupID.x;
	id.y += gl_WorkGroupSize.y * gl_WorkGroupID.y;

	ivec2 imgSize = imageSize(uOutput);

	if(id.x >= imgSize.x){
		return;
	}

	ivec2 pixelCoords = ivec2(id);
	ivec2 sourceCoords = ivec2(id);
	int pixelID = sourceCoords.x + sourceCoords.y * imgSize.x;

	//uvec4 icolor = depthAt(pixelID);
	uvec4 icolor = rgbAt(pixelID);

	icolor = uvec4(icolor.r, icolor.g, icolor.b, 255);
	imageStore(uOutput, pixelCoords, icolor);
	ssRGBA[2 * pixelID + 0] = 0x0;
	ssRGBA[2 * pixelID + 1] = 0x0;
	ssRGBA_slow[pixelID] = uvec4(0,0,0,0);
	ssDepthbuffer[pixelID] = 0xffffffffu;
}

