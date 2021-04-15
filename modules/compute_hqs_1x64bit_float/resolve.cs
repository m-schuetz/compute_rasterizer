
#version 450

#extension GL_ARB_gpu_shader_int64 : enable
#extension GL_NV_shader_atomic_int64 : enable
#extension GL_NV_gpu_shader5 : enable

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
	f16vec4 ssRGBA[];
};

layout(rgba8ui, binding = 0) uniform uimage2D uOutput;

layout(binding = 1) uniform sampler2D uGradient;

uvec4 rgbAt(int pixelID){

	// decoding the colors is a whole lot faster when you interpret the 
	// ssRGBA buffer as 2x32 bit uints, rather than 1x64 bit uint, on the GTX 1060.
	// 3.0ms vs 0.4ms

	f16vec4 rgba = ssRGBA[pixelID];

	uvec4 icolor = uvec4(uvec3((rgba.rgb / rgba.a) * 255.0f), uint(rgba.a));

	// The uint64 version
	// uint64_t rgba = ssRGBA[pixelID];

	// uint64_t a = uint(rgba & 0x3ffUL);

	// //r18-g18-b18-a10 bit counters
	// uint r = uint(((rgba >> 46ul) & 0x3ffffUL) / a);
	// uint g = uint(((rgba >> 28ul) & 0x3ffffUL) / a);
	// uint b = uint(((rgba >> 10ul) & 0x3ffffUL) / a);

	if(icolor.a == 0){
		return uvec4(30, 50, 60, 255);
	}

	if(icolor.a == 0xffffffff){
		icolor = uvec4(255, 255, 255, 255);
	}

	return icolor;
}

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
	ssRGBA[pixelID] = f16vec4(0,0,0,0);
	ssDepthbuffer[pixelID] = 0xffffffffu;
}

