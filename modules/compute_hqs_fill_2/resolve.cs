
#version 450

#extension GL_ARB_gpu_shader_int64 : enable
#extension GL_NV_shader_atomic_int64 : enable

layout(local_size_x = 32, local_size_y = 1) in;
// layout(local_size_x = 16, local_size_y = 16) in;

struct Vertex{
	float x;
	float y;
	float z;
	uint colors;
};

struct RGBA{
	uint g;
	uint r;
	uint a;
	uint b;
};

layout (std430, binding=0) buffer point_data {
	Vertex vertices[];
};

layout (std430, binding=2) buffer depthbuffer_data {
	uint64_t ssDepthbuffer[];
};

layout (std430, binding=3) buffer rg_data {
	RGBA ssRGBA[];
};


layout(rgba8ui, binding = 0) uniform uimage2D uOutput;
layout(r32ui, binding = 1) uniform uimage2D uOutDepth;

uvec4 rgbAt(int pixelID){
	RGBA rgba = ssRGBA[pixelID];

	uint a = rgba.a;
	uint r = rgba.r / a;
	uint g = rgba.g / a;
	uint b = rgba.b / a;

	uvec4 icolor;
	// if(a == 0){
	// 	icolor = uvec4(255, 255, 255, 0);
	// }else{
		// icolor = uvec4(0, 0, b, a);
		icolor = uvec4(r, g, b, a);

		if(a == 0xffffffff){
			icolor = uvec4(255, 255, 255, 0);
		}
	// }

	return icolor;
}

uvec4 depthAt(int pixelID){
	uint64_t val64 = ssDepthbuffer[pixelID];
	
	float depth = float(double(val64) / 1000.0);
	//depth = pow(depth + 100.0, 0.7);
	depth = (depth - 3000) / 4000;
	depth = pow(depth, 0.7);

	uint c = uint(255 * depth);

	uvec4 icolor = uvec4(c, c, c, 255);

	return icolor;
}

void main(){
	int pixelID = int(gl_GlobalInvocationID.x);
	
	ivec2 imgSize = imageSize(uOutput);

	ivec2 id = ivec2(
		pixelID % imgSize.x,
		pixelID / imgSize.x
	);

	ivec2 pixelCoords = ivec2(id);

	uvec4 icolor = rgbAt(pixelID);
	uint64_t depth = ssDepthbuffer[pixelID];
	uvec4 udepth = uvec4(depth, 0, 0, 0);

	imageStore(uOutput, pixelCoords, icolor);
	imageStore(uOutDepth, pixelCoords, udepth);
}

