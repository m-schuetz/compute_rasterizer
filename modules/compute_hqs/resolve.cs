
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

layout (std430, binding=3) buffer rg_data {
	uint ssRGBA[];
};

layout(rgba8ui, binding = 0) uniform uimage2D uOutput;

layout(binding = 1) uniform sampler2D uGradient;

uvec4 rgbAt(int pixelID){
	uint a = ssRGBA[4 * pixelID + 2];
	uint r = ssRGBA[4 * pixelID + 1] / a;
	uint g = ssRGBA[4 * pixelID + 0] / a;
	uint b = ssRGBA[4 * pixelID + 3] / a;

	if(a == 0){
		return uvec4(30, 50, 60, 255);
	}

	uvec4 icolor = uvec4(r, g, b, a);

	if(a == 0xffffffff){
		icolor = uvec4(255, 255, 255, 255);
	}

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

	uvec4 icolor = rgbAt(pixelID);

	imageStore(uOutput, pixelCoords, icolor);
	ssRGBA[4 * pixelID + 0] = 0;
	ssRGBA[4 * pixelID + 1] = 0;
	ssRGBA[4 * pixelID + 2] = 0;
	ssRGBA[4 * pixelID + 3] = 0;

	ssDepthbuffer[pixelID] = 0xffffffff;
}

