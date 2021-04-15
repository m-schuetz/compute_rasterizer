
#version 450

#extension GL_ARB_gpu_shader_int64 : enable
#extension GL_NV_shader_atomic_int64 : enable

// 
// Transfers the rendering results from render.cs from an SSBO into an OpenGL texture.
// 


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

layout (std430, binding=1) buffer framebuffer_data {
	uint64_t ssFramebuffer[];
};

layout(rgba8ui, binding = 0) uniform uimage2D uOutput;

layout(binding = 1) uniform sampler2D uGradient;
uniform int uSupersampleSize;

uvec4 colorAtSample(int x, int y){
	ivec2 imgSize = imageSize(uOutput);
	int pixelID = x + y * imgSize.x * uSupersampleSize;
	uint64_t val64 = ssFramebuffer[pixelID];

	uint ucol = uint(val64 & 0x00FFFFFFUL);

	if(ucol == 0x00ffffff){
		return uvec4(0, 0, 0, 0);
	}

	vec4 color = 255.0 * unpackUnorm4x8(ucol);
	uvec4 ucolor = uvec4(color);

	ucolor.a = 1;

	return ucolor;
}

uvec4 colorAt(int x, int y){

	uvec4 sum = uvec4(0, 0, 0, 0);

	for(int i = 0; i < uSupersampleSize; i++){
	for(int j = 0; j < uSupersampleSize; j++){
		sum += colorAtSample(
			uSupersampleSize * x + i, 
			uSupersampleSize * y + j);
	}
	}
	// sum += colorAtSample(2 * x + 0, 2 * y + 0);
	// sum += colorAtSample(2 * x + 0, 2 * y + 1);
	// sum += colorAtSample(2 * x + 1, 2 * y + 0);
	// sum += colorAtSample(2 * x + 1, 2 * y + 1);

	uvec4 color;
	
	if(sum.a == 0){
		color = uvec4(30, 50, 60, 255);
	}else{
		color = sum / sum.a;
	}

	return color;
}

void main(){
	uvec2 id = gl_LocalInvocationID.xy;
	id.x += gl_WorkGroupSize.x * gl_WorkGroupID.x;
	id.y += gl_WorkGroupSize.y * gl_WorkGroupID.y;

	ivec2 pixelCoords = ivec2(id);

	uvec4 icolor = colorAt(pixelCoords.x, pixelCoords.y);

	imageStore(uOutput, pixelCoords, icolor);

}

