
#version 450

#extension GL_ARB_gpu_shader_int64 : enable
#extension GL_NV_shader_atomic_int64 : enable
#extension GL_NV_gpu_shader5 : enable
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_KHR_shader_subgroup_vote : require
#extension GL_KHR_shader_subgroup_clustered : require

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
	//int64_t ssFramebuffer[];
	uint64_t ssFramebuffer[];
};

layout(rgba8ui, binding = 0) uniform uimage2D uOutput;

layout(binding = 1) uniform sampler2D uGradient;


uvec4 colorAt(int pixelID){
	uint64_t val64 = ssFramebuffer[pixelID];
	uint ucol = uint(val64 & 0x00FFFFFFUL);

	if(ucol == 0){
		return uvec4(0, 0, 0, 255);
	}

	vec4 color = 255.0 * unpackUnorm4x8(ucol);
	uvec4 icolor = uvec4(color);

	return icolor;
}

void main(){
	uvec2 id = gl_LocalInvocationID.xy;
	id.x += gl_WorkGroupSize.x * gl_WorkGroupID.x;
	id.y += gl_WorkGroupSize.y * gl_WorkGroupID.y;

	{
		ivec2 imgSize = imageSize(uOutput);

		if(id.x >= imgSize.x){
			return;
		}

		ivec2 pixelCoords = ivec2(id);
		ivec2 sourceCoords = ivec2(id);
		int pixelID = sourceCoords.x + sourceCoords.y * imgSize.x;

		
		uvec4 icolor = colorAt(pixelID);

		imageStore(uOutput, pixelCoords, icolor);

		// rgb(60, 50, 30) = 0x3c321e (0xRGB)
		ssFramebuffer[pixelID] = 0xffffffffff3c321eUL;
	}


	// { // n x n pixel
	// 	ivec2 imgSize = imageSize(uOutput);

	// 	ivec2 pixelCoords = ivec2(id);

	// 	int window = 1;
	// 	int edlWindow = 1;

	// 	float closestDepth = 1000000.0;
	// 	uint32_t closestPointColor = 0;

	// 	for(int ox = -window; ox <= window; ox++){
	// 	for(int oy = -window; oy <= window; oy++){

	// 		int pixelID = (pixelCoords.x + ox) + (pixelCoords.y + oy) * imgSize.x;

	// 		uint64_t data = ssFramebuffer[pixelID];
	// 		uint32_t uDepth = uint32_t(data >> 32l);
	// 		uint32_t pointColor = uint32_t(data & 0xffffffffl);
	// 		float depth = uintBitsToFloat(uDepth);

	// 		if(depth > 0.0 && depth < closestDepth){
	// 			closestDepth = depth;
	// 			closestPointColor = pointColor;
	// 		}
			
	// 	}
	// 	}

	// 	uint32_t ucol = closestPointColor;

	// 	vec4 color = 255.0 * unpackUnorm4x8(ucol);
	// 	uvec4 icolor = uvec4(color);

	// 	// imageAtomicExchange(uOutput, pixelCoords, color);

	// 	imageStore(uOutput, pixelCoords, icolor);

	// 	int pixelID = (pixelCoords.x) + (pixelCoords.y) * imgSize.x;
	// 	ssFramebuffer[pixelID] = 0xffffffffff3c321eUL;

	// }

}
