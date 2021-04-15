
#version 450

#extension GL_ARB_gpu_shader_int64 : enable
#extension GL_NV_shader_atomic_int64 : enable

layout(local_size_x = 16, local_size_y = 16) in;

layout (std430, binding=1) buffer framebuffer_data {
	uint ssFramebuffer[];
};

layout(rgba8ui, binding = 0) uniform uimage2D uOutput;

uvec4 colorAt(int pixelID){
	uint ucol = ssFramebuffer[pixelID];

	if(ucol == 0){
		return uvec4(30, 50, 60, 255);
	}

	vec4 color = 255.0 * unpackUnorm4x8(ucol);
	uvec4 icolor = uvec4(color);

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
	int pixelID = pixelCoords.x + pixelCoords.y * imgSize.x;

	uvec4 icolor = colorAt(pixelID);

	imageStore(uOutput, pixelCoords, icolor);

	ssFramebuffer[pixelID] = 0;
}

