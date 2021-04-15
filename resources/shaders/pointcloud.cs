
#version 450

#extension GL_ARB_gpu_shader_int64 : enable
#extension GL_NV_shader_atomic_int64 : enable

layout(local_size_x = 128, local_size_y = 1) in;

struct Vertex{
	float x;
	float y;
	float z;
	uint colors;
	float ar;
	float ag;
	float ab;
	float aw;
};

layout(location = 0) uniform mat4 uTransform;


layout (std430, binding=0) buffer point_data {
	Vertex vertices[];
};

layout(rgba8ui, binding = 0) uniform uimage2D uOutput;

void main(){
	uint workGroupSize = gl_WorkGroupSize.x * gl_WorkGroupSize.y;
	uint globalID = gl_WorkGroupID.x * workGroupSize
		+ gl_WorkGroupID.y * gl_NumWorkGroups.x * workGroupSize
		+ gl_LocalInvocationIndex;

	//globalID = globalID + uOffset;

	Vertex v = vertices[globalID];

	vertices[globalID].aw += 1.0;
	//vertices[globalID].aw = 0.0;

	vec4 pos = uTransform * vec4(v.x, v.y, v.z, 1.0);
	pos.xyz = pos.xyz / pos.w;

	if(pos.w <= 0.0 || pos.x < -1.0 || pos.x > 1.0 || pos.y < -1.0 || pos.y > 1.0){
		return;
	}

	ivec2 uImageSize = imageSize(uOutput);
	vec2 imgPos = (pos.xy * 0.5 + 0.5) * uImageSize;
	ivec2 pixelCoords = ivec2(imgPos);

	uvec4 color = uvec4(unpackUnorm4x8(v.colors) * 255.0);

	//uint w = uint(vertices[globalID].aw / 100.0);
	uint w = uint(log(vertices[globalID].aw) * 10.0);
	color.r = w;
	color.g = w;
	color.b = w;

	imageStore(uOutput, pixelCoords, color);
	//imageStore(uOutput, pixelCoords + ivec2(1, 0), color);
	//imageStore(uOutput, pixelCoords + ivec2(0, 1), color);
	//imageStore(uOutput, pixelCoords + ivec2(1, 1), color);
}

