
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
layout(location = 1) uniform int uOffset;

layout (std430, binding=0) buffer point_data {
	Vertex vertices[];
};

layout(rgba8ui, binding = 0) uniform uimage2D uOutput;

void main(){
	uint workGroupSize = gl_WorkGroupSize.x * gl_WorkGroupSize.y;
	uint globalID = gl_WorkGroupID.x * workGroupSize
		+ gl_WorkGroupID.y * gl_NumWorkGroups.x * workGroupSize
		+ gl_LocalInvocationIndex;
	
	globalID = globalID * 2 + uOffset;

	Vertex v1 = vertices[globalID];
	Vertex v2 = vertices[globalID + 1];

	vec4 p1 = uTransform * vec4(v1.x, v1.y, v1.z, 1.0);
	vec4 p2 = uTransform * vec4(v2.x, v2.y, v2.z, 1.0);

	barrier();

	if(p1.w > p2.w){
		vertices[globalID] = v2;
		vertices[globalID + 1] = v1;
	}

}

