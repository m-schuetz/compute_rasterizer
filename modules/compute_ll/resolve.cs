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

struct Node{
	//int pointID;
	uint colors;
	uint depth;
	int next;
};

layout(rgba8ui, binding = 0) uniform uimage2D uOutput;
layout(r32f, binding = 2) uniform image2D uDepth;

layout(binding = 5) uniform sampler1D uGradient;

layout (std430, binding=0) buffer shader_data {
	Vertex vertices[];
};

layout (std430, binding=2) buffer deph_data {
	uint ssDepth[];
};

layout (std430, binding=4) buffer header_data{
	int headPointers[];
};

layout (std430, binding=5) buffer nodes_data{
	Node nodes[];
};

layout (std430, binding=6) buffer meta_data{
	int counter;
};

//layout(location = 3) uniform ivec2 uOffset;

void main(){

	uvec2 id = gl_LocalInvocationID.xy;
	id.x += gl_WorkGroupSize.x * gl_WorkGroupID.x;
	id.y += gl_WorkGroupSize.y * gl_WorkGroupID.y;

	ivec2 imgSize = imageSize(uOutput);
	ivec2 pixelCoords = ivec2(id);
	//pixelCoords = pixelCoords + uOffset;
	int pixelID = pixelCoords.x + pixelCoords.y * imgSize.x;

	int pointer = headPointers[pixelID];
	headPointers[pixelID] = -1;

	vec4 col = vec4(0.0, 0.0, 0.0, 0.0);

	uint idepth = ssDepth[pixelID];
	int i = 0;
	//while(pointer >= 0 && i < 16384){
	while(pointer >= 0 && i < 10){
		Node node = nodes[pointer];
		//Vertex vertex = vertices[node.pointID];

		if(node.depth <= (float(idepth) * 1.005)){
			col = col + vec4(unpackUnorm4x8(node.colors).rgb, 1.0);
		}

		pointer = node.next;
		i++;
	}


	col.rgb = col.rgb / col.w;
	//col = col + 0.3;

	if(i == 0){
		col = vec4(0, 0, 0, 0);
	}
	

	//col.rgb = col.rgb / col.w;
	ivec4 icol = ivec4((col.rgb * 255.0), 255);




	//------------
	// colorize by fragment count
	//------------
	//if(i > 0){
	//	float u = float(i);
	//	u = log2(u) / 14.0;

	//	vec4 grad = texture(uGradient, u);
	//	icol.rgb = ivec3(grad.rgb * 255.0);
	//}else{
	//	icol = ivec4(255, 255, 255, 255);
	//}

	
	//icol.rgb = ivec3(idepth / 30, 0, 0);

	//ssDepth[pixelID] = 0xFFFFFF;
	ssDepth[pixelID] = 0xEFFFFFFF; // ???

	imageStore(uOutput, pixelCoords, icol);

	counter = 0;

}