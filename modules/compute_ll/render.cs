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

struct Node{
	//int pointID;
	uint colors;
	uint depth;
	int next;
};

layout(location = 2) uniform mat4 uTransform;
layout(location = 3) uniform int uOffset;
layout(location = 4) uniform ivec2 uImageSize;


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
	int ssCounter;
};


void main(){
	uint globalID = gl_GlobalInvocationID.x;

	int currentCount = atomicAdd(ssCounter, 1);

	Vertex v = vertices[globalID];

	vec4 pos = uTransform * vec4(v.x, v.y, v.z, 1.0);
	pos.xyz = pos.xyz / pos.w;

	if(pos.w <= 0.0 || pos.x < -1.0 || pos.x > 1.0 || pos.y < -1.0 || pos.y > 1.0){
		return;
	}

	ivec2 imgSize = uImageSize;
	vec2 imgPos = (pos.xy * 0.5 + 0.5) * imgSize;
	ivec2 pixelCoords = ivec2(imgPos);
	int pixelID = pixelCoords.x + pixelCoords.y * imgSize.x;

	float depth = pos.w;
	uint idepth = uint(depth * 1000.0);

	uint oldDepth = atomicMin(ssDepth[pixelID], idepth);

	if(idepth < (float(oldDepth) * 1.005))
	{
		int previousCounter = atomicExchange(headPointers[pixelID], currentCount);

		Node node;
		//node.pointID = int(globalID);
		node.colors = v.colors;
		node.depth = idepth;
		node.next = previousCounter;
		nodes[currentCount] = node;
	}


	// debug purposes
	//vec4 color = 255.0 * unpackUnorm4x8(v.colors);
	//ivec4 icol = ivec4(color);
	//imageStore(uOutput, pixelCoords, icol);

}