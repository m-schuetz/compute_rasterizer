#version 450

//layout(local_size_x = 8, local_size_y = 8) in;
layout(local_size_x = 16, local_size_y = 16) in;

struct VertexS{
	float ux;
	float uy;
	float uz;
	uint color;
};

struct VertexT{
	float ux;
	float uy;
	float uz;
	uint color;
	uint index;
};


layout(rgba8ui, binding = 0) uniform uimage2D uIndices;

layout(std430, binding = 1) buffer ssIndirectCommand{
	uint count;
	uint primCount;
	uint firstIndex;
	uint baseVertex;
	uint baseInstance;
};

layout(std430, binding = 2) buffer ssTarget{
	VertexT targetBuffer[];
};

layout(std430, binding = 3) buffer ssSource0{
	VertexS vbo0[];
};

layout(std430, binding = 4) buffer ssSource1{
	VertexS vbo1[];
};

layout(std430, binding = 5) buffer ssSource2{
	VertexS vbo2[];
};

layout(std430, binding = 6) buffer ssSource3{
	VertexS vbo3[];
};

layout(std430, binding = 7) buffer ssSource4{
	VertexS vbo4[];
};

layout(std430, binding = 8) buffer ssSource5{
	VertexS vbo5[];
};

layout(std430, binding = 9) buffer ssSource6{
	VertexS vbo6[];
};

layout(std430, binding = 10) buffer ssSource7{
	VertexS vbo7[];
};

VertexT getVertex(uint vertexID){
	uint maxPointsPerBuffer = 134 * 1000 * 1000;

	VertexS vs;
	if(vertexID < 1u * maxPointsPerBuffer){
		vs = vbo0[vertexID];
	}else if(vertexID < 2u * maxPointsPerBuffer){
		vs = vbo1[vertexID - 1u * maxPointsPerBuffer];
	}else if(vertexID < 3u * maxPointsPerBuffer){
		vs = vbo2[vertexID - 2u * maxPointsPerBuffer];
	}else if(vertexID < 4u * maxPointsPerBuffer){
		vs = vbo3[vertexID - 3u * maxPointsPerBuffer];
	}else if(vertexID < 5u * maxPointsPerBuffer){
		vs = vbo4[vertexID - 4u * maxPointsPerBuffer];
	}else if(vertexID < 6u * maxPointsPerBuffer){
		vs = vbo5[vertexID - 5u * maxPointsPerBuffer];
	}else if(vertexID < 7u * maxPointsPerBuffer){
		vs = vbo6[vertexID - 6u * maxPointsPerBuffer];
	}else if(vertexID < 8u * maxPointsPerBuffer){
		vs = vbo7[vertexID - 7u * maxPointsPerBuffer];
	}

	VertexT vt;
	vt.ux = vs.ux;
	vt.uy = vs.uy;
	vt.uz = vs.uz;
	vt.color = vs.color;
	vt.index = vertexID;

	return vt;
}

void main() {
	
	uvec2 id = gl_LocalInvocationID.xy + gl_WorkGroupSize.xy * gl_WorkGroupID.xy;
	ivec2 pixelCoords = ivec2(id);

	// operate on grids of 2x2 samples
	pixelCoords = pixelCoords * 2;
	
	// load 2x2 samples
	uvec4 vVertexID0 = imageLoad(uIndices, pixelCoords);
	uvec4 vVertexID1 = imageLoad(uIndices, pixelCoords + ivec2(1, 0));
	uvec4 vVertexID2 = imageLoad(uIndices, pixelCoords + ivec2(0, 1));
	uvec4 vVertexID3 = imageLoad(uIndices, pixelCoords + ivec2(1, 1));

	// transform to indices for each sample
	uint vertexID0 = vVertexID0.r | (vVertexID0.g << 8) | (vVertexID0.b << 16) | (vVertexID0.a << 24);
	uint vertexID1 = vVertexID1.r | (vVertexID1.g << 8) | (vVertexID1.b << 16) | (vVertexID1.a << 24);
	uint vertexID2 = vVertexID2.r | (vVertexID2.g << 8) | (vVertexID2.b << 16) | (vVertexID2.a << 24);
	uint vertexID3 = vVertexID3.r | (vVertexID3.g << 8) | (vVertexID3.b << 16) | (vVertexID3.a << 24);

	// ignore white background
	// vertexID0 = (vVertexID0.r + vVertexID0.g + vVertexID0.b) == (255 * 3) ? 0 : vertexID0;
	// vertexID1 = (vVertexID1.r + vVertexID1.g + vVertexID1.b) == (255 * 3) ? 0 : vertexID1;
	// vertexID2 = (vVertexID2.r + vVertexID2.g + vVertexID2.b) == (255 * 3) ? 0 : vertexID2;
	// vertexID3 = (vVertexID3.r + vVertexID3.g + vVertexID3.b) == (255 * 3) ? 0 : vertexID3;

	// ignore black background
	vertexID0 = (vVertexID0.r + vVertexID0.g + vVertexID0.b) == 0 ? 0 : vertexID0;
	vertexID1 = (vVertexID1.r + vVertexID1.g + vVertexID1.b) == 0 ? 0 : vertexID1;
	vertexID2 = (vVertexID2.r + vVertexID2.g + vVertexID2.b) == 0 ? 0 : vertexID2;
	vertexID3 = (vVertexID3.r + vVertexID3.g + vVertexID3.b) == 0 ? 0 : vertexID3;

	uint vCount = 0;
	bool v0Visible = false;
	bool v1Visible = false;
	bool v2Visible = false;
	bool v3Visible = false;

	// check visible and avoid duplicates
	if(vertexID0 != 0){
		vCount++;
		v0Visible = true;
	}
	if(vertexID1 != 0 && vertexID1 != vertexID0){
		vCount++;
		v1Visible = true;
	}
	if(vertexID2 != 0 && vertexID1 != vertexID2 && vertexID0 != vertexID2){
		vCount++;
		v2Visible = true;
	}
	if(vertexID3 != 0 && vertexID2 != vertexID3 && vertexID1 != vertexID3 && vertexID0 != vertexID3){
		vCount++;
		v3Visible = true;
	}

	// reserve space for visible points in reproject vertex buffer
	uint first = atomicAdd(count, vCount);

	// add all the visible points to the reproject vertex buffer
	uint offset = 0;
	if(v0Visible){
		VertexT vt = getVertex(vertexID0);
		targetBuffer[first + offset] = vt;
		offset++;
	}
	if(v1Visible){
		VertexT vt = getVertex(vertexID1);
		targetBuffer[first + offset] = vt;
		offset++;
	}
	if(v2Visible){
		VertexT vt = getVertex(vertexID2);
		targetBuffer[first + offset] = vt;
		offset++;
	}
	if(v3Visible){
		VertexT vt = getVertex(vertexID3);
		targetBuffer[first + offset] = vt;
	}
	
}






