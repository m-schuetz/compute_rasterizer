
#version 450

#extension GL_ARB_gpu_shader_int64 : enable
#extension GL_NV_shader_atomic_int64 : enable
#extension GL_NV_gpu_shader5 : enable
#extension GL_NV_shader_thread_group : require
#extension GL_NV_shader_thread_shuffle : require
#extension GL_ARB_shader_group_vote : require
#extension GL_ARB_shader_ballot : require
#extension GL_ARB_gpu_shader_int64: require

// To disable the robust behavior (gets rid of artefacts), undefine the ROBUST flag
// (must be done in resolve pass too)
// To disable the block-wide attempt for a fast path via shared memory,
// set blockWideFastPath to false

layout(local_size_x = 128, local_size_y = 1) in;

struct Vertex{
	float x;
	float y;
	float z;
	uint colors;
};

layout(location = 0) uniform mat4 uTransform;

layout (std430, binding = 0) buffer point_data {
	Vertex vertices[];
};

layout (std430, binding = 1) buffer framebuffer_data {
	int64_t ssFramebuffer[];
};

layout (std430, binding = 2) buffer depthbuffer_data {
	uint ssDepthbuffer[];
};

layout (std430, binding = 3) buffer rgba_data {
	uint64_t ssRGBA[];
};

layout (std430, binding = 4) buffer rgba_data_slow {
	uint64_t ssRGBA_slow[];
};

uniform ivec2 uImageSize;

layout(binding = 0) uniform sampler2D uGradient;

uvec2 encodeColor(uint r, uint g, uint b)
{
	return uvec2(r << 16 | g, b << 16 | 1);
}

#define ROBUST
#ifdef ROBUST
uvec4 genColor(uint r, uint g, uint b, uint a)
{
	return uvec4(r,g,b,a);
}

uvec4 decodeColor(uvec2 color)
{
	uvec4 res;
	res.r = (color.x >> 16);
	res.g = (color.x & 0xffff);
	res.b = (color.y >> 16);
	res.a = (color.y & 0xffff);
	return res;
}

void writeGlobalColor(uvec4 color, uint pixelID)
{
	uint64_t rgba = (uint64_t(color.r) << 48) | (uint64_t(color.g) << 32) | (uint64_t(color.b) << 16) | uint64_t(color.a);
	uint64_t old = atomicAdd(ssRGBA[pixelID], rgba);
	uint curr_count = uint(old & 0xFFFF);
	if((curr_count + color.a) >= 256) 
	{
		if(curr_count <= 255)
		{
		 	atomicExchange(ssRGBA[pixelID], 0);
		 	color.r += uint(old >> 48) & 0xFFFF;
		 	color.g += uint(old >> 32) & 0xFFFF;
		 	color.b += uint(old >> 16) & 0xFFFF;
		 	color.a += curr_count;
		}
		int64_t rgba1 = int64_t(color.r) | int64_t(color.g) << 32;
		int64_t rgba2 = int64_t(color.b) | int64_t(color.a) << 32;
		atomicAdd(ssRGBA_slow[2*pixelID+0], rgba1);
		atomicAdd(ssRGBA_slow[2*pixelID+1], rgba2);
	}
}
#else
uvec3 genColor(uint r, uint g, uint b, uint a)
{
	return uvec3(r,g,b);
}

uvec3 decodeColor(uvec2 color)
{
	uint count = max(1, color.y & 0xff);
	uvec3 res;
	res.r = (color.x >> 16) / count;
	res.g = (color.x & 0xffff) / count;
	res.b = (color.y >> 16) / count;
	return res;
}

void writeGlobalColor(uvec3 color, uint pixelID)
{
	int64_t rgba = (int64_t(color.r) << 46) | (int64_t(color.g) << 28) | (int64_t(color.b) << 10) | 1;
	int64_t old = atomicAdd(ssRGBA[pixelID], rgba);
}
#endif

shared uvec4 inc;
shared uvec2 foo[3];

void main()
{
	uint globalID = gl_GlobalInvocationID.x;

	Vertex v = vertices[globalID];

	vec4 pos = uTransform * vec4(v.x, v.y, v.z, 1.0);
	pos.xyz = pos.xyz / pos.w;

	if(pos.w <= 0.0 || pos.x < -1.0 || pos.x > 1.0 || pos.y < -1.0 || pos.y > 1.0){
		return;
	}

	vec2 imgPos = (pos.xy * 0.5 + 0.5) * uImageSize;
	ivec2 pixelCoords = ivec2(imgPos);
	int pixelID = pixelCoords.x + pixelCoords.y * uImageSize.x;

	float depth = pos.w;
	float depthInBuffer = uintBitsToFloat(ssDepthbuffer[pixelID]);

	// r18-g18-b18-a10 bit counters

	uint r = 0, g = 0, b = 0;
	uvec2 sum = uvec2(0, 0);

	bool depthFits = depth <= depthInBuffer * 1.01;
	if(depthFits)
	{
		r = (v.colors >>  0) & 0xFF;
		g = (v.colors >>  8) & 0xFF;
		b = (v.colors >> 16) & 0xFF;
		sum = encodeColor(r,g,b);
	}

	bool blockWideFastPath = true;
	int leaderID = shuffleNV(pixelID, 0, 32);
	uint canParticipate = ballotThreadNV(leaderID == pixelID || !depthFits); // pixel matches or nothing to add
	uint goodDepth = ballotThreadNV(depthFits); // something to add
	
	// optimize if all threads can participate and at least one valid depth entry
	bool optimized = (canParticipate == 0xffffffff) && (goodDepth > 0);

	uint warpID = gl_LocalInvocationID.x/32;
	bool warpLeader = (gl_ThreadInWarpNV == 0);
	bool groupLeader = (gl_LocalInvocationID.x == 0);

	if(blockWideFastPath)
	{
		if(warpLeader)
		{
			inc[warpID] = optimized ? pixelID : 0xFFFFFFFF; // if optimized access, write target pixel
		}
		barrier();
	}

	if(optimized)	
	{
		sum = sum + shuffleDownNV(sum, 16, 32);
		sum = sum + shuffleDownNV(sum,  8, 32);
		sum = sum + shuffleDownNV(sum,  4, 32);
		sum = sum + shuffleDownNV(sum,  2, 32);
		sum = sum + shuffleDownNV(sum,  1, 32);

		if(blockWideFastPath && inc.x == pixelID && inc.y == pixelID && inc.z == pixelID && inc.w == pixelID) // everybody wamts same pixel!
		{
			if(warpLeader && !groupLeader)
			{
				foo[warpID - 1] = sum; //write your sum!
			}
			barrier(); //synchronize!

			if(groupLeader)
			{
				sum = (foo[0] + foo[1] + foo[2] + sum); // single thread reads partial sums
				writeGlobalColor(decodeColor(sum), pixelID);
			}
		}
		else if(warpLeader)
		{
			writeGlobalColor(decodeColor(sum), pixelID);
		}
	}
	else 
	{
		// last chance for some optimizing
		uint neighborPixelID = shuffleXorNV(pixelID,  1, 32);
		uvec2 neighborSum = shuffleXorNV(sum, 1, 32);
		if(neighborPixelID == pixelID)
		{
			if(gl_LocalInvocationID.x%2 == 0 && ((goodDepth >> gl_LocalInvocationID.x) & 3) > 0)
			{
				writeGlobalColor(decodeColor(neighborSum + sum), pixelID);
			}
		}
		else if(depthFits) // every thread for himself
		{
			writeGlobalColor(genColor(r,g,b,1), pixelID);
		}
	}
}

