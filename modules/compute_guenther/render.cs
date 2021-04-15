#version 450

#extension GL_ARB_gpu_shader_int64 : enable
#extension GL_NV_shader_atomic_int64 : enable
#extension GL_NV_gpu_shader5 : enable
#extension GL_KHR_shader_subgroup_vote : require

// 
// This compute shader renders point clouds with 1 to 4 pixels per point.
// It is essentially vertex-shader, rasterizer and fragment-shader in one.
// Rendering happens by encoding depth values in the most significant bits, 
// and color values in the least significant bits of a 64 bit integer.
// atomicMin then writes the fragment with the smallest depth value into an SSBO.
// 
// The SSBO is essentially our framebuffer with color and depth attachment combined.
// A second compute shader, resolve.cs, then transfers the values from the SSBO to an 
// actual OpenGL texture.
// 

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

layout (std430, binding = 1) buffer depth_data {
	volatile coherent uint32_t depthLockBuffer[];
};

layout (std430, binding = 2) buffer color_data {
	uint32_t colorBuffer[];
};

shared uint running[4];

uniform ivec2 uImageSize;

layout(binding = 0) uniform sampler2D uGradient;

void main(){
	uint globalID = gl_GlobalInvocationID.x;
	Vertex v = vertices[globalID];
	vec4 pos = uTransform * vec4(v.x, v.y, v.z, 1.0);
	pos.xyz = pos.xyz / pos.w;

	//No threads must exit. Busy loop depends on threads remaining active.
	bool write = true;
	if(pos.w <= 0.0 || pos.x < -1.0 || pos.x >= 1.0 || pos.y < -1.0 || pos.y >= 1.0){
		write = false;
	}

	vec2 imgPos = (pos.xy * 0.5 + 0.5) * uImageSize;
	ivec2 pixelCoords = ivec2(imgPos);
	int pixelID = pixelCoords.x + pixelCoords.y * uImageSize.x;
	uint depth_int = floatBitsToInt(pos.w);

	// Remaining threads compute their current depth, lock and the
	// target value to switch it out with for the rendered point
	uint D, depth_swap;
	if(write) 
	{
		D = depthLockBuffer[pixelID];
		depth_swap = depth_int | (1U << 31);
	}

	// Locks can always lead to deadlock on the GPU, unless
	// handled with care. No divergent branch is allowed to
	// spin endlessly while the other holds a lock. The only
	// way to guarantee reconvergence is a warp-wide sync, 
	// which happens e.g. when ballot is called. Hence,
	// the entire warp must pariticpate in the loop and keep
	// calling ballot to enforce synchronization at the bottom
	// of the loop. Loop ends when no thread in the warp 
	// has anything left to update. 
	
	// Vote to see if someone wants to keep looping
	// and also synchronize all threads in the warp.
	while(subgroupAny(write))
	{
		// Only threads that still update will enter here
		if(write)
		{
			D = depthLockBuffer[pixelID];
			uint depth_test = D & 0x7fffffffU;
			// Early depth test
			if(depth_int >= depth_test)
			{
				// Early depth test failed
				// This thread no longer needs to write its result
			 	write = false;
			}
			else
			{
				// Try to lock the current pixel
				D = atomicCompSwap(depthLockBuffer[pixelID], depth_test, depth_swap);
				// Lock successful?
				if(D == depth_test) 
				{
					// Write result, order memory access so lock is freed after write
					colorBuffer[pixelID] = v.colors;
					memoryBarrier();
					atomicExchange(depthLockBuffer[pixelID], depth_int);
					// This thread has completed its update
					write = false;
				}
			}
		}
		// Back off to make sure a warp that is blocking
		// this one will eventually be scheduled. 
		memoryBarrier();
	}
}
