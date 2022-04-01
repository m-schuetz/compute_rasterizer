#version 450

#extension GL_ARB_gpu_shader_int64 : enable
#extension GL_NV_shader_atomic_int64 : enable
#extension GL_NV_gpu_shader5 : enable


#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_KHR_shader_subgroup_vote : require
#extension GL_KHR_shader_subgroup_clustered : require

// Maybe on the 3090??
// #extension GL_EXT_shader_realtime_clock : require
#extension GL_ARB_shader_clock : require


layout(local_size_x = 32, local_size_y = 32) in;

layout(location = 0) uniform dmat4 uTransform;
layout(location = 1) uniform ivec2 uImageSize;
// layout(rgba8ui, binding = 0) uniform uimage2D uOutput;
layout(r32ui, binding = 0) coherent uniform uimage2D uOutput;

void main(){

	uint ix = gl_GlobalInvocationID.x;
	uint iy = gl_GlobalInvocationID.y;

		
	uvec2 id = uvec2(0, 0); //gl_LocalInvocationID.xy;
	id.x += gl_WorkGroupID.x;
	id.y += gl_WorkGroupID.y;


	// for(int i = 0; i < 100; i++){
	// 	imageStore(uOutput, ivec2(i, i), uvec4(time % 255, 0, 0, 255));
	// }

	// uint64_t tStart = clockARB();
	// barrier();

	// uint y = globalID;
	// for(int x = 0; x < 100; x++){
	// 	// for(int y = 0; y < 100; y++){
	// 		uint64_t time = clockARB() - tStart;

	// 		uint64_t R = (time / 2550) % 255;
	// 		uint G = 0;

	// 		imageStore(uOutput, ivec2(x, y), uvec4(R, G, 0, 255));
	// 	// }
	// }


	// vec4 pos = mat4(uTransform) * vec4(0.0, 0.0, 0.0, 1.0);
	// pos.xyz = pos.xyz / pos.w;

	// if(pos.w <= 0.0 || pos.x < -1.0 || pos.x > 1.0 || pos.y < -1.0 || pos.y > 1.0){
	// 	return;
	// }

	// vec2 imgPos = (pos.xy * 0.5 + 0.5) * uImageSize;
	// ivec2 pixelCoords = ivec2(imgPos);
	// imageStore(uOutput, pixelCoords, uvec4(0, 255, 0, 255));


	// for(int ix = 0; ix < 128; ix++){
	// 	uint iy = globalID;
	{
		float size = 256.0;
		float u = float(ix) / float(gl_WorkGroupSize.x * gl_NumWorkGroups.x);
		float v = float(iy) / float(gl_WorkGroupSize.y * gl_NumWorkGroups.y);

		float x = (u - 0.5) * 2.0;
		float y = (v - 0.5) * 2.0;
		float z = cos(2.0 * length(vec2(x, y)));

		vec4 pos = mat4(uTransform) * vec4(x, y, z, 1.0);
		pos.xyz = pos.xyz / pos.w;

		if(pos.w <= 0.0 || pos.x < -1.0 || pos.x > 1.0 || pos.y < -1.0 || pos.y > 1.0){
			return;
		}

		vec2 imgPos = (pos.xy * 0.5 + 0.5) * uImageSize;
		ivec2 pixelCoords = ivec2(imgPos);

		{
			// uint64_t time = clockARB() - tStart;
			// uint64_t R = (time / 25) % 255;
			// uint64_t R = 200;
			uint R = (255 * id.x) / gl_NumWorkGroups.x;
			uint G = (255 * id.y) / gl_NumWorkGroups.y;
			uint B = 0;

			
			imageStore(uOutput, pixelCoords, uvec4(R, G, B, 255));
			// imageStore(uOutput, pixelCoords, uvec4(255.0 * u, 255.0 * v, 0, 255));
		}

		// uint R = uint(255.0 * u);
		// uint G = uint(255.0 * v);
		// uint B = uint(0);
		// uint data = (R << 0) | (G << 8) | (B << 16) | (0xFF << 24);
		// imageAtomicExchange(uOutput, pixelCoords, data);
	}




}