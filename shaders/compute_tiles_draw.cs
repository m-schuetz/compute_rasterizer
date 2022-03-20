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

struct Tile{

	float u;
	float v;
	float size;
	float pad;

};



layout(local_size_x = 8, local_size_y = 8) in;

layout(location = 0) uniform dmat4 uTransform;
layout(location = 1) uniform ivec2 uImageSize;
layout(r32ui, binding = 0) coherent uniform uimage2D uOutput;

layout (std430, binding = 1) buffer tiles_metadata {
	uint numTiles;
} metadata;

layout (std430, binding = 2) buffer tiles_data {
	Tile tiles[];
};

layout (std430, binding = 3) buffer depth_data {
	uint depthbuffer [];
};

layout (std430, binding = 4) buffer ss_d_rgba {
	uint64_t framebuffer[];
};

vec3 sampleP(vec2 uv){
	float x = (uv.x - 0.5) * 2.0;
	float y = (uv.y - 0.5) * 2.0;
	float d = length(vec2(x, y));

	float z = cos(8.0 * d) * (0.4 - 0.5 * d) 
		+ 0.01 * cos(350.0 * uv.x) * sin(350.0 * uv.y);

	// float x = (uv.x - 0.5) * 2.0;
	// float y = (uv.y - 0.5) * 2.0;
	// float z = 0.0;

	return vec3(x, y, z);
}

void main(){

	uint tileID = gl_WorkGroupID.x;

	if(tileID >= metadata.numTiles){
		return;
	}

	uint64_t tStart = clockARB();

	uvec2 id = gl_WorkGroupID.xy;

	Tile tile = tiles[tileID];


	// if(false)

	// for(int k = 0; k < 4; k++)
	{
		// float size = 256.0;
		// float u = float(ix) / float(gl_WorkGroupSize.x * gl_NumWorkGroups.x);
		// float v = float(iy) / float(gl_WorkGroupSize.y * gl_NumWorkGroups.y);

		uint ix = gl_LocalInvocationID.x;
		uint iy = gl_LocalInvocationID.y;

		float u = tile.u + (float(ix) * tile.size) / float(gl_WorkGroupSize.x);
		float v = tile.v + (float(iy) * tile.size) / float(gl_WorkGroupSize.y);
		
		vec2 step = tile.size / vec2(gl_WorkGroupSize.xy);
		vec2 uv = vec2(tile.u, tile.v) + vec2(ix, iy) * step;

		vec3 pos_center = sampleP(uv + 0.5 * step);
		vec3 pos_00 = sampleP(uv + vec2(0.5 - 0.33, 0.5 - 0.33) * step);
		vec3 pos_01 = sampleP(uv + vec2(0.5 - 0.33, 0.5 + 0.33) * step);
		vec3 pos_10 = sampleP(uv + vec2(0.5 + 0.33, 0.5 - 0.33) * step);
		vec3 pos_11 = sampleP(uv + vec2(0.5 + 0.33, 0.5 + 0.33) * step);

		vec4 proj_00 = mat4(uTransform) * vec4(pos_00, 1.0);
		vec4 proj_01 = mat4(uTransform) * vec4(pos_01, 1.0);
		vec4 proj_10 = mat4(uTransform) * vec4(pos_10, 1.0);
		vec4 proj_11 = mat4(uTransform) * vec4(pos_11, 1.0);

		vec4 projected = mat4(uTransform) * vec4(pos_center, 1.0);
		projected.xyz = projected.xyz / projected.w;

		if(projected.w <= 0.0 || projected.x < -1.0 || projected.x > 1.0 || projected.y < -1.0 || projected.y > 1.0){
			return;
		}

		vec2 imgPos = (projected.xy * 0.5 + 0.5) * uImageSize;
		ivec2 pixelCoords = ivec2(imgPos);
		int pixelID = pixelCoords.x + pixelCoords.y * uImageSize.x;

		uint depth = floatBitsToUint(projected.w);

		if(depth > depthbuffer[pixelID]){
			return;
		}

		// uint old = atomicMin(depthbuffer[pixelID], depth);

		// if(depth > old){
		// 	return;
		// }

		// if(gl_SubgroupInvocationID == 0)
		{
			vec3 tx = normalize(proj_00.xyz - proj_10.xyz);
			vec3 ty = normalize(proj_00.xyz - proj_01.xyz);
			vec3 N = cross(tx, ty);
			uint R = uint(255.0 * N.x);
			uint G = uint(255.0 * N.y * 2.5);
			uint B = uint(255.0 * N.z);


			// imageStore(uOutput, pixelCoords, uvec4(R, G, B, 255));

			// uint data = 256;
			// imageAtomicAdd(uOutput, pixelCoords, data);

			uint data = (R << 0) | (G << 8) | (B << 16) | (0xFF << 24);
			// imageAtomicExchange(uOutput, pixelCoords, data);
			// imageAtomicExchange(uOutput, pixelCoords + ivec2(0, 1), data);
			// imageAtomicExchange(uOutput, pixelCoords + ivec2(1, 0), data);
			// imageAtomicExchange(uOutput, pixelCoords + ivec2(1, 1), data);

			int lastPixelID = 0;
			{
				vec4 projected = mat4(uTransform) * vec4(pos_center, 1.0);
				projected.xyz = projected.xyz / projected.w;
				vec2 imgPos = (projected.xy * 0.5 + 0.5) * uImageSize;
				ivec2 pixelCoords = ivec2(imgPos);
				int pixelID = pixelCoords.x + pixelCoords.y * uImageSize.x;

				uint old = atomicMin(depthbuffer[pixelID], depth);

				if(depth < old){
					imageAtomicExchange(uOutput, pixelCoords, data);
				}

			}

			
			{
				vec4 projected = mat4(uTransform) * vec4(pos_00, 1.0);
				projected.xyz = projected.xyz / projected.w;
				vec2 imgPos = (projected.xy * 0.5 + 0.5) * uImageSize;
				ivec2 pixelCoords = ivec2(imgPos);
				int pixelID = pixelCoords.x + pixelCoords.y * uImageSize.x;

				// uint64_t point = (depth << 32) | data;
				// atomicMin(framebuffer[pixelID], point);

				// uint32_t point = depth;
				// atomicMin(framebuffer[pixelID], point);


				if(pixelID != lastPixelID){
					uint old = atomicMin(depthbuffer[pixelID], depth);

					if(depth < old){
						imageAtomicExchange(uOutput, pixelCoords, data);
					}
					lastPixelID = pixelID;
				}

			}

			
			{
				vec4 projected = mat4(uTransform) * vec4(pos_01, 1.0);
				projected.xyz = projected.xyz / projected.w;
				vec2 imgPos = (projected.xy * 0.5 + 0.5) * uImageSize;
				ivec2 pixelCoords = ivec2(imgPos);
				int pixelID = pixelCoords.x + pixelCoords.y * uImageSize.x;

				if(pixelID != lastPixelID){
					uint old = atomicMin(depthbuffer[pixelID], depth);

					if(depth < old){
						imageAtomicExchange(uOutput, pixelCoords, data);
					}
					lastPixelID = pixelID;
				}

			}

			{
				vec4 projected = mat4(uTransform) * vec4(pos_10, 1.0);
				projected.xyz = projected.xyz / projected.w;
				vec2 imgPos = (projected.xy * 0.5 + 0.5) * uImageSize;
				ivec2 pixelCoords = ivec2(imgPos);
				int pixelID = pixelCoords.x + pixelCoords.y * uImageSize.x;

				if(pixelID != lastPixelID){
					uint old = atomicMin(depthbuffer[pixelID], depth);

					if(depth < old){
						imageAtomicExchange(uOutput, pixelCoords, data);
					}
					lastPixelID = pixelID;
				}
			}

			{
				vec4 projected = mat4(uTransform) * vec4(pos_11, 1.0);
				projected.xyz = projected.xyz / projected.w;
				vec2 imgPos = (projected.xy * 0.5 + 0.5) * uImageSize;
				ivec2 pixelCoords = ivec2(imgPos);
				int pixelID = pixelCoords.x + pixelCoords.y * uImageSize.x;

				if(pixelID != lastPixelID){
					uint old = atomicMin(depthbuffer[pixelID], depth);

					if(depth < old){
						imageAtomicExchange(uOutput, pixelCoords, data);
					}
					lastPixelID = pixelID;
				}
			}

		}
	}




}