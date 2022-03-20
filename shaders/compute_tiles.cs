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

	uint ix = gl_GlobalInvocationID.x;
	uint iy = gl_GlobalInvocationID.y;
		
	uvec2 id = gl_WorkGroupID.xy;
	uvec2 dimension = gl_WorkGroupSize.xy * gl_NumWorkGroups.xy;
	vec2 fdimension = vec2(dimension);

	{
		float size = 256.0;
		float u = float(ix) / float(dimension.x);
		float v = float(iy) / float(dimension.y);

		vec2 fx = vec2(float(ix), float(iy));

		vec2 uv_center = (fx + vec2(0.5, 0.5)) / fdimension;
		vec2 uv_00 = (fx + vec2(0.0, 0.0)) / fdimension;
		vec2 uv_01 = (fx + vec2(0.0, 1.0)) / fdimension;
		vec2 uv_10 = (fx + vec2(1.0, 0.0)) / fdimension;
		vec2 uv_11 = (fx + vec2(1.0, 1.0)) / fdimension;

		vec3 pCenter = sampleP(uv_center);

		float r_00 = distance(pCenter, sampleP(uv_00));
		float r_01 = distance(pCenter, sampleP(uv_01));
		float r_10 = distance(pCenter, sampleP(uv_10));
		float r_11 = distance(pCenter, sampleP(uv_11));

		float r = max(max(r_00, r_01), max(r_10, r_11));
		
		vec4 pos = mat4(uTransform) * vec4(pCenter, 1.0);
		pos.xyz = pos.xyz / pos.w;

		if(pos.w <= 0.0 || pos.x < -1.02 || pos.x > 1.02 || pos.y < -1.02 || pos.y > 1.02){
			return;
		}else{

			float w = r / abs(pos.w);

			if(w < 0.003){

				vec2 imgPos = (pos.xy * 0.5 + 0.5) * uImageSize;
				ivec2 pixelCoords = ivec2(imgPos);
				int pixelID = pixelCoords.x + pixelCoords.y * uImageSize.x;

				uint depth = floatBitsToUint(pos.w);

				uint old = atomicMin(depthbuffer[pixelID], depth);

				if(depth > old){
					return;
				}

				

				vec3 stx = sampleP(vec2(uv_center.x + 1.0 / float(dimension.x) * 0.1, uv_center.y + 0.0));
				vec3 sty = sampleP(vec2(uv_center.x + 1.0 / float(dimension.x) * 0.0, uv_center.y + 0.1));
				vec3 tx = normalize(stx - pCenter);
				vec3 ty = normalize(sty - pCenter);
				vec3 N = cross(tx, ty);
				uint R = uint(255.0 * N.x);
				uint G = uint(255.0 * N.y);
				uint B = uint(255.0 * N.z);

				uint data = (R << 0) | (G << 8) | (B << 16) | (0xFF << 24);
				imageAtomicExchange(uOutput, pixelCoords, data);
			
			}else if(w < 0.01){
				uint counter = atomicAdd(metadata.numTiles, 1);

				Tile tile;
				tile.u = uv_00.x;
				tile.v = uv_00.y;
				tile.size = 1.0 / float(dimension.x);

				tiles[counter] = tile;
			}else if(w < 0.02){
				uint counter = atomicAdd(metadata.numTiles, 4);

				Tile tile;
				tile.size = 0.5 / float(dimension.x);

				tile.u = uv_00.x + 0.0 * tile.size;
				tile.v = uv_00.y + 0.0 * tile.size;
				tiles[counter + 0] = tile;

				tile.u = uv_00.x + 0.0 * tile.size;
				tile.v = uv_00.y + 1.0 * tile.size;
				tiles[counter + 1] = tile;

				tile.u = uv_00.x + 1.0 * tile.size;
				tile.v = uv_00.y + 0.0 * tile.size;
				tiles[counter + 2] = tile;

				tile.u = uv_00.x + 1.0 * tile.size;
				tile.v = uv_00.y + 1.0 * tile.size;
				tiles[counter + 3] = tile;
			}else if(w < 0.04){
				uint counter = atomicAdd(metadata.numTiles, 16);

				Tile tile;
				tile.size = 0.25 / float(dimension.x);

				int i = 0;
				for(float x = 0.0; x < 4.0; x = x + 1.0){
				for(float y = 0.0; y < 4.0; y = y + 1.0){

					tile.u = uv_00.x + x * tile.size;
					tile.v = uv_00.y + y * tile.size;
					tiles[counter + i] = tile;

					i++;
				}
				}

			}else{
				uint counter = atomicAdd(metadata.numTiles, 64);

				Tile tile;
				tile.size = 0.125 / float(dimension.x);

				int i = 0;
				for(float x = 0.0; x < 8.0; x = x + 1.0){
				for(float y = 0.0; y < 8.0; y = y + 1.0){

					tile.u = uv_00.x + x * tile.size;
					tile.v = uv_00.y + y * tile.size;
					tiles[counter + i] = tile;

					i++;
				}
				}

			}
		}

	}

	// metadata.numTiles = 123;


}