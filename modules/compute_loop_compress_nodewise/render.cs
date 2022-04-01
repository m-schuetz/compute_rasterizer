#version 450

#extension GL_ARB_gpu_shader_int64 : enable
#extension GL_NV_shader_atomic_int64 : enable
#extension GL_NV_gpu_shader5 : enable
// #extension GL_EXT_buffer_reference : enable


#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_KHR_shader_subgroup_vote : require
#extension GL_KHR_shader_subgroup_clustered : require

// Maybe on the 3090??
// #extension GL_EXT_shader_realtime_clock : require
#extension GL_ARB_shader_clock : require


layout(local_size_x = 128, local_size_y = 1) in;

struct Batch{
	float min_x;
	float min_y;
	float min_z;
	float max_x;
	float max_y;
	float max_z;
	uint colors;
	int numPoints;

	int pointOffset;
	int byteOffset;
	int bitsX;
	int bitsY;
	int bitsZ;
	int bits;
	int padding5;
	int padding6;
};

layout (std430, binding = 1) buffer framebuffer_data_0 {
	uint64_t ssFramebuffer_0[];
};

layout (std430, binding = 2) buffer framebuffer_data_1 {
	uint64_t ssFramebuffer_1[];
};

layout (std430, binding = 11) buffer batches_buffer_data {
	Batch ssBatchBuffer[];
};

layout (std430, binding = 20) buffer data_0 {
	uint32_t ssPointBuffer_0[];
};

layout (std430, binding = 21) buffer data_1 {
	uint32_t ssPointBuffer_1[];
};

layout (std430, binding = 22) buffer data_2 {
	uint32_t ssPointBuffer_2[];
};

layout (std430, binding = 23) buffer data_3 {
	uint32_t ssPointBuffer_3[];
};

layout (std430, binding = 24) buffer data_4 {
	uint32_t ssPointBuffer_4[];
};

layout (std430, binding = 25) buffer data_5 {
	uint32_t ssPointBuffer_5[];
};

layout (std430, binding = 26) buffer data_6 {
	uint32_t ssPointBuffer_6[];
};


layout (std430, binding = 30) buffer dbg_buffer {
	uint32_t ssDebug[];
};

layout (std430, binding = 31) buffer stats_buffer {
	uint32_t uNumRenderedPoints;
	uint32_t uNumVisibleNodes;

};

layout(location = 0) uniform mat4 uTransform;
layout(location = 1) uniform mat4 uTransformFrustum;
layout(location = 2) uniform mat4 uWorldView;
layout(location = 3) uniform mat4 uProj;


layout(location = 20) uniform mat4 uTransform_view_0;
layout(location = 21) uniform mat4 uTransform_view_1;


layout(location = 5) uniform vec3 uBoxMin;
layout(location = 6) uniform vec3 uBoxMax;
layout(location = 7) uniform int uNumPoints;
layout(location = 8) uniform int uVrEnabled;

layout(location =  9) uniform vec3 uCamPos;
layout(location = 10) uniform ivec2 uImageSize;

layout(location = 30) uniform float uLOD;
layout(location = 31) uniform int uEnableLOD;
layout(location = 32) uniform int uEnableFrustumCulling;

void writeVal_0(uint64_t val64, int pixelID){

	uint64_t old = ssFramebuffer_0[pixelID];
	if(val64 < old)
	{
		atomicMin(ssFramebuffer_0[pixelID], val64);
	}
}

void writeVal_1(uint64_t val64, int pixelID){

	uint64_t old = ssFramebuffer_1[pixelID];
	if(val64 < old)
	{
		atomicMin(ssFramebuffer_1[pixelID], val64);
	}
}

bool isInsideFrustum(vec3 point){

	vec4 pos = uTransformFrustum * vec4(point, 1.0);
	pos.xyz = pos.xyz / pos.w;

	if(pos.w <= 0.0 || pos.x < -1.4 || pos.x > 1.2 || pos.y < -1.2 || pos.y > 1.2){
		return false;
	}else{
		return true;
	}

}

void assert(bool value){

	int64_t val = 0x00000000FFFF00FFl;
	int offset = 0;

	if(value == true){
		val = 0x00000000FF00FF00l;
		offset = 1000 + 200 * uImageSize.x;
	}else{
		val = 0x00000000FF0000FFl;
		offset = 1000 + 250 * uImageSize.x;
	}

	for(int i = 0; i < 10; i++){
		for(int j = 0; j < 10; j++){
			int id = i + offset + j * uImageSize.x;
			writeVal_0(val, id);
		}
	}
}

void rasterize_left(vec4 pos, uint64_t colors, mat4 transform){
	pos = transform * pos;
	pos.xyz = pos.xyz / pos.w;

	bool isInsideFrustum = true;
	if(pos.w <= 0.0 || pos.x < -1.0 || pos.x > 1.0 || pos.y < -1.0 || pos.y > 1.0){
		isInsideFrustum = false;
	}

	if(isInsideFrustum){
		vec2 imgPos = (pos.xy * 0.5 + 0.5) * uImageSize;
		ivec2 pixelCoords = ivec2(imgPos);
		int pixelID = pixelCoords.x + pixelCoords.y * uImageSize.x;

		uint32_t depth = floatBitsToInt(pos.w);
		uint64_t val64 = (uint64_t(depth) << 32UL) | colors;
		writeVal_0(val64, pixelID);
	}
}

void rasterize_right(vec4 pos, uint64_t colors, mat4 transform){
	pos = transform * pos;
	pos.xyz = pos.xyz / pos.w;

	bool isInsideFrustum = true;
	if(pos.w <= 0.0 || pos.x < -1.0 || pos.x > 1.0 || pos.y < -1.0 || pos.y > 1.0){
		isInsideFrustum = false;
	}

	if(isInsideFrustum){
		vec2 imgPos = (pos.xy * 0.5 + 0.5) * uImageSize;
		ivec2 pixelCoords = ivec2(imgPos);
		int pixelID = pixelCoords.x + pixelCoords.y * uImageSize.x;

		uint32_t depth = floatBitsToInt(pos.w);
		uint64_t val64 = (uint64_t(depth) << 32UL) | colors;
		writeVal_1(val64, pixelID);
	}
}

// In order to address 20GB of memory with 32 bit indices,
// memory is grouped in blocks of, e.g., 20 bytes
// blockIndex references the respective block,
// and bitOffset the bit index within that block
//
// buffer: The OpenGL buffer. Stores up to 2'000'000'000 bytes. 
//         If the model is larger, additional buffers are used
// block:  Groups of 20 bytes
// word:   A 4 byte unsigned integer (smallest addressable unit)
//
#define BLOCKS_PER_BUFFER 100000000
uint readBits(uint blockIndex, uint bitOffset, uint bitSize){
	
	uint bufferIndex = blockIndex / BLOCKS_PER_BUFFER;
	uint localBlockIndex = blockIndex % BLOCKS_PER_BUFFER;
	uint wordIndex = localBlockIndex * 5 + bitOffset / 32;

	uint wordLocalOffset = bitOffset % 32;
	uint wordBitsRemaining = 32 - wordLocalOffset;
	uint valueBitsInWord = min(bitSize, wordBitsRemaining);
	uint valueBitsRemaining = bitSize - valueBitsInWord;

	uint wordValue = wordIndex;
	
	if(bufferIndex == 0){
		wordValue = ssPointBuffer_0[wordIndex];
	}else if(bufferIndex == 1){
		wordValue = ssPointBuffer_1[wordIndex];
	}

	// uint mask = uint(pow(2, valueBitsInWord)) - 1;
	uint mask = (1 << valueBitsInWord) - 1;
	uint value = (wordValue >> wordLocalOffset) & mask;

	if(valueBitsRemaining > 0){
		// uint remainderMask = uint(pow(2, valueBitsRemaining)) - 1;
		uint remainderMask = (1 << valueBitsRemaining) - 1;

		uint remainingValue = 0;
		if(bufferIndex == 0){
			remainingValue = ssPointBuffer_0[wordIndex + 1] & remainderMask;
		}else if(bufferIndex == 1){
			remainingValue = ssPointBuffer_1[wordIndex + 1] & remainderMask;
		}

		value = (remainingValue << valueBitsInWord) | value;
	}

	return value;
}

uint readWord(uint bufferIndex, uint wordIndex){

	if(bufferIndex == 0){
		return ssPointBuffer_0[wordIndex];
	}else if(bufferIndex == 1){
		return ssPointBuffer_1[wordIndex];
	}

}

// reads X,Y,Z values with bit lengths between 0 to 31 bits from an int array
//
// In order to address 20GB of memory with 32 bit indices,
// memory is grouped in blocks of, e.g., 20 bytes
// blockIndex references the respective block,
// and bitOffset the bit index within that block
//
// buffer: The OpenGL buffer. Stores up to 2'000'000'000 bytes. 
//         If the model is larger, additional buffers are used
// block:  Groups of 20 bytes
// word:   A 4 byte unsigned integer (smallest addressable unit)
//
ivec3 readXYZBits(uint blockIndex, uint bitOffset, uint bitsX, uint bitsY, uint bitsZ){

	uint bufferIndex = blockIndex / BLOCKS_PER_BUFFER;
	uint localBlockIndex = blockIndex % BLOCKS_PER_BUFFER;

	// indices of uint32 values for the bit encoded X, Y, Z coordinates
	uint wordIndex_X0 = localBlockIndex * 5 + bitOffset / 32;
	uint wordIndex_Y0 = localBlockIndex * 5 + (bitOffset + bitsX) / 32;
	uint wordIndex_Z0 = localBlockIndex * 5 + (bitOffset + bitsX + bitsY) / 32;

	// bit offset of encoded coodinate within these uint32 values
	uint word_X_localOffset = bitOffset % 32;
	uint word_Y_localOffset = (bitOffset + bitsX) % 32;
	uint word_Z_localOffset = (bitOffset + bitsX + bitsY) % 32;

	// bits may be distributed over two uint32 values
	// compute number of bits in first and in second uint value for each coordinate
	uint word_X_0_bits = min(bitsX, 32 - word_X_localOffset);
	uint word_X_1_bits = bitsX - word_X_0_bits;
	uint word_Y_0_bits = min(bitsY, 32 - word_Y_localOffset);
	uint word_Y_1_bits = bitsY - word_Y_0_bits;
	uint word_Z_0_bits = min(bitsZ, 32 - word_Z_localOffset);
	uint word_Z_1_bits = bitsZ - word_Z_0_bits;

	// compute value using non-spilt bits of each axis
	uint value_X = (readWord(bufferIndex, wordIndex_X0) >> word_X_localOffset) & ((1 << word_X_0_bits) - 1);
	uint value_Y = (readWord(bufferIndex, wordIndex_Y0) >> word_Y_localOffset) & ((1 << word_Y_0_bits) - 1);
	uint value_Z = (readWord(bufferIndex, wordIndex_Z0) >> word_Z_localOffset) & ((1 << word_Z_0_bits) - 1);

	// compute the value of the spilt bits
	uint X_1 = readWord(bufferIndex, wordIndex_X0 + 1);
	uint Y_1 = readWord(bufferIndex, wordIndex_Y0 + 1);
	uint Z_1 = readWord(bufferIndex, wordIndex_Z0 + 1);

	uint value_X1 = X_1 & ((1 << word_X_1_bits) - 1);
	uint value_Y1 = Y_1 & ((1 << word_Y_1_bits) - 1);
	uint value_Z1 = Z_1 & ((1 << word_Z_1_bits) - 1);

	// combine non-spilt and spilt bits
	value_X = (value_X1 << word_X_0_bits) | value_X;
	value_Y = (value_Y1 << word_Y_0_bits) | value_Y;
	value_Z = (value_Z1 << word_Z_0_bits) | value_Z;

	return ivec3(value_X, value_Y, value_Z);
}

void main(){

	uint pointsPerThread = 100;

	vec3 bbMin = uBoxMin;
	vec3 bbMax = uBoxMax;

	uint batchIndex = gl_WorkGroupID.x;

	Batch batch = ssBatchBuffer[batchIndex];
	uint batchSize = batch.numPoints;
	uint loopSize = pointsPerThread;
	loopSize = uint(ceil(float(batchSize) / float(gl_WorkGroupSize.x)));
	loopSize = min(loopSize, 500);
	// loopSize = 100;
	
	// uint wgFirstPoint = batch.pointOffset + gl_WorkGroupSize.x * gl_WorkGroupID.x * pointsPerThread;
	uint wgFirstPoint = batch.pointOffset;

	vec3 wgMin = vec3(batch.min_x, batch.min_y, batch.min_z) - bbMin;
	vec3 wgMax = vec3(batch.max_x, batch.max_y, batch.max_z) - bbMin;
	vec3 boxSize = wgMax - wgMin;


	// if(batchIndex < 257000 || batchIndex >= 257500)
	// if(batchIndex != 23635 )
	// if(batchIndex != 53010 )
	// {
	// 	return;
	// }


	int stepSize = 1;
	// Frustum and LOD culling
	// if(false)
	{ 
		vec3 wgMin = vec3(batch.min_x, batch.min_y, batch.min_z) - bbMin;
		vec3 wgMax = vec3(batch.max_x, batch.max_y, batch.max_z) - bbMin;

		// FRUSTUM CULLING
		if((uEnableFrustumCulling != 0) && (!isInsideFrustum(wgMin) && !isInsideFrustum(wgMax))){
			return;
		}


		// LOD CULLING
		vec3 wgCenter = (wgMin + wgMax) / 2.0;
		float wgRadius = distance(wgMin, wgMax);

		// if(wgRadius > 2.0){
		// 	return;
		// }

		vec4 viewCenter = uWorldView * vec4(wgCenter, 1.0);
		vec4 viewEdge = viewCenter + vec4(wgRadius, 0.0, 0.0, 0.0);

		vec4 projCenter = uProj * viewCenter;
		vec4 projEdge = uProj * viewEdge;

		projCenter.xy = projCenter.xy / projCenter.w;
		projEdge.xy = projEdge.xy / projEdge.w;

		float w_depth = distance(projCenter.xy, projEdge.xy);

		float d_screen = length(projCenter.xy);
		float w_screen = exp(- (d_screen * d_screen) / 1.0);

		float w = w_depth * w_screen;

		if((uEnableLOD != 0) && (w < uLOD * 0.01)){
			return;
		}

	}

	uint64_t maxval = 0xFFFFFFFF00000000UL;
	int previousPixelID = 0;
	uint64_t previousValue = maxval;


	for(int i = 0; i < loopSize; i += stepSize){

		uint index = wgFirstPoint + i * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
		uint localIndex = i * gl_WorkGroupSize.x + gl_LocalInvocationID.x;

		// if(batchIndex == 0 && i == 0 && gl_LocalInvocationID.x == 0)
		// {
		// 	ssDebug[0] = batch.numPoints;
		// 	ssDebug[0] = batch.byteOffset;
		// }


		if(index > uNumPoints){
			return;
		}

		if(localIndex > batch.numPoints){
			return;
		}

		uint64_t colors = index & 0xFFFFFFFFUL;
		// colors = 20123 * batchIndex;
		// uint64_t colors = ssPointBuffer[47111095 + index];
		// colors = batch.colors;

		float x = 0.0;
		float y = 0.0;
		float z = 0.0;

		{ 

			// uint localIndex = index % 250000000;
			// uint bufferIndex = index / 250000000;

			uint bitOffsetX = 0;
			uint bitOffsetY = batch.bitsX;
			uint bitOffsetZ = batch.bitsX + batch.bitsY;

			float factorX = pow(2.0, float(batch.bitsX));
			float factorY = pow(2.0, float(batch.bitsY));
			float factorZ = pow(2.0, float(batch.bitsZ));

			// use generalized readBits method
			uvec3 XYZ;
			// XYZ.x = readBits(batch.byteOffset, batch.bits * localIndex + bitOffsetX, batch.bitsX);
			// XYZ.y = readBits(batch.byteOffset, batch.bits * localIndex + bitOffsetY, batch.bitsY);
			// XYZ.z = readBits(batch.byteOffset, batch.bits * localIndex + bitOffsetZ, batch.bitsZ);

			// use single call to readXYZBits method
			// for now, this just computes the same as above, 
			// and checks if it gets the same result
			{
				
				uvec3 d_XYZ = readXYZBits(batch.byteOffset, batch.bits * localIndex, batch.bitsX, batch.bitsY, batch.bitsZ);

				// if(XYZ.x != d_XYZ.x){
				// 	atomicAdd(ssDebug[0], 1);
				// }
				// if(XYZ.y != d_XYZ.y){
				// 	atomicAdd(ssDebug[0], 1);
				// }
				// if(XYZ.z != d_XYZ.z){
				// 	atomicAdd(ssDebug[0], 1);
				// }

				XYZ = d_XYZ;
			}

			x = boxSize.x * (float(XYZ.x) / factorX) + batch.min_x - bbMin.x;
			y = boxSize.y * (float(XYZ.y) / factorY) + batch.min_y - bbMin.y;
			z = boxSize.z * (float(XYZ.z) / factorZ) + batch.min_z - bbMin.z;
		}

		vec4 pos = vec4(x, y, z, 1.0);

		rasterize_left(pos, colors, uTransform_view_0);

		if(uVrEnabled > 0){
			rasterize_right(pos, colors, uTransform_view_1);
		}

		barrier();
	}

	// writeVal(previousValue, previousPixelID);

	if(gl_LocalInvocationID.x == 0){
		atomicAdd(uNumRenderedPoints, uint(batch.numPoints));
		atomicAdd(uNumVisibleNodes, 1);
	}

}