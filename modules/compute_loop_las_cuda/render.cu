#define USE_PREFETCH 0

#include "kernel_data.h"
#include "helper_math.h"

__device__ float4 matMul(const Mat& m, const float4& v)
{
	return make_float4(dot(m.rows[0], v), dot(m.rows[1], v), dot(m.rows[2], v), dot(m.rows[3], v));
}

__device__ bool isInsideFrustum(const Mat& m, float3 point)
{
	float4 pos = matMul(m, make_float4(point, 1.0f));
	if(pos.w <= 0.0f || pos.x < -1.4f*pos.w || pos.x > 1.2f*pos.w || pos.y < -1.2f*pos.w || pos.y > 1.2f*pos.w){
		return false;
	}else{
		return true;
	}
}


struct Plane{
	float3 normal;
	float constant;
};

__device__ float access(float4 v, int i)
{
	if (i == 0)
		return v.x;
	if (i == 1)
		return v.y;
	if (i == 2)
		return v.z;
	return v.w;
}

__device__ float t(const ChangingRenderData& data, int index)
{
	int a = index % 4;
	int b = index / 4;
	return access(data.uTransform.rows[a], b);
}

__device__ float distanceToPoint(float3 point, Plane plane){
	return (plane.normal.x * point.x + plane.normal.y*point.y + plane.normal.z*point.z) + plane.constant;
}

__device__ Plane createPlane(float x, float y, float z, float w){

	float nLength = sqrt(x*x + y*y + z*z);
	Plane plane;
	plane.normal = make_float3(x, y, z) / nLength;
	plane.constant = w / nLength;
	return plane;
}

__device__ bool intersectsFrustum(const ChangingRenderData& d, float3 wgMin, float3 wgMax){

	Plane planes[6] = {
		createPlane(t( d,3) - t(d,0), t( d,7) - t(d,4), t(d,11) - t( d,8), t(d,15) - t(d,12)),
		createPlane(t( d,3) + t(d,0), t( d,7) + t(d,4), t(d,11) + t( d,8), t(d,15) + t(d,12)),
		createPlane(t( d,3) + t(d,1), t( d,7) + t(d,5), t(d,11) + t( d,9), t(d,15) + t(d,13)),
		createPlane(t( d,3) - t(d,1), t( d,7) - t(d,5), t(d,11) - t( d,9), t(d,15) - t(d,13)),
		createPlane(t( d,3) - t(d,2), t( d,7) - t(d,6), t(d,11) - t(d,10), t(d,15) - t(d,14)),
		createPlane(t( d,3) + t(d,2), t( d,7) + t(d,6), t(d,11) + t(d,10), t(d,15) + t(d,14)),
	};
	for(int i = 0; i < 6; i++){

		Plane plane = planes[i];

		float3 vector;
		vector.x = plane.normal.x > 0.0 ? wgMax.x : wgMin.x;
		vector.y = plane.normal.y > 0.0 ? wgMax.y : wgMin.y;
		vector.z = plane.normal.z > 0.0 ? wgMax.z : wgMin.z;

		float d = distanceToPoint(vector, plane);

		if(d < 0){
			return false;
		}
	}

	return true;
}

#define FACTOR_9BIT 512.0
#define FACTOR_19BIT 524288.0

static const float SCALE_19BIT = (1.0 / FACTOR_19BIT);
static const float SCALE_9BIT = (1.0 / FACTOR_9BIT);

#define STEPS_30BIT 1073741824
#define MASK_30BIT 1073741823
#define STEPS_20BIT 1048576
#define MASK_20BIT 1048575
#define STEPS_10BIT 1024
#define MASK_10BIT 1023

static const unsigned int SPECTRAL[5] = {
	0x00ba832b,
	0x00a4ddab,
	0x00bfffff,
	0x0061aefd,
	0x001c19d7
};

__device__ void rasterize(const ChangingRenderData& data, unsigned long long int* framebuffer, float3 point, unsigned int index)
{
	float4 pos = matMul(data.uTransform, make_float4(point, 1.0f));

	pos.x = pos.x / pos.w;
	pos.y = pos.y / pos.w;

	float2 imgPos = {(pos.x * 0.5f + 0.5f) * data.uImageSize.x, (pos.y * 0.5f + 0.5f) * data.uImageSize.y};
	int2 pixelCoords = make_int2(imgPos.x, imgPos.y);
	int pixelID = pixelCoords.x + pixelCoords.y * data.uImageSize.x;

	unsigned int depth = *((int*)&pos.w);
	unsigned long long int newPoint = (((unsigned long long int)depth) << 32) | index;

	if(!(pos.w <= 0.0 || pos.x < -1 || pos.x > 1 || pos.y < -1|| pos.y > 1)){
		unsigned long long int oldPoint = framebuffer[pixelID];
		if(newPoint < oldPoint){
			atomicMin(&framebuffer[pixelID], newPoint);
		}
	}
}

extern "C" __global__
void kernel(const ChangingRenderData data,
	unsigned long long int* framebuffer,
	XYZBatch* batches,
	unsigned int* ssXyz_12b,
	unsigned int* ssXyz_8b,
	unsigned int* ssXyz_4b,
	unsigned int* rgba)
{
	unsigned int batchIndex = blockIdx.x;
	unsigned int numPointsPerBatch = data.uPointsPerThread * blockDim.x;
	unsigned int wgFirstPoint = batchIndex * numPointsPerBatch;

	XYZBatch batch = batches[batchIndex];
	float3 wgMin = make_float3(batch.min_x, batch.min_y, batch.min_z);
	float3 wgMax = make_float3(batch.max_x, batch.max_y, batch.max_z);
	float3 boxSize = wgMax - wgMin;
	float3 boxSizeScaled = boxSize / STEPS_10BIT;
	float3 bigBoxSize = data.uBoxMax - data.uBoxMin;
	float3 bigBoxSizeScaled = bigBoxSize * SCALE_19BIT;


	// FRUSTUM CULLING
	if((data.uEnableFrustumCulling != 0) && !intersectsFrustum(data,wgMin, wgMax)){
		return;
	}

	float3 wgCenter = 0.5f * (wgMin + wgMax);

	__shared__ int sLevel;
	if(threadIdx.x == 0)
	{
		sLevel = 0;
		// LOD CULLING

		float wgRadius = length(wgMin - wgMax);

		float4 viewCenter = matMul(data.uWorldView, make_float4(wgCenter, 1.0f));
		float4 viewEdge = viewCenter + make_float4(wgRadius, 0.0f, 0.0f, 0.0f);
		float4 projCenter = matMul(data.uProj, viewCenter);
		float4 projEdge = matMul(data.uProj, viewEdge);

		float2 projCenter2D = make_float2(projCenter.x, projCenter.y);
		float2 projEdge2D = make_float2(projEdge.x, projEdge.y);
		projCenter2D /= projCenter.w;
		projEdge2D /= projEdge.w;
		
		float2 screenCenter = 0.5f * (projCenter2D + 1.0f);
		screenCenter = make_float2(data.uImageSize.x * screenCenter.x, data.uImageSize.y * screenCenter.y);
		float2 screenEdge = 0.5f * (projEdge2D + 1.0f);
		screenEdge = make_float2(data.uImageSize.x * screenEdge.x, data.uImageSize.y * screenEdge.y);
		float2 diff = screenEdge - screenCenter;
		float pixelSize = sqrt(diff.x*diff.x + diff.y*diff.y);

		if(pixelSize < 100){
			sLevel = 4;
		}else if(pixelSize < 200){
			sLevel = 3;
		}else if(pixelSize < 500){
			sLevel = 2;
		}else if(pixelSize < 10000){
			sLevel = 1;
		}else{
			sLevel = 0;
		}
	}
	__syncthreads();
	int level = sLevel;

	if (blockIdx.x == gridDim.x -1)
	return;

	int loopSize = data.uPointsPerThread;	
		
#if USE_PREFETCH

	unsigned int base = wgFirstPoint / 4 + threadIdx.x;
	if(level == 0)
	{
		uint4 prefetch_4b = ((uint4*)ssXyz_4b)[base];
		uint4 prefetch_8b = ((uint4*)ssXyz_8b)[base];
		uint4 prefetch_12b = ((uint4*)ssXyz_12b)[base];

		for(int i = 0; i < loopSize / 4; i++){
			unsigned int encodedw_4b[4] = {prefetch_4b.x, prefetch_4b.y, prefetch_4b.z, prefetch_4b.w};
			unsigned int encodedw_8b[4] = {prefetch_8b.x, prefetch_8b.y, prefetch_8b.z, prefetch_8b.w};
			unsigned int encodedw_12b[4] = {prefetch_12b.x, prefetch_12b.y, prefetch_12b.z, prefetch_12b.w};

			if (i < (loopSize / 4 - 1)){
				prefetch_4b = ((uint4*)ssXyz_4b)[base + (i+1)*blockDim.x];
				prefetch_8b = ((uint4*)ssXyz_8b)[base + (i+1)*blockDim.x];
				prefetch_12b = ((uint4*)ssXyz_12b)[base + (i+1)*blockDim.x];
			}

			for (int j = 0; j < 4; j++){
				uint index = 4 * (base + i * blockDim.x) + j; 

				uint b4 = encodedw_4b[j];
				uint b8 = encodedw_8b[j];
				uint b12 = encodedw_12b[j];

				uint X_4 = (b4 >>  0) & MASK_10BIT;
				uint Y_4 = (b4 >> 10) & MASK_10BIT;
				uint Z_4 = (b4 >> 20) & MASK_10BIT;

				uint X_8 = (b8 >>  0) & MASK_10BIT;
				uint Y_8 = (b8 >> 10) & MASK_10BIT;
				uint Z_8 = (b8 >> 20) & MASK_10BIT;

				uint X_12 = (b12 >>  0) & MASK_10BIT;
				uint Y_12 = (b12 >> 10) & MASK_10BIT;
				uint Z_12 = (b12 >> 20) & MASK_10BIT;

				uint X = (X_4 << 20) | (X_8 << 10) | X_12;
				uint Y = (Y_4 << 20) | (Y_8 << 10) | Y_12;
				uint Z = (Z_4 << 20) | (Z_8 << 10) | Z_12;

				float3 point;
				point.x = X * (boxSize.x / STEPS_30BIT) + wgMin.x;
				point.y = Y * (boxSize.y / STEPS_30BIT) + wgMin.y;
				point.z = Z * (boxSize.z / STEPS_30BIT) + wgMin.z;

				// now rasterize to screen
				rasterize(data, framebuffer, point, index);
			}
		}
	}
	else if(level == 1){
		uint4 prefetch_4b = ((uint4*)ssXyz_4b)[base];
		uint4 prefetch_8b = ((uint4*)ssXyz_8b)[base];

		for(int i = 0; i < loopSize / 4; i++){
			unsigned int encodedw_4b[4] = {prefetch_4b.x, prefetch_4b.y, prefetch_4b.z, prefetch_4b.w};
			unsigned int encodedw_8b[4] = {prefetch_8b.x, prefetch_8b.y, prefetch_8b.z, prefetch_8b.w};

			if (i < (loopSize/4 - 1)){
				prefetch_4b = ((uint4*)ssXyz_4b)[base + (i+1)*blockDim.x];
				prefetch_8b = ((uint4*)ssXyz_8b)[base + (i+1)*blockDim.x];
			}

			for (int j = 0; j < 4; j++){
				uint index = 4 * (base + i * blockDim.x) + j; 

				uint b4 = encodedw_4b[j];
				uint b8 = encodedw_8b[j];

				uint X_4 = (b4 >>  0) & MASK_10BIT;
				uint Y_4 = (b4 >> 10) & MASK_10BIT;
				uint Z_4 = (b4 >> 20) & MASK_10BIT;

				uint X_8 = (b8 >>  0) & MASK_10BIT;
				uint Y_8 = (b8 >> 10) & MASK_10BIT;
				uint Z_8 = (b8 >> 20) & MASK_10BIT;

				uint X = (X_4 << 20) | (X_8 << 10);
				uint Y = (Y_4 << 20) | (Y_8 << 10);
				uint Z = (Z_4 << 20) | (Z_8 << 10);

				float3 point;
				point.x = X * (boxSize.x / STEPS_30BIT) + wgMin.x;
				point.y = Y * (boxSize.y / STEPS_30BIT) + wgMin.y;
				point.z = Z * (boxSize.z / STEPS_30BIT) + wgMin.z;

				// now rasterize to screen
				rasterize(data, framebuffer, point, index);
			}
		}
	}
	else
	{
		uint4 prefetch = ((uint4*)ssXyz_4b)[base];
		
		for(int i = 0; i < loopSize / 4; i++){
			uint encodedw[4] = {prefetch.x, prefetch.y, prefetch.z, prefetch.w};
			
			if (i < (loopSize / 4 - 1)){
				prefetch = ((uint4*)ssXyz_4b)[base + (i+1)*blockDim.x];
			}

			for (int j = 0; j < 4; j++){
				uint index = 4 * (base + i * blockDim.x) + j; 

				uint encoded = encodedw[j];
				
				uint X = (encoded >>  0) & MASK_10BIT;
				uint Y = (encoded >> 10) & MASK_10BIT;
				uint Z = (encoded >> 20) & MASK_10BIT;

				float3 point;
				point.x = X * (boxSize.x / STEPS_10BIT) + wgMin.x;
				point.y = Y * (boxSize.y / STEPS_10BIT) + wgMin.y;
				point.z = Z * (boxSize.z / STEPS_10BIT) + wgMin.z;
			
				// now rasterize to screen
				rasterize(data, framebuffer, point, index);
			}
		}
	}
#else
	for(int i = 0; i < loopSize; i++)
	{
		float3 point;
		unsigned int index = wgFirstPoint + i * blockDim.x + threadIdx.x;	
		
		if(level == 0){
			unsigned int b4 = ssXyz_4b[index];
			unsigned int b8 = ssXyz_8b[index];
			unsigned int b12 = ssXyz_12b[index];

			unsigned int X_4 = (b4 >>  0) & MASK_10BIT;
			unsigned int Y_4 = (b4 >> 10) & MASK_10BIT;
			unsigned int Z_4 = (b4 >> 20) & MASK_10BIT;

			unsigned int X_8 = (b8 >>  0) & MASK_10BIT;
			unsigned int Y_8 = (b8 >> 10) & MASK_10BIT;
			unsigned int Z_8 = (b8 >> 20) & MASK_10BIT;

			unsigned int X_12 = (b12 >>  0) & MASK_10BIT;
			unsigned int Y_12 = (b12 >> 10) & MASK_10BIT;
			unsigned int Z_12 = (b12 >> 20) & MASK_10BIT;

			unsigned int X = (X_4 << 20) | (X_8 << 10) | X_12;
			unsigned int Y = (Y_4 << 20) | (Y_8 << 10) | Y_12;
			unsigned int Z = (Z_4 << 20) | (Z_8 << 10) | Z_12;

			float x = X * (boxSize.x / STEPS_30BIT) + wgMin.x;
			float y = Y * (boxSize.y / STEPS_30BIT) + wgMin.y;
			float z = Z * (boxSize.z / STEPS_30BIT) + wgMin.z;

			point = make_float3(x, y, z);
		}else if(level == 1){ 
			unsigned int  b4 = ssXyz_4b[index];
			unsigned int  b8 = ssXyz_8b[index];

			unsigned int  X_4 = (b4 >>  0) & MASK_10BIT;
			unsigned int  Y_4 = (b4 >> 10) & MASK_10BIT;
			unsigned int  Z_4 = (b4 >> 20) & MASK_10BIT;

			unsigned int  X_8 = (b8 >>  0) & MASK_10BIT;
			unsigned int  Y_8 = (b8 >> 10) & MASK_10BIT;
			unsigned int  Z_8 = (b8 >> 20) & MASK_10BIT;

			unsigned int  X = (X_4 << 20) | (X_8 << 10);
			unsigned int  Y = (Y_4 << 20) | (Y_8 << 10);
			unsigned int  Z = (Z_4 << 20) | (Z_8 << 10);

			float x = X * (boxSize.x / STEPS_30BIT) + wgMin.x;
			float y = Y * (boxSize.y / STEPS_30BIT) + wgMin.y;
			float z = Z * (boxSize.z / STEPS_30BIT) + wgMin.z;

			point = make_float3(x, y, z);
		}else{ 
			unsigned int encoded = ssXyz_4b[index];

			unsigned int X = (encoded >>  0) & MASK_10BIT;
			unsigned int Y = (encoded >> 10) & MASK_10BIT;
			unsigned int Z = (encoded >> 20) & MASK_10BIT;

			float x = X * (boxSize.x / STEPS_10BIT) + wgMin.x;
			float y = Y * (boxSize.y / STEPS_10BIT) + wgMin.y;
			float z = Z * (boxSize.z / STEPS_10BIT) + wgMin.z;

			point = make_float3(x, y, z);
		}
		rasterize(data, framebuffer, point, index);
	}
#endif

/*
	for(int i = 0; i < loopSize/4; i++)
	{
		unsigned int index = wgFirstPoint / 4 + i * blockDim.x + threadIdx.x;

		float3 point;

		uint encodedw[4] = {encodedl.x, encodedl.y, encodedl.z, encodedl.w};
		
		if (i < (loopSize/4 - 1))
			encodedl = xyz4[wgFirstPoint / 4 + threadIdx.x + (i+1)*blockDim.x];		
		
		for (int j = 0; j < 4; j++)
		{
			uint encoded = encodedw[j];
			int X = (encoded >>  0) & MASK_10BIT;
			int Y = (encoded >> 10) & MASK_10BIT;
			int Z = (encoded >> 20) & MASK_10BIT;
			point.x = X * boxSizeScaled.x + wgMin.x;
			point.y = Y * boxSizeScaled.y + wgMin.y;
			point.z = Z * boxSizeScaled.z + wgMin.z;

			float4 pos = matMul(data.uTransform, make_float4(point, 1.0f));


			pos.x = pos.x / pos.w;
			pos.y = pos.y / pos.w;

			float2 imgPos = {(pos.x * 0.5f + 0.5f) * data.uImageSize.x, (pos.y * 0.5f + 0.5f) * data.uImageSize.y};
			int2 pixelCoords = make_int2(imgPos.x, imgPos.y);
			int pixelID = pixelCoords.x + pixelCoords.y * data.uImageSize.x;

			unsigned int depth = *((int*)&pos.w);
			unsigned long long int newPoint = (((unsigned long long int)depth) << 32) | (index * 4 + j);

			if(!(pos.w <= 0.0 || pos.x < -1 || pos.x > 1 || pos.y < -1|| pos.y > 1)){
				unsigned long long int oldPoint = framebuffer[pixelID];
				if(newPoint < oldPoint){
					atomicMin(&framebuffer[pixelID], newPoint);
				}
			}
		}
	}
	*/
}
