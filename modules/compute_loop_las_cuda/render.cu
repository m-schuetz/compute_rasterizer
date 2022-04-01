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

#define FACTOR_9BIT 512.0
#define FACTOR_19BIT 524288.0
#define MASK_10BIT 1023
#define MASK_20BIT 1048575

static const float SCALE_19BIT = (1.0 / FACTOR_19BIT);
static const float SCALE_9BIT = (1.0 / FACTOR_9BIT);

static const float STEPS_10BIT = 1024;

static const unsigned int SPECTRAL[5] = {
	0x00ba832b,
	0x00a4ddab,
	0x00bfffff,
	0x0061aefd,
	0x001c19d7
};

extern "C" __global__
void kernel(const ChangingRenderData data,
	unsigned long long int* framebuffer,
	XYZBatch* batches,
	float* xyz12,
	uint2* xyz8,
	uint4* xyz4,
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
	if((data.uEnableFrustumCulling != 0) && (!isInsideFrustum(data.uTransform, wgMin) && !isInsideFrustum(data.uTransform, wgMax))){
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

		float w_depth = length(projCenter2D - projEdge2D);
		float d_screen = length(projCenter2D);
		float w_screen = expf(- (d_screen * d_screen));
		float w = w_depth * w_screen;

		if(w < 0.01f){
			sLevel = 4;
		}else if(w < 0.02f){
			sLevel = 3;
		}else if(w < 0.05f){
			sLevel = 2;
		}else if(w < 0.1f){
			sLevel = 1;
		}
	}
	__syncthreads();
	int level = sLevel;

	if (blockIdx.x == gridDim.x -1)
	return;

	int loopSize = data.uPointsPerThread;
	uint4 encodedl = xyz4[wgFirstPoint / 4 + threadIdx.x];

	for(int i = 0; i < loopSize/4; i++)
	{
		//unsigned int index = wgFirstPoint + i * blockDim.x + threadIdx.x;
		unsigned int index = wgFirstPoint / 4 + i * blockDim.x + threadIdx.x;
		//if(index >= data.uNumPoints/4){
		//	return;
		//}

		float3 point;
		/*if (level == 0)
		{
			point.x = xyz12[index];
			point.y = xyz12[index + data.uNumPoints];
			point.z = xyz12[index + 2*data.uNumPoints];
		}
		else if(level == 1)
		{
			uint2 ab = xyz8[index];

			unsigned int X = ab.x & MASK_20BIT;
			unsigned int Y = ab.y & MASK_20BIT;

			unsigned int Z_a = (ab.x >> 20) & MASK_10BIT;
			unsigned int Z_b = (ab.y >> 20) & MASK_10BIT;

			unsigned int Z = Z_a | (Z_b << 10);

			point.x = X * bigBoxSizeScaled.x + data.uBoxMin.x;
			point.y = Y * bigBoxSizeScaled.y + data.uBoxMin.y;
			point.z = Z * bigBoxSizeScaled.z + data.uBoxMin.z;
		}
		else
		{
			unsigned int encoded = xyz4[index];
			int X = (encoded >>  0) & MASK_10BIT;
			int Y = (encoded >> 10) & MASK_10BIT;
			int Z = (encoded >> 20) & MASK_10BIT;
			point.x = X * boxSizeScaled.x + wgMin.x;
			point.y = Y * boxSizeScaled.y + wgMin.y;
			point.z = Z * boxSizeScaled.z + wgMin.z;
		}*/

		//uint4 encodedl = xyz4[index];
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
}
