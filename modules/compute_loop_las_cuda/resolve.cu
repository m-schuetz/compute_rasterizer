extern "C" 
__global__ void kernel(
	int width, int height, 
	cudaSurfaceObject_t output, 
	long long unsigned int* framebuffer, 
	unsigned int* rgba) 
{                                                             
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	

	int pixelID = x + y * width;
	unsigned int pointID = framebuffer[pixelID];
	unsigned int color = 0x00443322;
	if(pointID < 0x7FFFFFFF){
		color = rgba[pointID];
	}
	surf2Dwrite(color, output, x*4, y);
}