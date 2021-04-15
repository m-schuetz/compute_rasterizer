#version 450

#extension GL_NV_gpu_shader5 : enable

layout(local_size_x = 32, local_size_y = 1) in;

layout(std430, set = 0, binding = 0) buffer LazData {
	uint lazdata[];
};

layout(std430, set = 0, binding = 1) buffer Positions {
	uint positions[];
};

layout(std430, set = 0, binding = 2) buffer Colors {
	uint colors[];
};

layout(std430, set = 0, binding = 3) buffer Batches {
	uint batches[];
};

layout(std430, set = 0, binding = 4) buffer Point10s {
	uint point10s[];
};

layout(std430, set = 0, binding = 5) buffer StructBuffers {
	uint struct_buffers[];
};

layout (std430, binding = 10) buffer framebuffer_data {
	uint ssTest[];
};





uniform int uNumChunks;




void main(){

	uint batch_index = gl_GlobalInvocationID.x;

	if(batch_index >= uNumChunks){
		return;
	}

	for(int i = 0; i < 100000; i++){
		int index = i % 8;
		ssTest[index] = i * i;
	}


}

