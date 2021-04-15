#version 450

#extension GL_NV_gpu_shader5 : enable

layout(local_size_x = 32, local_size_y = 1) in;

struct Vertex{
	float ux;
	float uy;
	float uz;
	uint color;
};

layout(std430, binding = 0) buffer ssInputBuffer{
	Vertex inputBuffer[];
};

layout(std430, binding = 2) buffer ssTargetBuffer0{
	Vertex targetBuffer0[];
};

layout(std430, binding = 3) buffer ssTargetBuffer1{
	Vertex targetBuffer1[];
};

layout(std430, binding = 4) buffer ssTargetBuffer2{
	Vertex targetBuffer2[];
};

layout(std430, binding = 5) buffer ssTargetBuffer3{
	Vertex targetBuffer3[];
};

layout(std430, binding = 6) buffer ssTargetBuffer4{
	Vertex targetBuffer4[];
};

layout(std430, binding = 7) buffer ssTargetBuffer5{
	Vertex targetBuffer5[];
};

layout(std430, binding = 8) buffer ssTargetBuffer6{
	Vertex targetBuffer6[];
};

layout(std430, binding = 9) buffer ssTargetBuffer7{
	Vertex targetBuffer7[];
};




// layout(std430, binding = 15) buffer ssDebugBuffer{
// 	uint debugBuffer[];
// };

layout(location = 2) uniform int uNumPoints;
layout(location = 3) uniform double uPrime;
layout(location = 4) uniform int uOffset;

// see https://preshing.com/20121224/how-to-generate-a-sequence-of-unique-random-integers/
// double permute(double number, double prime){

// 	if(number > prime){
// 		return number;
// 	}

// 	//double d2 = number * number;
// 	//double residue = mod(d2, prime);

// 	int64_t d2 = int64_t(number) * int64_t(number);
// 	int64_t iprime = int64_t(prime);

// 	int residue = int(mod(d2, iprime));

// 	if(number <= prime / 2){
// 		return residue;
// 	}else{
// 		return prime - residue;
// 	}
// }

// see https://preshing.com/20121224/how-to-generate-a-sequence-of-unique-random-integers/
int64_t permuteI(int64_t number, int64_t prime){

	if(number > prime){
		return number;
	}

	int64_t q = number * number;
	int64_t d = q / prime;
	int64_t residue = q - d * prime;

	if(number <= prime / 2){
		return residue;
	}else{
		return prime - residue;
	}
}

void main(){
	
	uint inputIndex = gl_GlobalInvocationID.x;

	if(inputIndex >= uNumPoints){
		return;
	}

	uint globalInputIndex = inputIndex + uOffset;

	//double p1 = permute(double(globalInputIndex), uPrime);
	//double p2 = permute(p1, uPrime);
	//uint targetIndex = uint(p2);

	int64_t primeI64 = int64_t(uPrime);
	int64_t t = permuteI(int64_t(globalInputIndex), primeI64);
	t = permuteI(t, primeI64);
	uint targetIndex = uint(t);

	// disable shuffling
	targetIndex = globalInputIndex;

	// atomicAdd(debugBuffer[targetIndex], 1);

	//targetIndex = globalInputIndex;
	//targetIndex = 124122626 - globalInputIndex;

	Vertex v = inputBuffer[inputIndex];

	if(targetIndex < 134000000u){
		targetBuffer0[targetIndex] = v;
	}else if(targetIndex < 2u * 134000000u){
		targetIndex = targetIndex - 1u * 134000000u;
		targetBuffer1[targetIndex] = v;
	}else if(targetIndex < 3u * 134000000u){
		targetIndex = targetIndex - 2u * 134000000u;
		targetBuffer2[targetIndex] = v;
	}else if(targetIndex < 4u * 134000000u){
		targetIndex = targetIndex - 3u * 134000000u;
		targetBuffer3[targetIndex] = v;
	}else if(targetIndex < 5u * 134000000u){
		targetIndex = targetIndex - 4u * 134000000u;
		targetBuffer4[targetIndex] = v;
	}else if(targetIndex < 6u * 134000000u){
		targetIndex = targetIndex - 5u * 134000000u;
		targetBuffer5[targetIndex] = v;
	}else if(targetIndex < 7u * 134000000u){
		targetIndex = targetIndex - 6u * 134000000u;
		targetBuffer6[targetIndex] = v;
	}else if(targetIndex < 8u * 134000000u){
		targetIndex = targetIndex - 7u * 134000000u;
		targetBuffer7[targetIndex] = v;
	}
}



