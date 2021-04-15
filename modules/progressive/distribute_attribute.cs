#version 450

#extension GL_NV_gpu_shader5 : enable

layout(local_size_x = 32, local_size_y = 1) in;

struct Vertex{
	float ux;
	float uy;
	float uz;
	uint value;
};

layout(std430, binding = 0) buffer ssInputBuffer{
	uint inputBuffer[];
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

layout(location = 2) uniform int uNumPoints;
layout(location = 3) uniform double uPrime;
layout(location = 4) uniform int uOffset;

// see https://preshing.com/20121224/how-to-generate-a-sequence-of-unique-random-integers/
// double permute(double number, double prime){

// 	if(number > prime){
// 		return number;
// 	}

// 	double residue = mod(number * number, prime);

// 	if(number <= prime / 2){
// 		return residue;
// 	}else{
// 		return prime - residue;
// 	}
// }

// see https://preshing.com/20121224/how-to-generate-a-sequence-of-unique-random-integers/
// maps numbers in range [0, prime) to different numbers in the same range without collision
// prime must be fullfill: prime % 4 = 3
int64_t permuteI(int64_t number, int64_t prime){

	if(number > prime){
		return number;
	}

	// an int64 workaround of: residue = (number * number) % prime
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

	//targetIndex = globalInputIndex;

	int64_t primeI64 = int64_t(uPrime);
	int64_t t = permuteI(int64_t(globalInputIndex), primeI64);
	t = permuteI(t, primeI64);
	uint targetIndex = uint(t);

	uint value = inputBuffer[inputIndex];

	if(targetIndex < 134000000){
		
		Vertex v = targetBuffer0[targetIndex];
		v.value = value;

		targetBuffer0[targetIndex] = v;
	}else if(targetIndex < 2 * 134000000){
		targetIndex = targetIndex - 134000000;

		Vertex v = targetBuffer1[targetIndex];
		v.value = value;

		targetBuffer1[targetIndex] = v;
	}else if(targetIndex < 3 * 134000000){
		targetIndex = targetIndex - 2 * 134000000;

		Vertex v = targetBuffer2[targetIndex];
		v.value = value;

		targetBuffer2[targetIndex] = v;
	}else if(targetIndex < 4 * 134000000){
		targetIndex = targetIndex - 3 * 134000000;

		Vertex v = targetBuffer3[targetIndex];
		v.value = value;

		targetBuffer3[targetIndex] = v;
	}
}



