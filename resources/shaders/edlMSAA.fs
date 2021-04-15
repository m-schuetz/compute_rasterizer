#version 450

layout(location = 0) out vec4 out_color;

in vec2 vUV;

layout(binding = 0) uniform sampler2DMS uColor;
layout(binding = 1) uniform sampler2DMS uDepth;

layout(std140, binding = 4) uniform shader_data{
	mat4 transform;
	mat4 world;
	mat4 view;
	mat4 proj;

	float time;
	vec2 screenSize;
	float near;
	float far;
	float edlStrength;
	float msaaSampleCount;

} ssArgs;

const int numSamples = 4;

ivec2 sampleLocations[4] = ivec2[](
	ivec2( 1,  0),
	ivec2( 0,  1),
	ivec2(-1,  0),
	ivec2( 0, -1)
);


// http://stackoverflow.com/questions/6652253/getting-the-true-z-value-from-the-depth-buffer
float linearize(float depth){
	// that doesn't seem to be right...
	//return 1 / pow(depth, 5.5);
	return 1 / depth;
}

struct Response{
	float sum;
	float acceptedSamples;
};

Response response(ivec2 pos, int msaaSampleNr){

	float d = texelFetch(uDepth, pos, msaaSampleNr).r;
	float depth = log2(linearize(d));

	if(d == 0){
		depth = 1;
	}

	float sum = 0.0;
	float acceptedSamples = 0.0;
	
	for(int i = 0; i < numSamples; i++){
		ivec2 samplePos = pos + sampleLocations[i] * 1;
		float neighborDepth = texelFetch(uDepth, samplePos, msaaSampleNr).r;
		//neighborDepth = neighborDepth == 0 ? 1 : neighborDepth;

		if(neighborDepth > 0.0){
			neighborDepth = log2(linearize(neighborDepth));
			sum += max(0.0, depth - neighborDepth);
			acceptedSamples += 1.0;
		}
	}

	Response r;
	r.sum = sum;
	r.acceptedSamples = acceptedSamples;
	
	return r;
}

float getShade(ivec2 pos){
	
	float sumShade = 0.0;
	float count = 0.0;
	for(int msaaSampleNr = 0; msaaSampleNr < ssArgs.msaaSampleCount; msaaSampleNr++){
		Response res = response(pos, msaaSampleNr);

		float v = res.sum / res.acceptedSamples;

		float shade = exp(-v * 300.0 * ssArgs.edlStrength * 0.8);
		if(res.acceptedSamples > 0.0){
			sumShade += shade;
			count += 1.0;
		}

	}

	float avgShade = sumShade / count;

	if(count == 0){
		return 10;
	}

	return avgShade;
}

vec4 getColor(ivec2 pos){
	vec4 col = vec4(0, 0, 0, 0);

	for(int msaaSampleNr = 0; msaaSampleNr < ssArgs.msaaSampleCount; msaaSampleNr++){
		col += texelFetch(uColor, pos, msaaSampleNr);
	}

	col = col / ssArgs.msaaSampleCount;

	return col;
}

void main() {
	ivec2 pos = ivec2(gl_FragCoord.xy);

	vec4 c = getColor(pos);
	float s = getShade(pos);

	out_color = c * s;
}