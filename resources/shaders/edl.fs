#version 450

layout(location = 0) out vec4 out_color;

in vec2 vUV;

layout(binding = 0) uniform sampler2D uColor;
layout(binding = 1) uniform sampler2D uDepth;

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
	return 1 / depth;
}


float response(ivec2 pos){

	float d = texelFetch(uDepth, pos, 0).r;
	float depth = log2(linearize(d));

	float sum = 0.0;
	
	for(int i = 0; i < numSamples; i++){
		ivec2 samplePos = pos + sampleLocations[i];
		float neighborDepth = texelFetch(uDepth, samplePos, 0).r;
		neighborDepth = log2(linearize(neighborDepth));
		sum += max(0.0, depth - neighborDepth);
	}
	
	return sum / float(numSamples);
}

void main() {
	ivec2 pos = ivec2(gl_FragCoord.xy);
	
	float res = response(pos);
	float shade = exp(-res * 300.0 * ssArgs.edlStrength * 1.8);
	//vec2 uv = out_color.xy = pos / 1600.0;

	vec4 col = texelFetch(uColor, pos, 0);

	//out_color = 0.01 * col * shade + vec4(1, 1, 1, 1) * shade;
	out_color = col * shade;
}