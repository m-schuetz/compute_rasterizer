#version 450

layout(local_size_x = 16, local_size_y = 16) in;

layout(rgba8ui, binding = 0) uniform uimage2D uColor;
//layout(r32f, binding = 1) uniform image2D uDepth;
layout(rgba8ui, binding = 2) uniform uimage2D uOutput;

layout(binding = 0) uniform sampler2DMS uDepth;
//layout(binding = 0) uniform sampler2D uDepth;

layout(location = 0) uniform vec2 uScreenSize;
layout(location = 1) uniform float uNear;
layout(location = 2) uniform float uFar;

layout(location = 12) uniform float uEDLStrength;

const int numSamples = 4;

#define MSAA_ENABLED

ivec2 sampleLocations[4] = ivec2[](
	ivec2( 1,  0),
	ivec2( 0,  1),
	ivec2(-1,  0),
	ivec2( 0, -1)
);


// http://stackoverflow.com/questions/6652253/getting-the-true-z-value-from-the-depth-buffer
float linearize(float depth){
	float z_n = 2.0 * depth - 1.0;
	//float z_n = 2.0 * (1.0 - depth) - 1.0;
	//float z_n = 1 - depth;
	float z_e = 2.0 * uNear * uFar / (uFar + uNear - z_n * (uFar - uNear));

	//return z_e;

	return 1 / depth;
}

#ifdef MSAA_ENABLED

float response(ivec2 pos, int sampleNr){

	float d = texelFetch(uDepth, pos, sampleNr).r;
	float depth = log2(linearize(d));

	float sum = 0.0;
	
	for(int i = 0; i < numSamples; i++){
		ivec2 samplePos = pos + sampleLocations[i];
		float neighborDepth = texelFetch(uDepth, samplePos, sampleNr).r;
		neighborDepth = log2(linearize(neighborDepth));
		sum += max(0.0, depth - neighborDepth);
	}
	
	return sum / float(numSamples);
}

#else

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

#endif


void main(){

	uvec2 id = gl_LocalInvocationID.xy;
	id.x += gl_WorkGroupSize.x * gl_WorkGroupID.x;
	id.y += gl_WorkGroupSize.y * gl_WorkGroupID.y;

	ivec2 pixelCoords = ivec2(id);

	#ifdef MSAA_ENABLED
		uvec4 c1 = imageLoad(uColor, 2 * pixelCoords + ivec2(0, 0));
		uvec4 c2 = imageLoad(uColor, 2 * pixelCoords + ivec2(1, 0));
		uvec4 c3 = imageLoad(uColor, 2 * pixelCoords + ivec2(0, 1));
		uvec4 c4 = imageLoad(uColor, 2 * pixelCoords + ivec2(1, 1));

		uvec4 c = (c1 + c2 + c3 + c4) / 4;


		float shade = 0.0;
		for(int sampleNr = 0; sampleNr < 4; sampleNr++){
			float res = response(pixelCoords, sampleNr);
			float lShade = exp(-res * 100.0 * uEDLStrength);

			shade += clamp(lShade, 0, 1);
		}
		shade = shade / 4.0;

		float d = texelFetch(uDepth, pixelCoords, 0).r;
		d = linearize(d);

		//d = 0.1 / d;

		//c = uvec4(d * 100, 0, 0, 255);

		if(c1.a == int(0.9 * 255)){
			//c = uvec4(255, 0, 0, 255);
		}else{
			//c = uvec4(255, 255, 255, 255);
		}

		//c = uvec4(255, 255, 255, 255);

	#else
		uvec4 c = imageLoad(uColor, pixelCoords);

		float res = response(pixelCoords);
		float shade = exp(-res * 300.0 * uEDLStrength);
	#endif

	c = uvec4(vec4(c) * shade);

	// if(pixelCoords.x > 798 && pixelCoords.x < 802){
	// 	//out_color = vec4(0, 0, 0, 255);
	// 	c = uvec4(0, 0, 0, 255);
	// }

	imageStore(uOutput, pixelCoords, c);

}




