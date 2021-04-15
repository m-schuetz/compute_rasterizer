#version 450

#extension GL_ARB_gpu_shader_int64 : enable
#extension GL_NV_shader_atomic_int64 : enable

layout(location = 0) out vec4 out_color;

in vec2 vUV;

layout(binding = 0) uniform sampler2D uColor;
// layout(binding = 1) uniform usampler2D uDepth;
layout(r32ui, binding = 1) uniform uimage2D uDepth;

layout (std430, binding=2) buffer depthbuffer_data {
	uint64_t ssDepthbuffer[];
};

// layout(std140, binding = 4) uniform shader_data{
// 	vec2 screenSize;
// } ssArgs;

void main() {
	ivec2 pos = ivec2(gl_FragCoord.xy);

	int window = 1;
	
	uint depth_ref = 0xffffffff;

	for(int i = -window; i <= window; i++){
	for(int j = -window; j <= window; j++){

		uint depth = imageLoad(uDepth, pos + ivec2(i, j)).r;

		depth_ref = min(depth, depth_ref);
	}}




	// float range = 0.0;
	// for(int i = -window; i <= window; i++){
	// for(int j = -window; j <= window; j++){

	// 	vec4 col = texelFetch(uColor, pos + ivec2(i, j), 0);
	// 	uint depth = imageLoad(uDepth, pos + ivec2(i, j)).r;

	// 	if(col.a < 1.0 / 255.0){
	// 		continue;
	// 	}

	// 	if(depth > float(depth_ref) * 1.01){
	// 		continue;
	// 	}

	// 	float factor = pow(2.0 * float(window) * float(window), 0.5);
	// 	float dist = length(vec2(ivec2(i, j)));
	// 	float w = exp(- pow(4.0 * dist / (factor / 2.0), 2.0));

	// 	range += w;
	// }}

	// if(range <= 0.1){
	// 	window = 1;
	// }

	// depth_ref = 0xffffffff;

	// for(int i = -window; i <= window; i++){
	// for(int j = -window; j <= window; j++){

	// 	uint depth = imageLoad(uDepth, pos + ivec2(i, j)).r;

	// 	depth_ref = min(depth, depth_ref);
	// }}




	vec4 sum = vec4(0.0, 0.0, 0.0, 0.0);
	for(int i = -window; i <= window; i++){
	for(int j = -window; j <= window; j++){

		vec4 col = texelFetch(uColor, pos + ivec2(i, j), 0);
		uint depth = imageLoad(uDepth, pos + ivec2(i, j)).r;

		if(col.a < 1.0 / 255.0){
			continue;
		}

		if(depth > float(depth_ref) * 1.01){
			continue;
		}

		float factor = pow(2.0 * float(window) * float(window), 0.5);
		float dist = length(vec2(ivec2(i, j)));
		float w = exp(- pow(4.0 * dist / (factor / 2.0), 2.0));

		// col.r = 3.0 * w;
		// col.g = 3.0 * w;
		// col.b = 3.0 * w;
		// col.a = 1.0;

		col = col * w;
		col.a = w;

		sum += col;
	}}

	if(sum.a == 0.0){
		out_color = vec4(0.1, 0.2, 0.3, 1.0);
	}else{

		vec4 avg = sum / sum.a;
		avg.a = 1.0;

		out_color = avg;
	}

}