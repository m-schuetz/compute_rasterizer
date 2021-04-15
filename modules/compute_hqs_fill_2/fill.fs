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

bool isBackground(ivec2 pos){

	uint depth_ref = 0xffffffff;

	uint deptj_0_0 = imageLoad(uDepth, pos + ivec2(-1, -1)).r;
	uint deptj_1_0 = imageLoad(uDepth, pos + ivec2(-0, -1)).r;
	uint deptj_2_0 = imageLoad(uDepth, pos + ivec2(+1, -1)).r;

	uint deptj_0_1 = imageLoad(uDepth, pos + ivec2(-1, -0)).r;
	uint deptj_1_1 = imageLoad(uDepth, pos + ivec2(-0, -0)).r;
	uint deptj_2_1 = imageLoad(uDepth, pos + ivec2(+1, -0)).r;

	uint deptj_0_2 = imageLoad(uDepth, pos + ivec2(-1, +1)).r;
	uint deptj_1_2 = imageLoad(uDepth, pos + ivec2(-0, +1)).r;
	uint deptj_2_2 = imageLoad(uDepth, pos + ivec2(+1, +1)).r;


	bool c0 = (deptj_0_0 & deptj_0_1 & deptj_0_2) == depth_ref;
	bool c1 = (deptj_1_0 & deptj_1_1 & deptj_1_2) == depth_ref;
	bool c2 = (deptj_2_0 & deptj_2_1 & deptj_2_2) == depth_ref;

	bool r0 = (deptj_0_0 & deptj_1_0 & deptj_2_0) == depth_ref;
	bool r1 = (deptj_0_1 & deptj_1_1 & deptj_2_1) == depth_ref;
	bool r2 = (deptj_0_2 & deptj_1_2 & deptj_2_2) == depth_ref;

	if(c1 && c2){
		return true;
	}else if(r1 && r2){
		return true;
	}else if(c0 && c1){
		return true;
	}else if(r0 && r1){
		return true;
	}else if(c2 && r2 && (deptj_1_1 == depth_ref)){
		return true;
	}else if(c0 && r2 && (deptj_1_1 == depth_ref)){
		return true;
	}else if(c0 && r0 && (deptj_1_1 == depth_ref)){
		return true;
	}else if(c2 && r0 && (deptj_1_1 == depth_ref)){
		return true;
	}


	return false;
}

bool isInnerSilhouette(){

	ivec2 pos = ivec2(gl_FragCoord.xy);

	uint depth_ref = 0xffffffff;

	uint deptj_0_0 = imageLoad(uDepth, pos + ivec2(-1, -1)).r;
	uint deptj_1_0 = imageLoad(uDepth, pos + ivec2(-0, -1)).r;
	uint deptj_2_0 = imageLoad(uDepth, pos + ivec2(+1, -1)).r;

	uint deptj_0_1 = imageLoad(uDepth, pos + ivec2(-1, -0)).r;
	uint deptj_1_1 = imageLoad(uDepth, pos + ivec2(-0, -0)).r;
	uint deptj_2_1 = imageLoad(uDepth, pos + ivec2(+1, -0)).r;

	uint deptj_0_2 = imageLoad(uDepth, pos + ivec2(-1, +1)).r;
	uint deptj_1_2 = imageLoad(uDepth, pos + ivec2(-0, +1)).r;
	uint deptj_2_2 = imageLoad(uDepth, pos + ivec2(+1, +1)).r;

	uint closest = min(deptj_0_0, deptj_1_0);
	closest = min(closest, deptj_2_0);
	closest = min(closest, deptj_0_1);
	closest = min(closest, deptj_1_1);
	closest = min(closest, deptj_2_1);
	closest = min(closest, deptj_0_2);
	closest = min(closest, deptj_1_2);
	closest = min(closest, deptj_2_2);

	uint acceptable = uint(float(closest) * 1.01);

	deptj_0_0 = deptj_0_0 > acceptable ? depth_ref : deptj_0_0;
	deptj_1_0 = deptj_1_0 > acceptable ? depth_ref : deptj_1_0;
	deptj_2_0 = deptj_2_0 > acceptable ? depth_ref : deptj_2_0;

	deptj_0_1 = deptj_0_1 > acceptable ? depth_ref : deptj_0_1;
	deptj_1_1 = deptj_1_1 > acceptable ? depth_ref : deptj_1_1;
	deptj_2_1 = deptj_2_1 > acceptable ? depth_ref : deptj_2_1;

	deptj_0_2 = deptj_0_2 > acceptable ? depth_ref : deptj_0_2;
	deptj_1_2 = deptj_1_2 > acceptable ? depth_ref : deptj_1_2;
	deptj_2_2 = deptj_2_2 > acceptable ? depth_ref : deptj_2_2;

	bool c0 = (deptj_0_0 & deptj_0_1 & deptj_0_2) == depth_ref;
	bool c1 = (deptj_1_0 & deptj_1_1 & deptj_1_2) == depth_ref;
	bool c2 = (deptj_2_0 & deptj_2_1 & deptj_2_2) == depth_ref;

	bool r0 = (deptj_0_0 & deptj_1_0 & deptj_2_0) == depth_ref;
	bool r1 = (deptj_0_1 & deptj_1_1 & deptj_2_1) == depth_ref;
	bool r2 = (deptj_0_2 & deptj_1_2 & deptj_2_2) == depth_ref;

	if(c1 && c2){
		return true;
	}else if(r1 && r2){
		return true;
	}else if(c0 && c1){
		return true;
	}else if(r0 && r1){
		return true;
	}else if(c2 && r2 && (deptj_1_1 == depth_ref)){
		return true;
	}else if(c0 && r2 && (deptj_1_1 == depth_ref)){
		return true;
	}else if(c0 && r0 && (deptj_1_1 == depth_ref)){
		return true;
	}else if(c2 && r0 && (deptj_1_1 == depth_ref)){
		return true;
	}


	return false;
}

bool isOccluded(){

	ivec2 pos = ivec2(gl_FragCoord.xy);

	uint depth_ref = 0xffffffff;

	uint deptj_0_0 = imageLoad(uDepth, pos + ivec2(-1, -1)).r;
	uint deptj_1_0 = imageLoad(uDepth, pos + ivec2(-0, -1)).r;
	uint deptj_2_0 = imageLoad(uDepth, pos + ivec2(+1, -1)).r;

	uint deptj_0_1 = imageLoad(uDepth, pos + ivec2(-1, -0)).r;
	uint deptj_1_1 = imageLoad(uDepth, pos + ivec2(-0, -0)).r;
	uint deptj_2_1 = imageLoad(uDepth, pos + ivec2(+1, -0)).r;

	uint deptj_0_2 = imageLoad(uDepth, pos + ivec2(-1, +1)).r;
	uint deptj_1_2 = imageLoad(uDepth, pos + ivec2(-0, +1)).r;
	uint deptj_2_2 = imageLoad(uDepth, pos + ivec2(+1, +1)).r;

	uint closest = min(deptj_0_0, deptj_1_0);
	closest = min(closest, deptj_2_0);
	closest = min(closest, deptj_0_1);
	closest = min(closest, deptj_1_1);
	closest = min(closest, deptj_2_1);
	closest = min(closest, deptj_0_2);
	closest = min(closest, deptj_1_2);
	closest = min(closest, deptj_2_2);

	uint acceptable = uint(float(closest) * 1.01);

	return deptj_1_1 > acceptable;

	// deptj_0_0 = deptj_0_0 > acceptable ? depth_ref : deptj_0_0;
	// deptj_1_0 = deptj_1_0 > acceptable ? depth_ref : deptj_1_0;
	// deptj_2_0 = deptj_2_0 > acceptable ? depth_ref : deptj_2_0;

	// deptj_0_1 = deptj_0_1 > acceptable ? depth_ref : deptj_0_1;
	// deptj_1_1 = deptj_1_1 > acceptable ? depth_ref : deptj_1_1;
	// deptj_2_1 = deptj_2_1 > acceptable ? depth_ref : deptj_2_1;

	// deptj_0_2 = deptj_0_2 > acceptable ? depth_ref : deptj_0_2;
	// deptj_1_2 = deptj_1_2 > acceptable ? depth_ref : deptj_1_2;
	// deptj_2_2 = deptj_2_2 > acceptable ? depth_ref : deptj_2_2;

	// bool c0 = (deptj_0_0 & deptj_0_1 & deptj_0_2) == depth_ref;
	// bool c1 = (deptj_1_0 & deptj_1_1 & deptj_1_2) == depth_ref;
	// bool c2 = (deptj_2_0 & deptj_2_1 & deptj_2_2) == depth_ref;

	// bool r0 = (deptj_0_0 & deptj_1_0 & deptj_2_0) == depth_ref;
	// bool r1 = (deptj_0_1 & deptj_1_1 & deptj_2_1) == depth_ref;
	// bool r2 = (deptj_0_2 & deptj_1_2 & deptj_2_2) == depth_ref;

	// if(c1 && c2){
	// 	return true;
	// }else if(r1 && r2){
	// 	return true;
	// }else if(c0 && c1){
	// 	return true;
	// }else if(r0 && r1){
	// 	return true;
	// }else if(c2 && r2 && (deptj_1_1 == depth_ref)){
	// 	return true;
	// }else if(c0 && r2 && (deptj_1_1 == depth_ref)){
	// 	return true;
	// }else if(c0 && r0 && (deptj_1_1 == depth_ref)){
	// 	return true;
	// }else if(c2 && r0 && (deptj_1_1 == depth_ref)){
	// 	return true;
	// }


	// return false;
}

vec4 getFillColor(){
	ivec2 pos = ivec2(gl_FragCoord.xy);
	int window = 1;

	uint depth_ref = 0xffffffff;

	for(int i = -window; i <= window; i++){
	for(int j = -window; j <= window; j++){

		uint depth = imageLoad(uDepth, pos + ivec2(i, j)).r;

		depth_ref = min(depth, depth_ref);
	}}

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

	vec4 avg = sum / sum.a;

	return avg;
}

void main() {
	ivec2 pos = ivec2(gl_FragCoord.xy);

	int window = 0;
	
	uint depth_ref = 0xffffffff;

	for(int i = -window; i <= window; i++){
	for(int j = -window; j <= window; j++){

		uint depth = imageLoad(uDepth, pos + ivec2(i, j)).r;

		depth_ref = min(depth, depth_ref);
	}}

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
		// avg.rgb = vec3(
		// 	sum.a / 5.0, 
		// 	0.0, 0.0
		// );
		avg.a = 1.0;

		out_color = avg;
	}



	if(isBackground(pos)){
		out_color = vec4(1.0, 0.0, 0.0, 1.0);
	}else if(isInnerSilhouette()){
		// out_color = vec4(0.0, 1.0, 0.0, 1.0);
		
		// out_color = vec4(1.0, 0.0, 1.0, 1.0);
		out_color = texelFetch(uColor, pos, 0);
		out_color.rgb = out_color.rgb / 1.3;
		out_color.a = 1;

		out_color = getFillColor();
	}else if(isOccluded() && true){
		out_color = getFillColor();
	}else{
		uint depth = imageLoad(uDepth, pos).r;

		if(depth == 0xffffffff){
			out_color = vec4(0.0, 1.0, 0.0, 1.0);
			out_color = getFillColor();
		}
	}

}