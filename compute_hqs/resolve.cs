
#version 450

#extension GL_ARB_gpu_shader_int64 : enable
#extension GL_NV_shader_atomic_int64 : enable

layout(local_size_x = 16, local_size_y = 16) in;

struct Vertex{
	float x;
	float y;
	float z;
	uint colors;
};

layout (std430, binding=0) buffer point_data {
	Vertex vertices[];
};

layout (std430, binding=1) buffer framebuffer_data {
	uint64_t ssFramebuffer[];
};

layout (std430, binding=2) buffer depthbuffer_data {
	uint64_t ssDepthbuffer[];
};

layout (std430, binding=3) buffer rg_data {
	uint64_t ssRG[];
};

layout (std430, binding=4) buffer va_data {
	uint64_t ssBA[];
};

layout(rgba8ui, binding = 0) uniform uimage2D uOutput;

layout(binding = 1) uniform sampler2D uGradient;


// uvec4 colorAt(int pixelID){
// 	uint64_t val64 = ssFramebuffer[pixelID];
// 	uint ucol = uint(val64 & 0x00FFFFFFUL);
// 	//uint ucol = ssFramebuffer[pixelID];

// 	if(ucol == 0){
// 		return uvec4(0, 0, 0, 255);
// 	}

// 	vec4 color = 255.0 * unpackUnorm4x8(ucol);

// 	//float w = float(ucol) * 0.0001;
// 	//float w = log2(float(ucol));

// 	//float w = pow(float(ucol), 0.6) / 500;
// 	//w = clamp(w, 0, 1);
// 	//w = -w;

// 	//float w = log2(float(ucol)) / 15;

// 	{
// 		//float w = pow(float(ucol), 0.9) / 500;
// 		float fcol = float(ucol);
// 		float w = 0.1 * log2(fcol) / log2(1.6);
// 		w = clamp(w, 0, 1);
		
		

// 		color = 255.0 * texture(uGradient, vec2(w, 0.0));
// 		color.a = 255;

// 		// if(ucol >= 60){
// 		// 	color = 255 * vec4(1.0, 0.1, 0.1, 1.0);
// 		// }else{
// 		// 	color = 255 * vec4(0.1, 0.1, 0.1, 1.0);
// 		// }
// 	}

// 	// {
// 	// 	float fcol = float(ucol);
// 	// 	float w = (log2(fcol) / log2(3)) / 3;
// 	// 	w = pow(w, 1.2) * 0.4;
// 	// 	w = clamp(w, 0, 1);
// 	// 	w = -w;

// 	// 	//color = clamp(-w * 1, 0.1, 1.0) * 255.0 * texture(uGradient, vec2(w, 0.0));
// 	// 	color = smoothstep(0.0, 0.1, clamp(-w, 0.03, 1.0)) * 255.0 * texture(uGradient, vec2(w, 0.0));
// 	// 	//color = 255.0 * texture(uGradient, vec2(w, 0.0));
// 	// }

// 	uvec4 icolor = uvec4(color);

// 	return icolor;
// }

uvec4 rgbAt(int pixelID){
	//uint64_t val64 = ssFramebuffer[pixelID];
	//uint ucol = uint(val64 & 0x00FFFFFFUL);
	//uint ucol = ssFramebuffer[pixelID];

	uint64_t rg = ssRG[pixelID];
	uint64_t ba = ssBA[pixelID];

	uint a = uint(ba & 0xFFFFFFFFUL);

	uint r = uint((rg >> 32) / a);
	uint g = uint((rg & 0xFFFFFFFFUL) / a);
	uint b = uint((ba >> 32) / a);

	//g = uint(rg & 0xFFFFFFFF) / 40;

	if(a == 0){
		return uvec4(0, 0, 0, 255);
	}

	uvec4 icolor = uvec4(r, g, b, a);

	//icolor = uvec4(a, a, a, a) * 100;
	//icolor = uvec4(b, b, b, a);

	return icolor;
}

uvec4 grayscaleAt(int pixelID){
	uint64_t val64 = ssFramebuffer[pixelID];
	uint ucol = uint(val64 & 0x00FFFFFFUL);
	uint weight = uint(val64 >> 32);

	if(ucol == 0){
		return uvec4(0, 0, 0, 255);
	}

	uint c = uint(ucol / weight);
	//c = weight / 10;

	uvec4 icolor = uvec4(c, c, c, 255);

	return icolor;
}

void main(){
	uvec2 id = gl_LocalInvocationID.xy;
	id.x += gl_WorkGroupSize.x * gl_WorkGroupID.x;
	id.y += gl_WorkGroupSize.y * gl_WorkGroupID.y;

	ivec2 imgSize = imageSize(uOutput);

	ivec2 pixelCoords = ivec2(id);
	ivec2 sourceCoords = ivec2(id);
	int pixelID = sourceCoords.x + sourceCoords.y * imgSize.x;

	
	uvec4 icolor = rgbAt(pixelID);

	//icolor.x = 100;


	//if(val64 != 0xffffffffff000000UL){
	imageStore(uOutput, pixelCoords, icolor);
	ssFramebuffer[pixelID] = 0UL;
	ssRG[pixelID] = 0UL;
	ssBA[pixelID] = 0UL;
	ssDepthbuffer[pixelID] = 0xffffffffff000000UL;
	//ssFramebuffer[pixelID] = 0UL;
	//ssFramebuffer[pixelID] = 0x00000000;
	//}

}

