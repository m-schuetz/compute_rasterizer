
#version 450

#extension GL_ARB_gpu_shader_int64 : enable
#extension GL_NV_shader_atomic_int64 : enable

// 
// Transfers the rendering results from render.cs from an SSBO into an OpenGL texture.
// 


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
	int ssFramebuffer[];
};

layout(rgba8ui, binding = 0) uniform uimage2D uOutput;

layout(binding = 1) uniform sampler2D uGradient;

uvec4 gradientAt(int pixelID){
	int ucol = ssFramebuffer[pixelID];

	if(ucol == 0){
		return uvec4(0, 0, 0, 255);
	}

	float fcol = float(ucol);
	// float w = fcol / 2000;
	float a = 10.0;
	float b = 9.0;
	float c = 2.0;
	float w = log2(fcol / a) / (b * log2(c));
	// fcol = 7 * 2^(10.0 * log2(2.1))
	// float w = pow(fcol / 1000.0, 0.4);
	w = clamp(w, 0.0, 1.0);

	// Umkehrfunktion:
	// w = 1.0   =>   fcol = a * 2^(b * log2(c))

	// float w = 0.1 * log2(fcol / 7.0) / log2(2.1);
	// fcol = a * 2^(b * log2(c))
	// fcol -> 11675.0 
	
	vec4 color = 255.0 * texture(uGradient, vec2(w, 0.0));
	color.a = 255;
	uvec4 icolor = uvec4(color);

	if(fcol > 10000.0){
		icolor = uvec4(255, 0, 255, 255);
	}

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

	uvec4 icolor = gradientAt(pixelID);

	imageStore(uOutput, pixelCoords, icolor);

	// reset to max depth and background color black
	ssFramebuffer[pixelID] = 0;
}

