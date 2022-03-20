#version 450

#extension GL_NV_shader_atomic_int64 : enable
#extension GL_NV_gpu_shader5 : enable
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_NV_shader_atomic_float : require
// #extension GL_EXT_shader_atomic_float2 : require

#define Infinity (1.0 / 0.0)

layout(local_size_x = 128, local_size_y = 1) in;

layout (std430, binding = 1) buffer framebuffer_data_0 {
	uint64_t ssFramebuffer[];
};

layout (std430, binding = 30) buffer data_debug {
	uint32_t value;
} debug;

layout(location = 0) uniform mat4 uTransform;
layout(location = 1) uniform mat4 uTransformFrustum;
layout(location = 2) uniform mat4 uWorldView;
layout(location = 3) uniform mat4 uProj;

layout(location =  9) uniform vec3 uCamPos;
layout(location = 10) uniform ivec2 uImageSize;
layout(location = 11) uniform int uPointsPerThread;


uint SPECTRAL[5] = {
	0x00ba832b,
	0x00a4ddab,
	0x00bfffff,
	0x0061aefd,
	0x001c19d7
};

#define PI 3.1415

struct Sample{
	vec3 pos;
	vec3 color;
};

Sample samplePosition(vec2 uv){

	vec3 pos;
	vec3 color;

	// { // very basic function
	// 	pos.x = 10.0 * (uv.x - 0.5);
	// 	pos.y = 10.0 * (uv.y - 0.5);
	// 	pos.z = 1.0 * cos(10.0 * uv.x) * sin(10.0 * uv.y);

	// 	color = vec3(uv.x, uv.y + pos.z * 0.5, pos.z);
	// }

	{ // quantized plane

		int bits = 5;
		// float u = int(uv.x * pow(2, bits)) / pow(2, bits);
		// float v = int(uv.y * pow(2, bits)) / pow(2, bits);

		float a = uv.x;
		float b = uv.y;

		pos.x = cos(2 * PI * a) / 1.001;
		pos.y = sin(2 * PI * a) / 1.001;
		pos.z = 0;

		pos.x = int(pos.x * pow(2, bits)) / pow(2, bits);
		pos.y = int(pos.y * pow(2, bits)) / pow(2, bits);

		color = vec3(1.0, 0.0, 0.0);
		// color = vec3(uv.x + 0.01, uv.y + pos.z * 0.5, pos.z);
	}

	
	// { // PIPE
	// 	float r = 0.5;
	// 	float h = 2.0;

	// 	pos = vec3(
	// 		r * cos(2 * PI * uv.x),
	// 		r * sin(2 * PI * uv.x),
	// 		h * uv.y - h / 2
	// 	);
	// 	color = vec3(uv.x, uv.y + pos.z * 0.5, pos.z);
	// }

	// { // MODERN ART
	// 	float r = 0.5;
	// 	float h = 8.0;

	// 	float z = h * uv.y - h / 2;
	// 	pos.x = r * cos(2 * PI * uv.x) * z * z ;
	// 	pos.y = r * sin(2 * PI * uv.x) * z *  pos.x;
	// 	pos.z = z;

	// 	color = vec3(uv.x, uv.y + pos.z * 0.5, pos.z);
	// }

	// { // SPHERE

	// 	float a = 2 * PI * uv.x;
	// 	float b = PI * uv.y;

	// 	pos.x = cos(a) *sin(b);
	// 	pos.y = sin(a) * sin(b);
	// 	pos.z = cos(b);

	// 	color = vec3(uv.x, uv.y + pos.z * 0.5, pos.z);
	// }

	// { // CORONA!

	// 	float a = 2 * PI * uv.x;
	// 	float b = PI * uv.y;
	// 	float f = 20;
	// 	float bh = 0.1;

	// 	pos.x = cos(a) *sin(b);
	// 	pos.y = sin(a) * sin(b);
	// 	pos.z = cos(b);

	// 	vec3 N = normalize(pos);

	// 	float bumpHeight = cos(f * a) * sin(f * b);

	// 	pos = pos + bh * N * bumpHeight;

	// 	color = vec3(2 * uv.x * bumpHeight, 1 - uv.y + pos.z * 0.5 * bumpHeight, bumpHeight);
	// }

	// { // bumpy cosine
	// 	float t = uv.x * 20 - 10;
	// 	float a = 0.2;
	// 	float f = 10.0;

	// 	float height = cos(t * f);

	// 	vec2 T = normalize(vec2(1.0, -sin(t)));
	// 	float phi = atan(T.y, T.x);
		
	// 	vec2 D = vec2(0.0, a * height);

	// 	pos.x = t - D.y * sin(phi);
	// 	pos.z = cos(t) + D.y * cos(phi);

	// 	// smaller depth
	// 	pos.y = uv.y * 0.01;

	// 	// bigger lines
	// 	pos.x += 0.02 * cos(2 * PI * uv.y);
	// 	pos.y += 0.02 * sin(2 * PI * uv.y);

	// 	color = vec3(255, 0, 0);
	// }

	// { // funky
	// 	float t = uv.x * 20 - 10;
	// 	float a = 0.2;
	// 	float f = 10.0;

	// 	float height = cos(t * f);

	// 	vec2 T = height * normalize(vec2(1.0, -sin(t * f)));
	// 	float phi = PI / 2;

	// 	pos.x = t + T.x * cos(phi) - T.y * sin(phi);
	// 	pos.z = cos(t) + T.x * sin(phi) + T.y * cos(phi);

	// 	// smaller depth
	// 	pos.y = uv.y * 0.01;

	// 	// bigger lines
	// 	pos.x += 0.02 * cos(2 * PI * uv.y);
	// 	pos.y += 0.02 * sin(2 * PI * uv.y);

	// 	color = vec3(0, 255, 0);
	// }



	// color.x = uv.x;
	// color.y = uv.y;
	// color.z = (pos.z / 10.0 + 1.0) / 2.0;


	Sample point;
	point.pos = pos;
	point.color = color;
	
	return point;
}

void main(){
	
	uint32_t gridSize = 100;
	float uv_spacing = 1.0 / float(128 * 100 * gridSize);

	uvec2 batchID = uvec2(
		gl_WorkGroupID.x % 100,
		gl_WorkGroupID.x / 100
	);

	uint32_t id = gl_GlobalInvocationID.x;

	for(int i = 0; i < 128; i++){

		vec2 uv = vec2(
			float(batchID.x) / float(gridSize) + float(gl_LocalInvocationID.x) / (128 * gridSize),
			float(batchID.y) / float(gridSize) + float(i) / (128 * gridSize)
		);
		Sample point = samplePosition(uv);


		// now project to screen
		vec4 pos = vec4(point.pos, 1.0);
		pos = uTransform * pos;
		pos.xyz = pos.xyz / pos.w;


		int R = int(clamp(255.0 * point.color.x, 0.0, 255.0));
		int G = int(clamp(255.0 * point.color.y, 0.0, 255.0));
		int B = int(clamp(255.0 * point.color.z, 0.0, 255.0));


		uint uColor = R | (G << 8) | (B << 16);



		bool isInsideFrustum = true;
		if(pos.w <= 0.0 || pos.x < -1.0 || pos.x > 1.0 || pos.y < -1.0 || pos.y > 1.0){
			isInsideFrustum = false;
		}

		if(isInsideFrustum){
			vec2 imgPos = (pos.xy * 0.5 + 0.5) * uImageSize;
			ivec2 pixelCoords = ivec2(imgPos);
			int pixelID = pixelCoords.x + pixelCoords.y * uImageSize.x;

			uint32_t depth = floatBitsToInt(pos.w);
			uint64_t newPoint = (uint64_t(depth) << 32UL) | uint64_t(uColor);

			uint64_t oldPoint = ssFramebuffer[pixelID];
			if(newPoint < oldPoint){
				atomicMin(ssFramebuffer[pixelID], newPoint);
			}
		}

	}
	
	


}