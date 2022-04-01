#version 450

#extension GL_ARB_gpu_shader_int64 : enable
#extension GL_NV_shader_atomic_int64 : enable
#extension GL_NV_gpu_shader5 : enable
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_KHR_shader_subgroup_vote : require
#extension GL_KHR_shader_subgroup_clustered : require

layout(local_size_x = 16, local_size_y = 16) in;

layout(r32ui, binding = 0) coherent uniform uimage2D uFboLeft;
layout(r32ui, binding = 1) coherent uniform uimage2D uFboRight;

layout (std430, binding = 1) buffer abc_0 { uint32_t ssLeft_depth[]; };
layout (std430, binding = 2) buffer abc_1 { uint32_t ssLeft_rgba[]; };
layout (std430, binding = 3) buffer abc_2 { uint32_t ssRight_depth[]; };
layout (std430, binding = 4) buffer abc_3 { uint32_t ssRight_rgba[]; };

layout (std430, binding = 30) buffer abc_10 { 
	uint32_t value;
	bool enabled;
	uint32_t depth_numPointsProcessed;
	uint32_t depth_numNodesProcessed;
	uint32_t depth_numPointsRendered;
	uint32_t depth_numNodesRendered;
	uint32_t color_numPointsProcessed;
	uint32_t color_numNodesProcessed;
	uint32_t color_numPointsRendered;
	uint32_t color_numNodesRendered;
	uint32_t numPointsVisible;
} debug;

layout(std140, binding = 31) uniform UniformData{
	mat4 left_world;
	mat4 left_view;
	mat4 left_proj;
	mat4 left_transform;
	mat4 left_transformFrustum;
	mat4 right_world;
	mat4 right_view;
	mat4 right_proj;
	mat4 right_transform;
	mat4 right_transformFrustum;
	int pointsPerThread;
	int enableFrustumCulling;
	int showBoundingBox;
	int numPoints;
	ivec2 imageSize;
} uniforms;

float rand(float n){return fract(sin(n) * 43758.5453123);}

float rand(vec2 n) { 
	return fract(sin(dot(n, vec2(12.9898, 4.1414))) * 43758.5453);
}

float noise(float p){
	float fl = floor(p);
  float fc = fract(p);
	return mix(rand(fl), rand(fl + 1.0), fc);
}
	
float noise(vec2 n) {
	const vec2 d = vec2(0.0, 1.0);
  vec2 b = floor(n), f = smoothstep(vec2(0.0), vec2(1.0), fract(n));
	return mix(mix(rand(b), rand(b + d.yx), f.x), mix(rand(b + d.xy), rand(b + d.yy), f.x), f.y);
}

float getWeight(vec2 uv){
	float d_screen = length(uv * 2.0 - 1.0);
	float w = exp(- (d_screen * d_screen) / 0.20);

	w = w * 0.5 + 0.5 * noise(uv * 10000.0);

	return w;
}

void main(){

	uvec2 id = gl_LocalInvocationID.xy;
	id.x += gl_WorkGroupSize.x * gl_WorkGroupID.x;
	id.y += gl_WorkGroupSize.y * gl_WorkGroupID.y;

	ivec2 imgSize = uniforms.imageSize;


	// { // VR 1 pixel

	// 	ivec2 pixelCoords = ivec2(id);
	// 	ivec2 sourceCoords = ivec2(id);
	// 	int pixelID = sourceCoords.x + sourceCoords.y * imgSize.x;

	// 	{ // LEFT
	// 		uint32_t R = ssLeft_rgba[4 * pixelID + 0];
	// 		uint32_t G = ssLeft_rgba[4 * pixelID + 1];
	// 		uint32_t B = ssLeft_rgba[4 * pixelID + 2];
	// 		uint32_t count = ssLeft_rgba[4 * pixelID + 3];

	// 		uint32_t r = R / count;
	// 		uint32_t g = G / count;
	// 		uint32_t b = B / count;

	// 		if(count == 0){
	// 			r = 0;
	// 			g = 0;
	// 			b = 0;
	// 		}

	// 		uint32_t color = r | (g << 8) | (b << 16);

	// 		imageAtomicExchange(uFboLeft, pixelCoords, color);
	// 		// imageAtomicExchange(uFboRight, pixelCoords, color);
	// 	}

	// 	{ // RIGHT
	// 		uint32_t R = ssRight_rgba[4 * pixelID + 0];
	// 		uint32_t G = ssRight_rgba[4 * pixelID + 1];
	// 		uint32_t B = ssRight_rgba[4 * pixelID + 2];
	// 		uint32_t count = ssRight_rgba[4 * pixelID + 3];

	// 		uint32_t r = R / count;
	// 		uint32_t g = G / count;
	// 		uint32_t b = B / count;

	// 		if(count == 0){
	// 			r = 0;
	// 			g = 0;
	// 			b = 0;
	// 		}

	// 		uint32_t color = r | (g << 8) | (b << 16);

	// 		imageAtomicExchange(uFboRight, pixelCoords, color);
	// 	}

	// }

	{ // VR n x n pixel

		ivec2 pixelCoords = ivec2(id);
		int window = 1;


		vec2 uv = vec2(pixelCoords.xy) / vec2(imgSize.xy);
		float w = getWeight(uv);

		if(w < 0.5){
			window = 3;
		}

		{ // LEFT
			

			float R = 0;
			float G = 0;
			float B = 0;
			float count = 0;

			float depth = 1000000.0;
			for(int ox = -window; ox <= window; ox++){
			for(int oy = -window; oy <= window; oy++){

				int pixelID = (pixelCoords.x + ox) + (pixelCoords.y + oy) * imgSize.x;

				float pixelDepth = uintBitsToFloat(ssLeft_depth[pixelID]);
				if(pixelDepth >= 0.0){
					depth = min(depth, pixelDepth);
				}
			}
			}

			for(int ox = -window; ox <= window; ox++){
			for(int oy = -window; oy <= window; oy++){

				int pixelID = (pixelCoords.x + ox) + (pixelCoords.y + oy) * imgSize.x;
				float pixelDepth = uintBitsToFloat(ssLeft_depth[pixelID]);

				float w = 1.0;
				if(ox == 0 && oy == 0){
					w = 100;
				}else if(ox <= 1 && oy <= 1){
					w = 2;
				}else{
					w = 1;
				}

				if(pixelDepth > depth * 1.01){
					w = 0;
				}

				// R += float(ssLeft_rgba[4 * pixelID + 0]) * w;
				// G += float(ssLeft_rgba[4 * pixelID + 1]) * w;
				// B += float(ssLeft_rgba[4 * pixelID + 2]) * w;
				// count += float(ssLeft_rgba[4 * pixelID + 3]) * w;

				uint32_t a = ssLeft_rgba[2 * pixelID + 0];
				uint32_t b = ssLeft_rgba[2 * pixelID + 1];

				uint32_t G_l = (a >> 28) & 15;
				uint32_t G_h = (b >>  0) & 16383;

				R += float((a >> 10) & 262143) * w;
				G += float(G_l | (G_h << 4)) * w;
				B += float((b >> 14) & 262143) * w;
				count += float((a >> 0) & 1023) * w;
			}
			}

			uint32_t r = uint32_t(R / count);
			uint32_t g = uint32_t(G / count);
			uint32_t b = uint32_t(B / count);

			
			// r = uint(255 * w);
			// g = 0;
			// b = 0;

			// r = 100 * window;
			// g = 0;
			// b = 0;

			if(count == 0){
				r = 0;
				g = 0;
				b = 0;
			}

			// if(count == 0){
			// 	r = 0;
			// 	g = 0;
			// 	b = 0;
			// }else if(count < 10){
			// 	r = 0;
			// 	g = 255;
			// 	b = 0;
			// }else if(count < 109){
			// 	r = 0;
			// 	g = 100;
			// 	b = 0;
			// }else if(count < 1000){
			// 	r = 200;
			// 	g = 0;
			// 	b = 200;
			// }else{
			// 	r = 255;
			// 	g = 0;
			// 	b = 0;
			// }

			uint32_t color = r | (g << 8) | (b << 16);
			
			if(depth < 0.0){
				color = 0x0000FF00;
			}
			imageAtomicExchange(uFboLeft, pixelCoords, color);
		}

		// if(false)
		{ // RIGHT
			ivec2 pixelCoords = ivec2(id);

			float R = 0;
			float G = 0;
			float B = 0;
			float count = 0;

			float depth = 1000000.0;
			for(int ox = -window; ox <= window; ox++){
			for(int oy = -window; oy <= window; oy++){

				int pixelID = (pixelCoords.x + ox) + (pixelCoords.y + oy) * imgSize.x;

				float pixelDepth = uintBitsToFloat(ssRight_depth[pixelID]);
				if(pixelDepth >= 0.0){
					depth = min(depth, pixelDepth);
				}
			}
			}

			for(int ox = -window; ox <= window; ox++){
			for(int oy = -window; oy <= window; oy++){

				int pixelID = (pixelCoords.x + ox) + (pixelCoords.y + oy) * imgSize.x;
				float pixelDepth = uintBitsToFloat(ssRight_depth[pixelID]);

				float w = 1.0;
				if(ox == 0 && oy == 0){
					w = 100;
				}else if(ox <= 1 && oy <= 1){
					w = 2;
				}else{
					w = 1;
				}

				if(pixelDepth > depth * 1.01){
					w = 0;
				}

				uint32_t a = ssRight_rgba[2 * pixelID + 0];
				uint32_t b = ssRight_rgba[2 * pixelID + 1];

				uint32_t G_l = (a >> 28) & 15;
				uint32_t G_h = (b >>  0) & 16383;

				R += float((a >> 10) & 262143) * w;
				G += float(G_l | (G_h << 4)) * w;
				B += float((b >> 14) & 262143) * w;
				count += float((a >> 0) & 1023) * w;

				// R += float(ssRight_rgba[4 * pixelID + 0]) * w;
				// G += float(ssRight_rgba[4 * pixelID + 1]) * w;
				// B += float(ssRight_rgba[4 * pixelID + 2]) * w;
				// count += float(ssRight_rgba[4 * pixelID + 3]) * w;
			}
			}

			uint32_t r = uint32_t(R / count);
			uint32_t g = uint32_t(G / count);
			uint32_t b = uint32_t(B / count);

			if(count == 0){
				r = 0;
				g = 0;
				b = 0;
			}


			uint32_t color = r | (g << 8) | (b << 16);
			// color = uint32_t(depth);
			if(depth < 0.0){
				color = 0x0000FF00;
			}
			imageAtomicExchange(uFboRight, pixelCoords, color);
		}

	}


	// { // 1 pixel
	// 	ivec2 pixelCoords = ivec2(id);
	// 	ivec2 sourceCoords = ivec2(id);
	// 	int pixelID = sourceCoords.x + sourceCoords.y * imgSize.x;

	// 	uint32_t R = ssRGBA[4 * pixelID + 0];
	// 	uint32_t G = ssRGBA[4 * pixelID + 1];
	// 	uint32_t B = ssRGBA[4 * pixelID + 2];
	// 	uint32_t count = ssRGBA[4 * pixelID + 3];

	// 	uint32_t r = R / count;
	// 	uint32_t g = G / count;
	// 	uint32_t b = B / count;

	// 	if(count == 0){
	// 		r = 0;
	// 		g = 0;
	// 		b = 0;
	// 	}

	// 	uint32_t color = r | (g << 8) | (b << 16);

	// 	imageAtomicExchange(uOutput, pixelCoords, color);
	// }

	// { // n x n pixel
	// 	ivec2 pixelCoords = ivec2(id);

	// 	float R = 0;
	// 	float G = 0;
	// 	float B = 0;
	// 	float count = 0;

	// 	int window = 1;
	// 	float depth = 1000000.0;
	// 	for(int ox = -window; ox <= window; ox++){
	// 	for(int oy = -window; oy <= window; oy++){

	// 		int pixelID = (pixelCoords.x + ox) + (pixelCoords.y + oy) * imgSize.x;

	// 		float pixelDepth = uintBitsToFloat(ssDepth[pixelID]);
	// 		if(pixelDepth >= 0.0){
	// 			depth = min(depth, pixelDepth);
	// 		}
	// 	}
	// 	}

	// 	for(int ox = -window; ox <= window; ox++){
	// 	for(int oy = -window; oy <= window; oy++){

	// 		int pixelID = (pixelCoords.x + ox) + (pixelCoords.y + oy) * imgSize.x;
	// 		float pixelDepth = uintBitsToFloat(ssDepth[pixelID]);

	// 		float w = 1.0;
	// 		if(ox == 0 && oy == 0){
	// 			w = 100;
	// 		}else if(ox <= 1 && oy <= 1){
	// 			w = 2;
	// 		}else{
	// 			w = 1;
	// 		}

	// 		if(pixelDepth > depth * 1.01){
	// 			w = 0;
	// 		}

	// 		R += float(ssRGBA[4 * pixelID + 0]) * w;
	// 		G += float(ssRGBA[4 * pixelID + 1]) * w;
	// 		B += float(ssRGBA[4 * pixelID + 2]) * w;
	// 		count += float(ssRGBA[4 * pixelID + 3]) * w;
	// 	}
	// 	}

	// 	uint32_t r = uint32_t(R / count);
	// 	uint32_t g = uint32_t(G / count);
	// 	uint32_t b = uint32_t(B / count);

	// 	if(count == 0){
	// 		r = 0;
	// 		g = 0;
	// 		b = 0;
	// 	}


	// 	uint32_t color = r | (g << 8) | (b << 16);
	// 	// color = uint32_t(depth);
	// 	if(depth < 0.0){
	// 		color = 0x0000FF00;
	// 	}
	// 	imageAtomicExchange(uOutput, pixelCoords, color);
	// }
}