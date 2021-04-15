
#version 450

layout(local_size_x = 16, local_size_y = 16) in;

layout(rgba8ui, binding = 0) uniform uimage2D uSource;
layout(rgba8ui, binding = 1) uniform uimage2D uTarget;

layout(location = 0) uniform ivec2 uSrcOffset;
layout(location = 1) uniform ivec2 uTargetOffset;
layout(location = 2) uniform ivec2 uSize;


void main(){
	uvec2 id = gl_LocalInvocationID.xy;
	id.x += gl_WorkGroupSize.x * gl_WorkGroupID.x;
	id.y += gl_WorkGroupSize.y * gl_WorkGroupID.y;

	if(id.x > uSize.x || id.y > uSize.y){
		//return;
	}

	ivec2 srcUV = uSrcOffset + ivec2(id);
	ivec2 targetUV = uTargetOffset + ivec2(id);
	//targetUV = ivec2(id);

	uvec4 color = imageLoad(uSource, srcUV);

	if(color.a == 0){
	//	return;
	}

	color = uvec4(255, 0, 0, 255);

	for(int i = 0; i < 200; i++){
		for(int j = 0; j < 200; j++){
			targetUV = ivec2(i, j) + 10 * ivec2(id);
			imageStore(uTarget, targetUV, color);
		}
	}
}
