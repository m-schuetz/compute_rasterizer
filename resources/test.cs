#version 450

layout(local_size_x = 32, local_size_y = 32) in;

layout(rgba8, binding = 0) uniform image2D uOutput;

void main(){
	
	uvec2 id = gl_LocalInvocationID.xy;
	id.x += gl_WorkGroupSize.x * gl_WorkGroupID.x;
	id.y += gl_WorkGroupSize.y * gl_WorkGroupID.y;

	ivec2 imgSize = imageSize(uOutput);

	//if(id.x > imgSize.x){
	//	return;
	//}else if(id.y > imgSize.y){
	//	return;
	//}

	ivec2 pixelCoords = ivec2(id);

	vec4 val = vec4(vec2(id.xy) / vec2(imgSize), 0.0, 1.0);
	float distance = length(val.xy - 0.5);
	
	//val.xy = vec2(distance, distance);

	if(distance > 0.5){
		//return;
	}

	imageStore(uOutput, pixelCoords, val);
}
