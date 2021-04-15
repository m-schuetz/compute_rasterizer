#version 450

layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec2 aUV;

out vec2 vUV;

void main() {
	vec4 pos = vec4(aPosition, 1.0);

	if(gl_VertexID == 0){
		gl_Position = vec4(-1, -1, 0, 1);
	}else if(gl_VertexID == 1){
		gl_Position = vec4(1, -1, 0, 1);
	}else if(gl_VertexID == 2){
		gl_Position = vec4(1, 1, 0, 1);
	}else if(gl_VertexID == 3){
		gl_Position = vec4(-1, -1, 0, 1);
	}else if(gl_VertexID == 4){
		gl_Position = vec4(1, 1, 0, 1);
	}else if(gl_VertexID == 5){
		gl_Position = vec4(-1, 1, 0, 1);
	}

	//gl_Position = pos;

	vUV = aUV;
}