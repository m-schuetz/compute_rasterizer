#version 450

// RUNTIME GENERATED DEFINES

layout(location = 0) out vec4 out_color;
layout(location = 1) out vec4 out_indices;

in vec3 vColor;
in vec4 vVertexID;

void main() {
	//out_color = vec4(1, 0, 0, 1);
	out_color = vec4(vColor, 1.0);
	out_indices = vVertexID;
}

