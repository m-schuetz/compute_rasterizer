#version 450

layout(location = 0) out vec4 out_color;

in vec4 vColor;

void main() {

	out_color = vColor; 
}

