
#version 450

layout(location = 0) out vec4 out_color;

in vec3 vColor;

void main() {
	out_color = vec4(vColor, 1.0);
}

