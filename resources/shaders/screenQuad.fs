#version 450

layout(location = 0) out vec4 out_color;

in vec2 vUV;

layout(binding = 0) uniform sampler2D uTexture;


void main() {
	vec4 texcol = texture(uTexture, vUV);

	out_color = texcol; 
}

