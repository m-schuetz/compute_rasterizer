#version 450

layout(location = 0) out vec4 out_color;

in vec3 vTexcoord;

layout(binding = 0) uniform samplerCube uTexture;


void main() {
	vec4 texcol = texture(uTexture, vTexcoord);

	out_color = vec4(texcol.rgb, 1.0); 
}

