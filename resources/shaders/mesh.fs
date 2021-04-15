#version 450

layout(location = 0) out vec4 out_color;

in vec2 vUV;
in vec3 vNormal;

layout(binding = 0) uniform sampler2D uTexture;


void main() {
	//vec4 texcol = textureLod(uTexture, vUV * vec2(1, -1), 2.0);
	vec4 texcol = texture(uTexture, vUV * vec2(1, -1));
	
	out_color = vec4(texcol.rgb, 0.9);

	if(texcol.a == 0){
		discard;
	}

	//out_color = vec4(vUV, 0, 1);

	//out_color = vec4(1, 0.1, 0.1, 1);

	//out_color = vec4(vNormal, 1.0);
	//out_color = vec4(vUV, 0.0, 1.0);
	//out_color = vec4(texcol.rgb, 1.0);
}

