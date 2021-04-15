#version 450

layout(location = 0) out vec4 out_color;

in vec2 vUV;

layout(binding = 0) uniform sampler2D uTexture;


void main() {
	vec4 texcol = texture(uTexture, vUV * vec2(1, 1));
	//texcol.xy = texcol.xy + vUV;
	//texcol.a = 1.0;

	if(texcol.a == 0){
		discard;
	}

	out_color = texcol; 


	//out_color = vec4(texcol.rgb, 1.0); 
}

