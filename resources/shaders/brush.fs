#version 450

layout(location = 0) out vec4 out_color;

layout(binding = 0) uniform sampler2D uTexture;

in vec4 vColor;
in float vSize;
in float vPivotDistance;
in float vRandom;
in float vTime;

void main() {


	float dc = 1 - 2 * length(gl_PointCoord.xy - 0.5);

	if(dc < 0){
		discard;
	}

	out_color = vColor;

	//float d = vSize / 1.0;
	//d = vPivotDistance * 50;
	//vec4 texcol = texture(uTexture, d * vec2(1, -1));
	//vec4 texcol = texture(uTexture, vTime * vec2(1, -1));

	//out_color = vec4(texcol.xyz, 1);

	//out_color = vec4(0, 1, 0, 1);
	//out_color = vec4(texcol.xyz * (dc * 0.3 + 0.7), 1);
}

