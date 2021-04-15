#version 450

layout(location = 0) out vec4 out_color;



//in vec2 vUV;
in vec3 vColor;

//layout(binding = 0) uniform sampler2D uTexture;
//layout(location = 0) uniform float uR;
//layout(location = 1) uniform float uG;
//layout(location = 2) uniform float uB;
//layout(location = 123) uniform vec3 uRGB;


void main() {
	//out_color = vec4(uRGB, 1.0);
	out_color = vec4(vColor, 1.0);
	//out_color = vec4(1.0, 0.0, 0.0, 1.0);
	//out_color.xy = gl_FragCoord.xy * 0.0005;

	// if(gl_FragCoord.x > 800){
	// 	discard;
	// }
}

