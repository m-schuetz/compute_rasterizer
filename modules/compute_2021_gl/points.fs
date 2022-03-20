#version 450

layout(location = 0) out vec4 out_color;

//layout (depth_greater) out float gl_FragDepth;
layout (depth_less) out float gl_FragDepth;

in vec3 vColor;
in float vPointSize;
in float vRadius;
in float vLinearDepth;

void main() {
	out_color = vec4(vColor, 1.0);

	// out_color = vec4(1.0, 0.0, 0.0, 1.0);
}