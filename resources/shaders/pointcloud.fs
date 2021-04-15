#version 450

layout(location = 0) out vec4 out_color;

layout(location = 1) uniform mat4 uTransform;
layout(location = 4) uniform mat4 uProj;

//layout (depth_greater) out float gl_FragDepth;
layout (depth_less) out float gl_FragDepth;

//in vec2 vUV;
in vec3 vColor;
in float vPointSize;
in float vRadius;
in float vLinearDepth;

void main() {
	out_color = vec4(vColor, 1.0);
}

