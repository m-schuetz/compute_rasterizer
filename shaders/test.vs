
#version 450

layout(location = 0) uniform mat4 uView;
layout(location = 1) uniform mat4 uProj;

out vec3 vColor;

void main() {

	int X = gl_VertexID % 100;
	int Y = gl_VertexID / 100;

	float u = float(X) / 100.0;
	float v = float(Y) / 100.0;

	vec3 pos = vec3(
		10.0 * u - 5.0,
		10.0 * v - 5.0,
		0.0
	);

	gl_Position = uProj * uView * vec4(pos, 1.0);
	// gl_Position = vec4(pos, 1.0);
	gl_PointSize = 20.0;

	vColor = vec3(u, v, 0.0);
}