
#version 450

// layout(location = 0) uniform mat4 uView;
// layout(location = 1) uniform mat4 uProj;
layout(location = 0) uniform dmat4 uViewProj;

struct Point{
	double x;
	double y;
	double z;
	uint color;
	uint padding;
};

layout(std430, binding = 0) buffer data_1 {
	Point points[];
};

out vec3 vColor;

void main() {
	Point point = points[gl_VertexID];

	dvec3 pos = dvec3(point.x, point.y, point.z);

	gl_Position = vec4(uViewProj * dvec4(pos, 1.0));
	// gl_Position = uProj * uView * vec4(pos, 1.0);

	gl_PointSize = 2.0;

	uint uColor = point.color;
	
	// vColor = vec3(
	// 	float((uColor >>  0) & 0xFF) / 255.0,
	// 	float((uColor >>  8) & 0xFF) / 255.0,
	// 	float((uColor >> 16) & 0xFF) / 255.0
	// );

	vColor = vec3(
		float((uColor >>  0) & 0xFF) / 255.0,
		float((uColor >>  8) & 0xFF) / 255.0,
		float((uColor >> 16) & 0xFF) / 255.0
	);

}