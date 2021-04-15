#version 450 core

// RUNTIME GENERATED DEFINES

layout(location = 0) in vec3 aPosition;
layout(location = 1) in int aValue;
layout(location = 2) in int aIndex;

uniform mat4 uWorldViewProj;

uniform int uAttributeMode;
uniform float uPointSize;

#define ATT_MODE_SCALAR 0
#define ATT_MODE_VECTOR 1

layout(binding = 0) uniform sampler2D uGradient;

out vec3 vColor;
out vec4 vVertexID;



vec3 getColorFromV1(){
	float w = intBitsToFloat(aValue);
	w = clamp(w, 0, 1);
	vec3 v = texture(uGradient, vec2(w, 0.0)).rgb;

	return v;
}

vec3 getColorFromV3(){
	vec3 v = vec3(
		(aValue >>   0) & 0xFF,
		(aValue >>   8) & 0xFF,
		(aValue >>  16) & 0xFF
	);

	v = v / 255.0;

	return v;
}

void main() {
	
	gl_Position = uWorldViewProj * vec4(aPosition, 1.0);
	gl_PointSize = uPointSize;

	if(uAttributeMode == ATT_MODE_VECTOR){
		vColor = getColorFromV3();
	}else if(uAttributeMode == ATT_MODE_SCALAR){
		vColor = getColorFromV1();	
	}

	//float gamma = 0.6;
	//vColor = pow(vColor, vec3(gamma));

	//vColor = vec3(1, 1, 1);

	//float a = 0.3;
	//vColor = vec3(
	//	pow(vColor.r, a),
	//	pow(vColor.g, a),
	//	pow(vColor.b, a)
	//) + 1.0;

	
	uint index = uint(aIndex);
	vVertexID = vec4(
		float((index >>  0) & 0xFF) / 255.0,
		float((index >>  8) & 0xFF) / 255.0,
		float((index >> 16) & 0xFF) / 255.0,
		float((index >> 24) & 0xFF) / 255.0
	);

}


