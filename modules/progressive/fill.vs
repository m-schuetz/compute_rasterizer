#version 450 core

// RUNTIME GENERATED DEFINES

layout(location = 0) in vec3 aPosition;
layout(location = 1) in int aValue;

uniform mat4 uWorldViewProj;
uniform int uOffset;

layout(binding = 0) uniform sampler2D uGradient;

uniform int uAttributeMode;

#define ATT_MODE_SCALAR 0
#define ATT_MODE_VECTOR 1


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
	
	if(uAttributeMode == ATT_MODE_VECTOR){
		vColor = getColorFromV3();
	}else if(uAttributeMode == ATT_MODE_SCALAR){
		vColor = getColorFromV1();	
	}
	//vColor = vec3(1, 0, 0);
	//vColor = vec3(1, 1, 0);

	//vColor = aValue.xyz;

	int vertexID = gl_VertexID + uOffset;
	vVertexID = vec4(
		float((vertexID >>  0) & 0xFF) / 255.0,
		float((vertexID >>  8) & 0xFF) / 255.0,
		float((vertexID >> 16) & 0xFF) / 255.0,
		float((vertexID >> 24) & 0xFF) / 255.0
	);

	if(aValue == 0 && aPosition.x == 0.0){

		// discard uninitialized points
		gl_Position = vec4(10.0, 10.0, 10.0, 1.0);
	}

	// {
	// 	//float t = float(aIndex / 100) / 500000.0;
		
	// 	//float t = float(aIndex / 100) / 1800000.0 + 0.1;
	// 	float t = float(gl_VertexID) / (5 * 1000 * 1000);
	// 	//float t = float(aIndex / 100) / 900000.0 - 0.2;

	// 	vec3 c = texture(uGradient, vec2(t, 0.0)).xyz;
	// 	vColor = c;
	// }

	// {
	// 	//float t = float(aIndex / 100) / 500000.0;
		
	// 	//float t = float(aIndex / 100) / 1800000.0 + 0.1;
	// 	float t = float(gl_VertexID / 277) / 1000000;
	// 	//float t = float(aIndex / 100) / 900000.0 - 0.2;

	// 	vec3 c = texture(uGradient, vec2(t, 0.0)).xyz;
	// 	vColor = c;
	// }

	// for progression figure in paper using retz data set
	// if(aPosition.x < 480 || aPosition.x > 620 || aPosition.y < 800 || aPosition.y > 950){
	// 	gl_Position.w = 0;
	// }

	// {
		
	// 	//float t = float(aIndex / 13) / (1000 * 1000) - 0.2;
	// 	float t = float(gl_VertexID / 1000) / (14.5 * 1000.0);

	// 	uint classes = gl_VertexID / 100000;
	// 	t = float(classes) / 140;
		

	// 	vec3 c = texture(uGradient, vec2(t, 0.0)).xyz;
	// 	vColor = c;
	// }


	//vColor = texture(uGradient, vec2(0.2, 0.0)).xyz;
}


