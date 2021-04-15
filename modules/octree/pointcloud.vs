#version 450

layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aColor;
layout(location = 2) in float aRandom;
layout(location = 3) in int aWhatever;

layout(location = 0) uniform int uNodeIndex;
//layout(location = 1) uniform vec3 uPivot;

//layout(location = 2) uniform mat4 uProj;
//layout(location = 3) uniform vec2 uScreenSize;

//layout(location = 1) uniform mat4 uTransform;
//layout(location = 2) uniform mat4 uWorld;
//layout(location = 3) uniform mat4 uView;
layout(location = 4) uniform mat4 uProj;

layout(location = 6) uniform vec2 uScreenSize;


layout(location = 20) uniform float uSpacing;
layout(location = 33) uniform float uScale;

layout(location = 60) uniform float uMinMilimeters;
layout(location = 61) uniform float uPointSize;

layout(location = 71) uniform float uColorMultiplier;

layout(location = 75) uniform int uSizeMode;
layout(location = 77) uniform int uFixedSize;

layout(location = 74) uniform float uUSTW;

layout(binding = 0) uniform sampler2D uGradient;

layout (std430, binding = 0) buffer point_data {
	float pointSize;
} ss;

// mat4: 16 * 4 bytes
// due to std430 alignment rules, the size of a Node will be a multiple of 16 bytes
struct Node{
	mat4 worldViewProj;
	mat4 worldView;
	mat4 world;
	vec4 offset;
	uint numPoints;
	uint level;
	uint vnStart;
};


layout (std430, binding = 1) buffer node_data {
	Node nodes[];
} ssNodeData;

layout (std430, binding = 2) buffer hierarchy_data {
	uint data[];
} ssHierarchy;

layout (std430, binding = 3) buffer octree_data {
	float size;
	int count;
	int updateEnabled;
} ssOctree;

layout (std430, binding = 4) buffer what_data {
	int whatever[];
} ssWhatever;

out vec3 vColor;

uint getLOD(Node node){
	
	vec3 offset = vec3(0.0, 0.0, 0.0);
	offset = node.offset.xyz;
	uint iOffset = node.vnStart;
	uint depth = node.level;

	//vColor = vec3(0, 0, 0);

	for(int i = 0; i <= 30; i++){

		float nodeSizeAtLevel = ssOctree.size  / pow(2.0, float(i + node.level));
		
		vec3 index3d = (aPosition - offset) / nodeSizeAtLevel;
		index3d = floor(index3d + 0.5);
		int index = int(round(4.0 * index3d.x + 2.0 * index3d.y + index3d.z));

		uint hdata = ssHierarchy.data[iOffset];
		uint mask = hdata & 0xFF;
		
		bool hasVisibleChild = (mask & (1 << index)) > 0;

		if(hasVisibleChild){
			uint advanceToFirstChild = ((hdata & 0xFF00)) | ((hdata & 0xFF0000) >> 16);

			uint advanceChild = bitCount(bitfieldExtract(mask, 0, index));
			uint advance = advanceToFirstChild + advanceChild;

			iOffset = iOffset + advance;

			depth++;
		}else{
			uint lod = hdata >> 24;

			return depth;
		}

		offset = offset + (vec3(1.0, 1.0, 1.0) * nodeSizeAtLevel * 0.5) * index3d;

	}

	return 0;
}

void main() {

	Node node = ssNodeData.nodes[uNodeIndex];


	vec4 pos = node.worldViewProj * vec4(aPosition, 1.0);

	gl_Position = pos;

	vColor = aColor;

	if(uSizeMode == 1 || true){

		// LOD
		float lod = float(getLOD(node));

		{
			float worldSpaceSize = uPointSize * (uSpacing / pow(2, lod));
			worldSpaceSize = uScale * max(0.38 * worldSpaceSize, uMinMilimeters / 1000.0);
			//float worldSpaceSize = 0.1347329020500183 / pow(2, lod);

			vec4 v1 = node.worldView * vec4(aPosition, 1.0);
			vec4 v2 = vec4(v1.x, v1.y + 2 * worldSpaceSize, v1.z, 1.0);

			vec4 vp1 = uProj * v1;
			vec4 vp2 = uProj * v2;

			vec2 vs1 = vp1.xy / vp1.w;
			vec2 vs2 = vp2.xy / vp2.w;

			float ds = distance(vs1, vs2);
			float dp = ds * uScreenSize.y;

			// desktop
			gl_PointSize = (dp / 1) * 0.2 + ssOctree.size * 0.00001;

			// VR
			//gl_PointSize = (dp / 1);

		}
	}else{
		gl_PointSize = uFixedSize;
	}


	vColor = vColor * uColorMultiplier;

	{
		float lod = float(getLOD(node));	
		lod = node.level;
		//vec4 gColor = texture(uGradient, -vec2(float(node.level) / 6.5, 0.1));
		float u = lod / 10;
		u = -pow(u, 2.0);
		vec4 gColor = texture(uGradient, vec2(u, 0));

		//vColor = gColor.xyz;
	}

	{
		float a = 0.3;
		vColor = vec3(
			pow(vColor.r, a),
			pow(vColor.g, a),
			pow(vColor.b, a)
		);

		vColor = vec3(1, 1, 1);

	}

	gl_PointSize = min(10.0, gl_PointSize);
	gl_PointSize = 1.0;
}