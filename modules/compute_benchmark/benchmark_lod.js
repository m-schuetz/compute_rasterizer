
function parsePoints(metadata, node, data){
	let buffer = data.slice(node.byteOffset, node.byteOffset + node.byteSize);
	let numPoints = node.numPoints;

	let vboData = new ArrayBuffer(16 * numPoints);
	let source = new DataView(buffer);
	let target = new DataView(vboData);

	let {scale, offset} = metadata;

	for(let i = 0; i < numPoints; i++){

		let X = source.getInt32(18 * i + 0, true);
		let Y = source.getInt32(18 * i + 4, true);
		let Z = source.getInt32(18 * i + 8, true);
		let R = source.getUint16(18 * i + 12, true);
		let G = source.getUint16(18 * i + 14, true);
		let B = source.getUint16(18 * i + 16, true);

		let x = X * scale[0];// + offset[0];
		let y = Y * scale[1];// + offset[1];
		let z = Z * scale[2];// + offset[2];

		let r = R <= 255 ? R : R / 256;
		let g = G <= 255 ? G : G / 256;
		let b = B <= 255 ? B : B / 256;

		target.setFloat32(16 * i + 0, x, true);
		target.setFloat32(16 * i + 4, y, true);
		target.setFloat32(16 * i + 8, z, true);
		target.setUint8(16 * i + 12, r);
		target.setUint8(16 * i + 13, g);
		target.setUint8(16 * i + 14, b);
		target.setUint8(16 * i + 15, 255);

	}

	return vboData;
}


if(typeof e4called === "undefined"){
	e4called = true;
	
	let dir = "D:/dev/pointclouds/mschuetz/lion_s512";
	// let dir = "D:/dev/pointclouds/lion_converted";
	let metadata = JSON.parse(readTextFile(`${dir}/metadata.json`));
	let dataHierarchy = readFile(`${dir}/hierarchy.bin`);
	let dataPoints = readFile(`${dir}/octree.bin`);

	log(dataHierarchy.byteLength);
	log(dataPoints.byteLength);

	let viewHierarchy = new DataView(dataHierarchy);
	let viewPoints = new DataView(dataPoints);

	let bytesPerNode = 22;
	let numNodes = dataHierarchy.byteLength / bytesPerNode;
	let root = {name: "r"};
	let nodes = [root];
	let nodeMap = new Map();
	nodeMap.set("r", root);

	for(let i = 0; i < numNodes; i++){

		let node = nodes[i];

		let type = viewHierarchy.getUint8(bytesPerNode * i + 0);
		let childMask = viewHierarchy.getUint8(bytesPerNode * i + 1);
		let numPoints = viewHierarchy.getUint32(bytesPerNode * i + 2, true);
		let byteOffset = viewHierarchy.getBigInt64(bytesPerNode * i + 6, true);
		let byteSize = viewHierarchy.getBigInt64(bytesPerNode * i + 14, true);

		node.numPoints = numPoints;
		node.byteOffset = Number(byteOffset);
		node.byteSize = Number(byteSize);

		for(let childIndex = 0; childIndex < 8; childIndex++){

			let childExists = ((1 << childIndex) & childMask) !== 0;
			if(!childExists){
				continue;
			}
		
			let childName = `${node.name}${childIndex}`;
			let childNode = {name: childName};

			nodes.push(childNode);
			nodeMap.set(childName, childNode);
		};

	}

	let attributes = [
		new GLBufferAttribute("position", 0, 3, gl.FLOAT, gl.FALSE, 12, 0),
		new GLBufferAttribute("color",    1, 4, gl.UNSIGNED_BYTE, gl.TRUE, 4, 12),
	];

	let node = nodes[0];
	let pc = new PointCloudProgressive(node.name);

	// { // many buffers

	// 	let glBuffers = nodes.map(node => {
	// 		let vboData = parsePoints(metadata, node, dataPoints);
	// 		let glBuffer = new GLBuffer();
	// 		glBuffer.setInterleaved(vboData, attributes, node.numPoints);

	// 		return glBuffer;
	// 	});

	// 	let smallBuffers = glBuffers.filter(b => b.count <= 10000);
	// 	glBuffers = glBuffers.filter(b => b.count > 10000);

	// 	{
	// 		let mergedSize = smallBuffers.reduce((a, v) => a + v.buffer.byteLength, 0);
	// 		let mergedCount = smallBuffers.reduce((a, v) => a + v.count, 0);
	// 		let mergedBuffer = new Uint8Array(mergedSize);
	// 		let offset = 0;
	// 		for(let buffer of smallBuffers){
	// 			mergedBuffer.set(new Uint8Array(buffer.buffer), offset);
	// 			offset += buffer.buffer.byteLength;
	// 		}
	// 		let glBuffer = new GLBuffer();
	// 		glBuffer.setInterleaved(mergedBuffer.buffer, attributes, mergedCount);

	// 		log(`mergedSize: ${mergedSize}`);
	// 		log(`mergedCount: ${mergedCount}`);

	// 		glBuffers.push(glBuffer);
	// 	}

	// 	pc.glBuffers = glBuffers;


	// 	log("#nodes: " + glBuffers.length);
	// 	log(glBuffers.map(b => b.count).join(", "));
	// }

	{ // single buffer
		let glBuffer = new GLBuffer();
		let vboData = new ArrayBuffer(16 * metadata.points);

		let processed = 0;
		for(let node of nodes){
			let nodeVboData = parsePoints(metadata, node, dataPoints);
			new Uint8Array(vboData).set(new Uint8Array(nodeVboData), 16 * processed);

			processed += node.numPoints;
		}

		glBuffer.setInterleaved(vboData, attributes, metadata.points);
		pc.glBuffers = [glBuffer];
	}

	let s = 1;
	pc.transform.elements.set([
		s, 0, 0, 0, 
		0, 0, -s, 0, 
		0, s, 0, 0, 
		0, 0, 0, 1, 
	]);

	scene.root.add(pc);

}

window.width = 1920;
window.height = 1080;
window.x = 2560;
window.y = 0;

view.set(
	[4.482548084301243, 3.9856851176075163, 1.9479170192704958],
	[0.6162662017196654, 1.65415456111992, -2.6637675862770447],
);

camera.fov = 60;
camera.near = 0.1;

setTimestampEnabled(false);


if(typeof af032f !== "undefined"){
	listeners.update = listeners.update.filter(l => l !== af032f);
}

var dbgLast = now();
var dbgLastFC = frameCount;
var af032f = () => {

	if(now() - dbgLast > 1){
		log(frameCount - dbgLastFC);
		dbgLast = now();

		dbgLastFC = frameCount;
	}

};
listeners.update.push(af032f);


renderDebug = render_compute_ballot_earlyDepth;
renderDebug = renderPointCloudBasic