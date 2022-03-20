
// convert a point cloud from potree2 format to workgroup-render format

// const fs = require('fs');

import * as fs from 'fs';
const fsp = fs.promises;

import {Vector3} from "./src/Vector3.mjs";
import {Box3} from "./src/Box3.mjs";
import {Bitstream} from "./src/Bitstream.mjs";

// let path = "F:/pointclouds/benchmark/endeavor/morton.las_converted";
// let outpath = "F:/temp/wgtest/endeavor_blockwise";

let path = "F:/pointclouds/benchmark/retz/morton.las_converted";
let outpath = "F:/temp/wgtest/retz/compressed_nodewise";

// let path = "F:/pointclouds/lion.las_converted";
// let outpath = "F:/temp/wgtest/lion_blockwise";

// let path = "F:/pointclouds/benchmark/lifeboat/morton.las_converted";
// let outpath = "F:/temp/wgtest/lifeboat_blockwise";

const NodeType = {
	NORMAL: 0,
	LEAF: 1,
	PROXY: 2,
};



class Node{

	constructor(){
		this.name = "";
		this.level = 0;
		this.numPoints = 0;
		this.children = [];
	}

	traverse(callback){

		callback(this);

		for(let child of this.children){
			if(child){
				child.traverse(callback);
			}
		}

	}

}

let tmpVec3 = new Vector3();
function createChildAABB(aabb, index){
	let min = aabb.min.clone();
	let max = aabb.max.clone();
	let size = tmpVec3.copy(max).sub(min);

	if ((index & 0b0001) > 0) {
		min.z += size.z / 2;
	} else {
		max.z -= size.z / 2;
	}

	if ((index & 0b0010) > 0) {
		min.y += size.y / 2;
	} else {
		max.y -= size.y / 2;
	}
	
	if ((index & 0b0100) > 0) {
		min.x += size.x / 2;
	} else {
		max.x -= size.x / 2;
	}

	return new Box3(min, max);
}

function loadHierarchy(root, buffer){

	let bytesPerNode = 22;
	let numNodes = buffer.byteLength / bytesPerNode;

	let nodes = new Array(numNodes);
	nodes[0] = root;
	let nodePos = 1;

	for(let i = 0; i < numNodes; i++){
		let current = nodes[i];

		let type = buffer.readUInt8(i * bytesPerNode + 0);
		let childMask = buffer.readUInt8(i * bytesPerNode + 1);
		let numPoints = buffer.readUInt32LE(i * bytesPerNode + 2);
		let byteOffset = Number(buffer.readBigInt64LE(i * bytesPerNode + 6));
		let byteSize = Number(buffer.readBigInt64LE(i * bytesPerNode + 14));

		if(current.nodeType === NodeType.PROXY){
			// replace proxy with real node
			current.byteOffset = byteOffset;
			current.byteSize = byteSize;
			current.numPoints = numPoints;
		}else if(type === NodeType.PROXY){
			// load proxy
			current.hierarchyByteOffset = byteOffset;
			current.hierarchyByteSize = byteSize;
			current.numPoints = numPoints;
		}else{
			// load real node 
			current.byteOffset = byteOffset;
			current.byteSize = byteSize;
			current.numPoints = numPoints;
		}
		
		current.nodeType = type;

		if(current.nodeType === NodeType.PROXY){
			continue;
		}

		for(let childIndex = 0; childIndex < 8; childIndex++){
			let childExists = ((1 << childIndex) & childMask) !== 0;

			if(!childExists){
				continue;
			}

			let childName = current.name + childIndex;

			let child = new Node();
			child.name = childName;
			child.boundingBox = createChildAABB(current.boundingBox, childIndex);
			child.name = childName;
			child.spacing = current.spacing / 2;
			child.level = current.level + 1;
			// child.octree = this.octree;

			current.children[childIndex] = child;
			child.parent = current;

			nodes[nodePos] = child;
			nodePos++;
		}
	}
}

async function generateWgBatches(root, metadata){

	try{
		await fsp.mkdir(outpath);
	}catch(e){}

	let fBatches = fs.createWriteStream(outpath + "/batches.bin");
	let fPoints = fs.createWriteStream(outpath + "/points.bin");
	let fColors = fs.createWriteStream(outpath + "/colors.bin");

	let fd = fs.openSync(path + "/octree.bin");

	let sourceStride = 18;

	let scale = new Vector3(...metadata.scale);
	let offset = new Vector3(...metadata.offset);
	// let bbSize = root.boundingBox.size();
	let pointsWritten = 0;
	let batchesWritten = 0;
	let bitsTotal = 0;
	let posBytesWritten = 0;

	root.traverse(node => {

		// if(batchesWritten > 1000){
		// 	return;
		// }

		
		let numPoints = node.numPoints;

		let sourceBuffer = Buffer.alloc(18 * numPoints);
		fs.readSync(fd, sourceBuffer, 0, node.byteSize, node.byteOffset);

		let box = new Box3();

		for(let i = 0; i < numPoints; i++){
			let X = sourceBuffer.readInt32LE(i * sourceStride + 0);
			let Y = sourceBuffer.readInt32LE(i * sourceStride + 4);
			let Z = sourceBuffer.readInt32LE(i * sourceStride + 8);

			let x = X * scale.x + offset.x;
			let y = Y * scale.y + offset.y;
			let z = Z * scale.z + offset.z;

			box.expandByXYZ(x, y, z);
		}

		let boxSize = box.size();

		let {ceil, log2} = Math;
		let precision = 0.001;
		let bitsX = Math.max(boxSize.x == 0 ? 0 : ceil(log2(boxSize.x / precision)), 0);
		let bitsY = Math.max(boxSize.y == 0 ? 0 : ceil(log2(boxSize.y / precision)), 0);
		let bitsZ = Math.max(boxSize.z == 0 ? 0 : ceil(log2(boxSize.z / precision)), 0);
		let factorX = 2 ** bitsX;
		let factorY = 2 ** bitsY;
		let factorZ = 2 ** bitsZ;

		let bits = bitsX + bitsY + bitsZ;

		if(bits < 0 || bits > 100){
			bits = 0;
			debugger;
		}

		let alignTo20 = (value) => value + (20 - (value % 20));
		let alignTo16 = (value) => value + (16 - (value % 16));
		let alignTo4 = (value) => value + (4 - (value % 4));
		let posBufferSize = alignTo20(Math.ceil((bits * numPoints) / 8));
		let colorBufferSize = 4 * numPoints;

		let posBitstream = new Bitstream(posBufferSize);
		let colorBuffer = Buffer.alloc(colorBufferSize);

		for(let i = 0; i < numPoints; i++){

			{// Position, X bit
				let X = sourceBuffer.readInt32LE(i * sourceStride + 0);
				let Y = sourceBuffer.readInt32LE(i * sourceStride + 4);
				let Z = sourceBuffer.readInt32LE(i * sourceStride + 8);

				let x = X * scale.x + offset.x;
				let y = Y * scale.y + offset.y;
				let z = Z * scale.z + offset.z;

				let ix = factorX * ((x - box.min.x) / boxSize.x);
				let iy = factorY * ((y - box.min.y) / boxSize.y);
				let iz = factorZ * ((z - box.min.z) / boxSize.z);

				ix = Math.min(ix, factorX - 1);
				iy = Math.min(iy, factorY - 1);
				iz = Math.min(iz, factorZ - 1);

				if(pointsWritten < 2){
					console.log(`posBitstream.write(${ix}, ${bitsX});`);
					console.log(`posBitstream.write(${iy}, ${bitsX});`);
					console.log(`posBitstream.write(${iz}, ${bitsX});`);
					console.log(``);
				}

				posBitstream.write(ix, bitsX);
				posBitstream.write(iy, bitsY);
				posBitstream.write(iz, bitsZ);
			}

			{ // RGB
				let R = sourceBuffer.readUInt16LE(i * sourceStride + 12);
				let G = sourceBuffer.readUInt16LE(i * sourceStride + 14);
				let B = sourceBuffer.readUInt16LE(i * sourceStride + 16);

				R = Math.floor(R > 255 ? R / 256 : R);
				G = Math.floor(G > 255 ? G / 256 : G);
				B = Math.floor(B > 255 ? B / 256 : B);

				colorBuffer.writeUInt8(R, 4 * i + 0);
				colorBuffer.writeUInt8(G, 4 * i + 1);
				colorBuffer.writeUInt8(B, 4 * i + 2);
				colorBuffer.writeUInt8(255, 4 * i + 3);
			}

			pointsWritten++;
		}

		let bitsInBlock = bits * numPoints;
		bitsTotal += bitsInBlock;

		// console.log("====");
		// console.log(size.x.toFixed(3), size.y.toFixed(3), size.z.toFixed(3));
		// console.log(`bits: ${bits} (${bitsX}, ${bitsY}, ${bitsZ})`);

		fPoints.write(posBitstream.buffer);
		fColors.write(colorBuffer);

		node.output = {
			minimumBoundingBox: box,
			pointOffset: pointsWritten - numPoints,
			posByteOffset: posBytesWritten,
			bitsX, bitsY, bitsZ, bits,
		}

		batchesWritten++;
		posBytesWritten += posBufferSize;

	});

	let bytesTotal = Math.ceil(bitsTotal / 8);
	let mbCompressed = bytesTotal / (1024 ** 2);
	let mbUncompressed = (pointsWritten * 12) / (1024 ** 2);
	let bitsPerPoint = bitsTotal / pointsWritten;

	console.log("uncompressed size: ", mbUncompressed);
	console.log("compressed size: ", mbCompressed);
	console.log("bitsPerPoint: ", Math.ceil(bitsPerPoint));

	fPoints.end();
	fColors.end();


	{ // write batches

		let i = 0;
		root.traverse(node => {

			let batchBuffer = Buffer.alloc(64);

			let box = node.output.minimumBoundingBox;

			batchBuffer.writeFloatLE(box.min.x,  0);
			batchBuffer.writeFloatLE(box.min.y,  4);
			batchBuffer.writeFloatLE(box.min.z,  8);
			batchBuffer.writeFloatLE(box.max.x, 12);
			batchBuffer.writeFloatLE(box.max.y, 16);
			batchBuffer.writeFloatLE(box.max.z, 20);

			batchBuffer.writeUInt8(255, 24 + 0);
			batchBuffer.writeUInt8( 50, 24 + 1);
			batchBuffer.writeUInt8( 50, 24 + 2);
			batchBuffer.writeUInt8(255, 24 + 3);

			batchBuffer.writeInt32LE(node.numPoints, 28);
			batchBuffer.writeInt32LE(node.output.pointOffset, 32);

			let blockOffset = node.output.posByteOffset / 20;
			if(!Number.isInteger(blockOffset)){
				throw "should have been an integer";
			}

			batchBuffer.writeInt32LE(blockOffset, 36);
			batchBuffer.writeInt32LE(node.output.bitsX, 40);
			batchBuffer.writeInt32LE(node.output.bitsY, 44);
			batchBuffer.writeInt32LE(node.output.bitsZ, 48);
			batchBuffer.writeInt32LE(node.output.bits,  52);
			
			fBatches.write(batchBuffer);

			if(i === 0){
				console.log("node.output.pointOffset", node.output.pointOffset);
				console.log("node.output.posByteOffset", node.output.posByteOffset);
			}
			
			i++;
		});

		fBatches.end();
	}

	let stats = {
		numBatches: batchesWritten,
		bitsPerPoint: Math.ceil(bitsPerPoint),
	};

	return stats;
}

async function run(){
	
	let txtJson = (await fsp.readFile(path + "/metadata.json")).toString();
	let json = JSON.parse(txtJson);
	
	let binHierarchy = await fsp.readFile(path + "/hierarchy.bin");
	let numNodes = binHierarchy.byteLength / 22;

	let boundingBox = new Box3(
		new Vector3(...json.boundingBox.min),
		new Vector3(...json.boundingBox.max),
	);

	let numPoints = json.points;
	console.log({numNodes, numPoints});

	let root = new Node();
	root.name = "r";
	root.boundingBox = boundingBox;
	root.spacing = json.spacing;
	root.nodeType = NodeType.NORMAL;
	loadHierarchy(root, binHierarchy);

	let stats = await generateWgBatches(root, json);


	let metadata = {
		points: numPoints,
		batches: stats.numBatches,
		boundingBox: json.boundingBox,
	};

	let strMetadata = JSON.stringify(metadata, null, '\t');

	await fsp.writeFile(outpath + "/metadata.json", strMetadata);

}

run();

console.log("test");