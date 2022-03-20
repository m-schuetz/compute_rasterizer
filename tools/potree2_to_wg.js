
// convert a point cloud from potree2 format to workgroup-render format

const fs = require('fs');
const fsp = fs.promises;

// let path = "F:/pointclouds/benchmark/retz/morton.las_converted";
// let outpath = "F:/temp/wgtest/retz";
// let path = "E:/dev/pointclouds/benchmark/lifeboat/morton.las_converted";
// let outpath = "E:/temp/wgtest";
let path = "D:/dev/pointclouds/lion_converted";
let outpath = "F:/temp/wgtest/lion";

const NodeType = {
	NORMAL: 0,
	LEAF: 1,
	PROXY: 2,
};

class Vector3{

	constructor(x, y, z){
		this.x = x ?? 0;
		this.y = y ?? 0;
		this.z = z ?? 0;
	}

	set(x, y, z){
		this.x = x;
		this.y = y;
		this.z = z;

		return this;
	}

	copy(b){
		this.x = b.x;
		this.y = b.y;
		this.z = b.z;

		return this;
	}

	multiplyScalar(s){
		this.x = this.x * s;
		this.y = this.y * s;
		this.z = this.z * s;

		return this;
	}

	divideScalar(s){
		this.x = this.x / s;
		this.y = this.y / s;
		this.z = this.z / s;

		return this;
	}

	add(b){
		this.x = this.x + b.x;
		this.y = this.y + b.y;
		this.z = this.z + b.z;

		return this;
	}

	addScalar(s){
		this.x = this.x + s;
		this.y = this.y + s;
		this.z = this.z + s;

		return this;
	}

	sub(b){
		this.x = this.x - b.x;
		this.y = this.y - b.y;
		this.z = this.z - b.z;

		return this;
	}

	subScalar(s){
		this.x = this.x - s;
		this.y = this.y - s;
		this.z = this.z - s;

		return this;
	}

	subVectors( a, b ) {

		this.x = a.x - b.x;
		this.y = a.y - b.y;
		this.z = a.z - b.z;

		return this;
	}

	cross(v) {
		return this.crossVectors( this, v );
	}

	crossVectors( a, b ) {

		const ax = a.x, ay = a.y, az = a.z;
		const bx = b.x, by = b.y, bz = b.z;

		this.x = ay * bz - az * by;
		this.y = az * bx - ax * bz;
		this.z = ax * by - ay * bx;

		return this;
	}

	dot( v ) {
		return this.x * v.x + this.y * v.y + this.z * v.z;
	}

	distanceTo( v ) {
		return Math.sqrt( this.distanceToSquared( v ) );
	}

	distanceToSquared( v ) {
		const dx = this.x - v.x, dy = this.y - v.y, dz = this.z - v.z;

		return dx * dx + dy * dy + dz * dz;
	}

	clone(){
		return new Vector3(this.x, this.y, this.z);
	}

	applyMatrix4(m){
		const x = this.x, y = this.y, z = this.z;
		const e = m.elements;

		const w = 1 / ( e[ 3 ] * x + e[ 7 ] * y + e[ 11 ] * z + e[ 15 ] );

		this.x = ( e[ 0 ] * x + e[ 4 ] * y + e[ 8 ] * z + e[ 12 ] ) * w;
		this.y = ( e[ 1 ] * x + e[ 5 ] * y + e[ 9 ] * z + e[ 13 ] ) * w;
		this.z = ( e[ 2 ] * x + e[ 6 ] * y + e[ 10 ] * z + e[ 14 ] ) * w;

		return this;
	}

	length() {
		return Math.sqrt( this.x * this.x + this.y * this.y + this.z * this.z );
	}

	lengthSq() {
		return this.x * this.x + this.y * this.y + this.z * this.z;
	}

	normalize(){
		let l = this.length();

		this.x = this.x / l;
		this.y = this.y / l;
		this.z = this.z / l;

		return this;
	}

	toString(precision){
		if(precision != null){
			return `${this.x.toFixed(precision)}, ${this.y.toFixed(precision)}, ${this.z.toFixed(precision)}`;
		}else{
			return `${this.x}, ${this.y}, ${this.z}`;
		}
	}

	toArray(){
		return [this.x, this.y, this.z];
	}

	isFinite(){
		let {x, y, z} = this;
		
		return Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z);
	}

};

class Box3{

	constructor(min, max){
		this.min = min ?? new Vector3(+Infinity, +Infinity, +Infinity);
		this.max = max ?? new Vector3(-Infinity, -Infinity, -Infinity);
	}

	clone(){
		return new Box3(
			this.min.clone(),
			this.max.clone()
		);
	}

	copy(box){
		this.min.copy(box.min);
		this.max.copy(box.max);
	}

	size(){
		return this.max.clone().sub(this.min);
	}

	center(){
		return this.min.clone().add(this.max).multiplyScalar(0.5);
	}

	cube(){
		let cubeSize = Math.max(...this.size().toArray());
		let min = this.min.clone();
		let max = this.min.clone().addScalar(cubeSize);
		let cube = new Box3(min, max);

		return cube;
	}

	expandByXYZ(x, y, z){
		this.min.x = Math.min(this.min.x, x);
		this.min.y = Math.min(this.min.y, y);
		this.min.z = Math.min(this.min.z, z);

		this.max.x = Math.max(this.max.x, x);
		this.max.y = Math.max(this.max.y, y);
		this.max.z = Math.max(this.max.z, z);
	}

	expandByPoint(point){
		this.min.x = Math.min(this.min.x, point.x);
		this.min.y = Math.min(this.min.y, point.y);
		this.min.z = Math.min(this.min.z, point.z);

		this.max.x = Math.max(this.max.x, point.x);
		this.max.y = Math.max(this.max.y, point.y);
		this.max.z = Math.max(this.max.z, point.z);
	}

	expandByBox(box){
		this.expandByPoint(box.min);
		this.expandByPoint(box.max);
	}

	applyMatrix4(matrix){

		let {min, max} = this;

		let points = [
			new Vector3(min.x, min.y, min.z),
			new Vector3(min.x, min.y, max.z),
			new Vector3(min.x, max.y, min.z),
			new Vector3(min.x, max.y, max.z),
			new Vector3(max.x, min.y, min.z),
			new Vector3(max.x, min.y, max.z),
			new Vector3(max.x, max.y, min.z),
			new Vector3(max.x, max.y, max.z),
		];

		let newBox = new Box3();

		for(let point of points){
			let projected = point.applyMatrix4(matrix);
			newBox.expandByPoint(projected);
		}

		this.min.copy(newBox.min);
		this.max.copy(newBox.max);

		return this;
	}

	isFinite(){
		return this.min.isFinite() && this.max.isFinite();
	}

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
			child.octree = this.octree;

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
	let bbSize = root.boundingBox.size();
	let pointsWritten = 0;

	root.traverse(node => {
		let batchBuffer = Buffer.alloc(64);

		batchBuffer.writeFloatLE(node.boundingBox.min.x,  0);
		batchBuffer.writeFloatLE(node.boundingBox.min.y,  4);
		batchBuffer.writeFloatLE(node.boundingBox.min.z,  8);
		batchBuffer.writeFloatLE(node.boundingBox.max.x, 12);
		batchBuffer.writeFloatLE(node.boundingBox.max.y, 16);
		batchBuffer.writeFloatLE(node.boundingBox.max.z, 20);

		batchBuffer.writeUInt8(255, 24 + 0);
		batchBuffer.writeUInt8( 50, 24 + 1);
		batchBuffer.writeUInt8( 50, 24 + 2);
		batchBuffer.writeUInt8(255, 24 + 3);

		batchBuffer.writeInt32LE(node.numPoints, 28);
		batchBuffer.writeInt32LE(pointsWritten, 32);

		fBatches.write(batchBuffer);
		let numPoints = node.numPoints;
		let targetBuffer = Buffer.alloc(8 * numPoints);
		let colorBuffer = Buffer.alloc(4 * numPoints);

		let sourceBuffer = Buffer.alloc(18 * numPoints);
		fs.readSync(fd, sourceBuffer, 0, node.byteSize, node.byteOffset);

		for(let i = 0; i < numPoints; i++){
			let X = sourceBuffer.readInt32LE(i * sourceStride + 0);
			let Y = sourceBuffer.readInt32LE(i * sourceStride + 4);
			let Z = sourceBuffer.readInt32LE(i * sourceStride + 8);

			let x = X * scale.x + offset.x;
			let y = Y * scale.y + offset.y;
			let z = Z * scale.z + offset.z;

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

			{// 20 bit
				let factor = 1048576.0;
				let ix = factor * ((x - root.boundingBox.min.x) / bbSize.x);
				let iy = factor * ((y - root.boundingBox.min.y) / bbSize.y);
				let iz = factor * ((z - root.boundingBox.min.z) / bbSize.z);

				let a = ix | ((iy & 4095) << 20);
				let b = iz | ((iy >> 12) << 20);


				targetBuffer.writeInt32LE(a, i * 8 + 0);
				targetBuffer.writeInt32LE(b, i * 8 + 4);
			}

			// {// 16 bit
			// 	let factor = 65536.0;
			// 	let ix = factor * ((x - root.boundingBox.min.x) / bbSize.x);
			// 	let iy = factor * ((y - root.boundingBox.min.y) / bbSize.y);
			// 	let iz = factor * ((z - root.boundingBox.min.z) / bbSize.z);

			// 	let a = ix | (iy << 16);
			// 	let b = iz;

			// 	targetBuffer.writeInt32LE(a, i * 8 + 0);
			// 	targetBuffer.writeInt32LE(b, i * 8 + 4);
			// }

			pointsWritten++;
		}

		fPoints.write(targetBuffer);
		fColors.write(colorBuffer);

	});


	fBatches.end();
	fPoints.end();
	fColors.end();


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

	let root = new Node();
	root.name = "r";
	root.boundingBox = boundingBox;
	root.spacing = json.spacing;
	root.nodeType = NodeType.NORMAL;
	loadHierarchy(root, binHierarchy);

	generateWgBatches(root, json);
}

run();

console.log("test");