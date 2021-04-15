
class Node{
	
	constructor(){
		this.children = new Array(8).fill(null);
		this.index = null;
		this.level = 0;
		this.name = "";
		this.boundingBox = null;
		this.buffer = null;
		this.numPoints = 0;
		this.hasHierarchyToLoad = false;
	}

	find(name){

		if(name.length > this.name.length){
			let remaining = name.replace(this.name, "");
			let index = parseInt(remaining.charAt(0));

			let child = this.children[index];

			if(child){
				return child.find(name);
			}else{
				return null;
			}
		}else if(name.length < this.name.length){
			return null;
		}else if(this.name == name){
			return this;
		}else{
			return null;
		}
	}

	traverse(callback){
		let stack = [{node: this, level: 0}];

		while(stack.length > 0){
			let entry = stack.pop();
			let node = entry.node;
			let level = entry.level;

			let expand = callback(node, level);

			if(expand === false){
				continue;
			}

			let children = node.children.filter( c => c !== null );
			for(let child of children.reverse()){
				stack.push({node: child, level: level + 1});
			}
		}
	}

	level(){
		return this.name.length - 1;
	}

}

function getHierarchyPath(name, hierarchyStepSize){
	let path = "r/";
	let indices = name.substr(1);
	let numParts = Math.floor(indices.length / hierarchyStepSize);
	for (let i = 0; i < numParts; i++) {
		path += indices.substr(i * hierarchyStepSize, hierarchyStepSize) + '/';
	}
	path = path.slice(0, -1);
	return path;
}

function parseHierarchy(hrcData, rootName, box, hierarchyStepSize){

	let root = new Node();
	root.hasHierarchyToLoad = true;
	root.name = rootName;
	root.boundingBox = box;

	let nodes = [root];

	let n = hrcData.length / 5;

	let msg = "";

	for(let i = 0; i < n; i++){
		let childMask = hrcData[5 * i];

		let node = nodes[i];

		for(let j = 0; j < 8; j++){
			let hasChildJ = childMask & (1 << j);

			if(hasChildJ){

				let child = new Node();
				child.index = j;
				child.name = `${node.name}${j}`;
				child.level = child.name.length - 1;
				child.boundingBox = createChildAABB(node.boundingBox, child.index);
				child.hasHierarchyToLoad = (child.level % hierarchyStepSize) === 0;

				node.children[j] = child;

				nodes.push(child);
			}
		}
	}

	return root;
}

function createChildAABB(aabb, index){
	let min = aabb.min.clone();
	let max = aabb.max.clone();
	let size = max.clone().sub(min);

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

let pointsLoaded = 0;
let nodesLoaded = 0;
let loadDuration = 0;

class PotreeLoader{

	constructor(path){

		this.path = path;
		let cloudjsStr = readTextFile(path);
		this.meta = JSON.parse(cloudjsStr);
		this.attributes = new PointAttributes(this.meta.pointAttributes.map(name => PointAttribute[name]));

		let {lx, ly, lz, ux, uy, uz} = this.meta.boundingBox;
		let min = new Vector3(lx, ly, lz);
		let max = new Vector3(ux, uy, uz);

		max = max.sub(min);
		min = new Vector3(0, 0, 0);
		let boundingBox = new Box3(min, max);

		let hrcRoot = `${path}/../data/r/r.hrc`;
		let hrcData = readFile(hrcRoot);
		hrcData = new Uint8Array(hrcData);


		this.start = now();
		
		//let root = parseHierarchy(hrcData, "r", boundingBox, this.meta.hierarchyStepSize);
		let root = new Node();
		root.hasHierarchyToLoad = true;
		root.name = "r";
		root.boundingBox = boundingBox;
		this.root = root;
		//this.root.hasHierarchyToLoad = false;

		//this.root.traverse( node => {

		//	log(node.name);

		//	return true;
		//});

		//this.toLoad = [];

		//root.traverse( (node, level) => {
		//	if(level > 10){
		//		return false;
		//	}

		//	this.toLoad.push(node);
		//});

		//this.toLoad.sort( (a, b) => {
		//	return b.name.length - a.name.length;
		//});

	}

	async loadHierarchy(node){

		let hierarchyPath = getHierarchyPath(node.name, this.meta.hierarchyStepSize);
		let hrcPath = `${this.path}/../data/${hierarchyPath}/${node.name}.hrc`;
		let hrcData = await readFile(hrcPath);
		hrcData = new Uint8Array(hrcData);

		//if(node.name === "r40624"){
		//	log("10");
		//}

		let localRoot = parseHierarchy(hrcData, node.name, node.boundingBox, this.meta.hierarchyStepSize);

		//if(node.name === "r40624"){
		//	let numNodes = hrcData.byteLength / 5;

		//	additionalHierarchy.traverse( node => {

		//		log(node.name);

		//		return true;
		//	});
		//}


		node.children = localRoot.children;
	}

	async load(name){

		//log(`load ${name}`);
		let node = this.root.find(name);

		let additionalHierarchy = null;
		if(node.hasHierarchyToLoad){
			await this.loadHierarchy(node);
			additionalHierarchy = node;
		}



		let hierarchyPath = getHierarchyPath(node.name, this.meta.hierarchyStepSize);
		let nodePath = `${this.path}/../data/${hierarchyPath}/${node.name}.bin`;

		let scale = this.meta.scale;
		let bbMin = node.boundingBox.min;

		let vertexData = await loadNodeAsync(nodePath, scale, bbMin.x, bbMin.y, bbMin.z);
		let numPoints = vertexData.byteLength / 16;

		//let data = await readFileAsync(nodePath);

		//let dataU8 = new Uint8Array(data);
		////let dataU32 = new Uint32Array(data);
		//let dataView = new DataView(data);
		//let bytesPerPoint = this.attributes.bytes;
		//let numPoints = data.byteLength / bytesPerPoint;

		//let tmp = new ArrayBuffer(4);
		//let tmp8 = new Uint8Array(tmp);
		//let tmp32 = new Uint32Array(tmp);
		////numPoints = 10;

		//let vertexData = new ArrayBuffer(numPoints * 16);
		//let vertices = new Float32Array(vertexData);
		//let verticesU8 = new Uint8Array(vertexData);

		//pointsLoaded += numPoints;
		//nodesLoaded++;

		//let shuffle = function(a) {
		//	for (let i = a.length - 1; i > 0; i--) {
		//		const j = Math.floor(Math.random() * (i + 1));
		//		[a[i], a[j]] = [a[j], a[i]];
		//	}
		//	return a;
		//};

		//let positions = new Array(numPoints);
		//let boxSize = node.boundingBox.getSize();
		//for(let i = 0; i < numPoints; i++){

		//	// 30M points / sec
		//	tmp8[0] = dataU8[i * bytesPerPoint + 0];
		//	tmp8[1] = dataU8[i * bytesPerPoint + 1];
		//	tmp8[2] = dataU8[i * bytesPerPoint + 2];
		//	tmp8[3] = dataU8[i * bytesPerPoint + 3];
		//	let ux = tmp32[0];

		//	tmp8[0] = dataU8[i * bytesPerPoint + 4];
		//	tmp8[1] = dataU8[i * bytesPerPoint + 5];
		//	tmp8[2] = dataU8[i * bytesPerPoint + 6];
		//	tmp8[3] = dataU8[i * bytesPerPoint + 7];
		//	let uy = tmp32[0];

		//	tmp8[0] = dataU8[i * bytesPerPoint + 8];
		//	tmp8[1] = dataU8[i * bytesPerPoint + 9];
		//	tmp8[2] = dataU8[i * bytesPerPoint + 10];
		//	tmp8[3] = dataU8[i * bytesPerPoint + 11];
		//	let uz = tmp32[0];

		//	let mx = parseInt(((ux * 0.8) / boxSize.x) * 1024);
		//	let my = parseInt(((uy * 0.8) / boxSize.y) * 1024);
		//	let mz = parseInt(((uz * 0.8) / boxSize.z) * 1024);

		//	let w = mx | (my << 8) | (mz << 16);
		//	
		//	positions.push({index: i, w: w});
		//}

		//positions.sort( (a, b) => {
		//	return a.w - b.w
		//});
		//let order = positions.map( p => p.index);

		////let order = new Array(numPoints).fill(0).map( (a, i) => i);
		////order = shuffle(order);

		//for(let j = 0; j < numPoints; j++){
		//	let i = order[j];
		////for(let i = 0; i < numPoints; i++){

		//	// 8M points / Sec
		//	//let ux = dataView.getInt32(i * bytesPerPoint + 0, true);
		//	//let uy = dataView.getInt32(i * bytesPerPoint + 4, true);
		//	//let uz = dataView.getInt32(i * bytesPerPoint + 8, true);

		//	// 70-90M points / sec
		//	//let ux = dataU32[i * 4 + 0];
		//	//let uy = dataU32[i * 4 + 1];
		//	//let uz = dataU32[i * 4 + 2];

		//	// 30M points / sec
		//	tmp8[0] = dataU8[i * bytesPerPoint + 0];
		//	tmp8[1] = dataU8[i * bytesPerPoint + 1];
		//	tmp8[2] = dataU8[i * bytesPerPoint + 2];
		//	tmp8[3] = dataU8[i * bytesPerPoint + 3];
		//	let ux = tmp32[0];

		//	tmp8[0] = dataU8[i * bytesPerPoint + 4];
		//	tmp8[1] = dataU8[i * bytesPerPoint + 5];
		//	tmp8[2] = dataU8[i * bytesPerPoint + 6];
		//	tmp8[3] = dataU8[i * bytesPerPoint + 7];
		//	let uy = tmp32[0];

		//	tmp8[0] = dataU8[i * bytesPerPoint + 8];
		//	tmp8[1] = dataU8[i * bytesPerPoint + 9];
		//	tmp8[2] = dataU8[i * bytesPerPoint + 10];
		//	tmp8[3] = dataU8[i * bytesPerPoint + 11];
		//	let uz = tmp32[0];

		//	let scale = this.meta.scale;
		//	let x = ux * scale + node.boundingBox.min.x;
		//	let y = uy * scale + node.boundingBox.min.y;
		//	let z = uz * scale + node.boundingBox.min.z;

		//	let r = dataU8[i * bytesPerPoint + 12];
		//	let g = dataU8[i * bytesPerPoint + 13];
		//	let b = dataU8[i * bytesPerPoint + 14];

		//	vertices[4 * j + 0] = x;
		//	vertices[4 * j + 1] = y;
		//	vertices[4 * j + 2] = z;

		//	verticesU8[16 * j + 12] = r;
		//	verticesU8[16 * j + 13] = g;
		//	verticesU8[16 * j + 14] = b;
		//	verticesU8[16 * j + 15] = 255;
		//}

		node.numPoints = numPoints;
		node.data = vertexData;

		let ret = {
			numPoints: numPoints, 
			vertices: vertexData, 
			additionalHierarchy: additionalHierarchy};

		return ret;
	}

};