


if(typeof PointCloudOctree === "undefined"){

	PointCloudOctreeNode = class PointCloudOctreeNode{
	
		constructor(name, boundingBox){
			this.name = name;
			this.children = new Array(8).fill(null);
			this.boundingBox = boundingBox;
			this.buffer = null;
		}

		traverse(callback, level = 0){

			callback(this, level);

			for(let child of this.children){
				if(child !== null){
					child.traverse(callback, level + 1);
				}
			}

		}

	}

	PointCloudOctree = class PointCloudOctree extends SceneNode{

		constructor(name, path){
			super(name);

			this.path = path;
			this.loader = new PotreeLoader(path);
			this.root = new PointCloudOctreeNode("r", this.loader.root.boundingBox);
			this.visibleNodes = [];
			//this.pointBudget = 2 * 1000 * 1000;

			{
				let vsPath = `${rootDir}/modules/octree/pointcloud.vs`;
				let fsPath = `${rootDir}/modules/octree/pointcloud.fs`;
				let shader = new Shader([
					{type: gl.VERTEX_SHADER, path: vsPath},
					{type: gl.FRAGMENT_SHADER, path: fsPath},
				]);
				shader.watch();

				let material = new GLMaterial();
				material.shader = shader;
				
				this.components.push(material);
			}
		}

	}
}

PointCloudOctree.prototype.computeVisibleHierarchyData = function(visibleNodes){

	

	let data = new Uint8Array(visibleNodes.length * 4);
	let visibleNodeTextureOffsets = new Map();
	let nodes = visibleNodes.slice();

	// sort by level and index, e.g. r, r0, r3, r4, r01, r07, r30, ...
	let sort = function (a, b) {
		let na = a.name;
		let nb = b.name;
		if (na.length !== nb.length) return na.length - nb.length;
		if (na < nb) return -1;
		if (na > nb) return 1;
		return 0;
	};
	nodes.sort(sort);


	let nodeMap = new Map();
	let offsetsToChild = new Array(nodes.length).fill(Infinity);

	for(let i = 0; i < nodes.length; i++){
		let node = nodes[i];

		nodeMap.set(node.name, node);
		visibleNodeTextureOffsets.set(node, i);

		if(i > 0){
			let index = parseInt(node.name.slice(-1));
			let parentName = node.name.slice(0, -1);
			let parent = nodeMap.get(parentName);
			let parentOffset = visibleNodeTextureOffsets.get(parent);

			let parentOffsetToChild = (i - parentOffset);

			offsetsToChild[parentOffset] = Math.min(offsetsToChild[parentOffset], parentOffsetToChild);

			data[parentOffset * 4 + 0] = data[parentOffset * 4 + 0] | (1 << index);
			data[parentOffset * 4 + 1] = (offsetsToChild[parentOffset] >> 8);
			data[parentOffset * 4 + 2] = (offsetsToChild[parentOffset] % 256);
		}

		data[i * 4 + 3] = node.name.length - 1;
	}

	return {
		data: data,
		offsets: visibleNodeTextureOffsets
	};

};

PointCloudOctree.prototype.update = function(){

	//return;

	//if(!LOD_UPDATES_ENABLED || !USER_STUDY_RENDER_OCTREE){
	//	return;
	//}

	let start = now();

	let priorityQueue = new BinaryHeap(function (x) { return 1 / x.priority; })

	priorityQueue.push({node: this.root, priority: Number.MAX_VALUE});

	let numVisibleNodes = 0;
	let numVisiblePoints = 0;
	let loadQueue = [];
	let nodeBudget = 5000;
	let pointBudget;
	let u = (USER_STUDY_LOD_MODIFIER + 1) / 2;
	pointBudget = parseInt((1 - u) * POINT_BUDGET_RANGE[0] + u * POINT_BUDGET_RANGE[1]);

	setDebugValue("point budget", pointBudget);

	let visibleNodes = [];

	let frustum = camera.getFrustum();
	
	if(vr.isActive()){

		let world = camera.world.clone();
		let trans = new Matrix4().makeTranslation(0, 0, 0);
		//let trans = new Matrix4().makeTranslation(0.1, +0.1, -0.1);

		world = world.multiply(trans);

		for(let plane of frustum){
			plane.applyMatrix4(world);
		}
	}else{
		let world = camera.world;

		for(let plane of frustum){
			plane.applyMatrix4(world);
		}
	}

	let computePriority = (node, camera) => {
		let campos = camera.position;
		let camdir = camera.getDirectionWorld();
		let box = node.boundingBox;

		let boxcenter = box.getCenter().applyMatrix4(this.transform);

		let camToBox = new Vector3().subVectors(boxcenter, campos);
		let camToBoxDir = camToBox.clone().normalize();
		let acosb = camdir.dot(camToBoxDir);
		let angle = Math.abs(Math.acos(acosb));

		let fov = Math.PI * camera.fov / 180;
		let slope = Math.tan(fov / 2);
		let distance = boxcenter.distanceTo(campos);
		let radius = box.getSize().length() / 2;

		let screenHeight = window.height;
		let projectedSize = (screenHeight / 2) * (radius / (slope * distance));



		
		//if(distance < radius){
		//	return 1000000000.0;
		//}

		let priority = 0;

		if(projectedSize < 300){
			priority = projectedSize * Math.pow((Math.PI - angle), 0.9);
			//priority = projectedSize * Math.pow((Math.PI - angle), 1);
			//priority = projectedSize * Math.pow((Math.PI - angle), 1.5);
		}else{
			priority = projectedSize
		}

		//let u = Math.pow((Math.PI - angle), 5.5);
		//u = Math.max(u, 0);
		//u = Math.min(u, 2);
		//priority = projectedSize * u;
		//let priority = projectedSize;



		//return radius;
		return priority;

	};

	let isInFrustum = (pointcloud, node, frustum) => {

		let center = node.boundingBox.getCenter();
		let max = node.boundingBox.max.clone();
		let world = pointcloud.transform;

		let centerWorld = center.applyMatrix4(world);
		let maxWorld = max.applyMatrix4(world);
		let radiusWorld = centerWorld.distanceTo(maxWorld);

		for(let plane of frustum){
			let distanceToSphereCenter = plane.normal.dot(centerWorld) + plane.distance;
			let isOutside = distanceToSphereCenter < -radiusWorld;

			if(isOutside){
				return false;
			}

		}

		return true;
	};

	while(priorityQueue.size() > 0){
		let element = priorityQueue.pop();
		let node = element.node;

		// TODO
		let visible = true;
		let numPoints = (node.buffer === null) ? 10000 : node.buffer.count;
		visible = visible && (numVisiblePoints + numPoints < pointBudget);
		visible = visible && (numVisibleNodes < nodeBudget);
		visible = visible && isInFrustum(this, node, frustum);

		if(numVisiblePoints + numPoints > pointBudget){
			break;
		}

		if(!visible){
			continue;
		}else if(visible && node.buffer === null){
			loadQueue.push(node);
			numVisibleNodes++;
			numVisiblePoints += 10000;

			continue;
		}else{
			numVisibleNodes++;
			numVisiblePoints += node.buffer.count;
		}

		visibleNodes.push(node);

		//if(node.name.startsWith("r406242")){
		//	log(node.children);
		//}

		for(let child of node.children){

			if(child !== null){
				let priority = 1;
				priority = computePriority(child, camera);

				priorityQueue.push({node: child, priority: priority});
			}
		}

	}

	this.visibleNodes = visibleNodes;

	//log(this.visibleNodes.length);
	//log(this.visibleNodes.map( n => `"${n.name}"` ).join(", "));

	for(let i = 0; i < Math.min(loadQueue.length, 1); i++){
		let node = loadQueue[i];
		this.load(node);
	}

	let duration = now() - start;
	let durationMS = (duration * 1000).toFixed(3);
	setDebugValue("duration.cp.updateOctree", `${durationMS}ms`);


}

PointCloudOctree.prototype.load = async function(node){

	if(node.buffer !== null){
		return;
	}else if(node.isLoading){
		return;
	}else{
		node.isLoading = true;
	}
	//log(`load ${node.name}`);

	let {numPoints, vertices, additionalHierarchy} = await this.loader.load(node.name);

	//log(`loaded data ${node.name}`);

	// "r406242"
	//if(node.name.startsWith("r40624")){
	//	log(node.name);
	//	log(additionalHierarchy.constructor.name);
	//	
	//	additionalHierarchy.traverse( node => {

	//		log(node.name);

	//		return true;
	//	});
	//}

	if(additionalHierarchy){

		let targets = new Map();
		targets.set(node.name, node);

		let stack = [additionalHierarchy];
		while(stack.length > 0){
			let source = stack.pop();

			let target = targets.get(source.name);

			for(let i = 0; i < 8; i++){
				let sourceChild = source.children[i];

				if(sourceChild !== null){
					let {name, boundingBox} = sourceChild;
					let targetChild = new PointCloudOctreeNode(name, boundingBox);
					target.children[i] = targetChild;
					targets.set(targetChild.name, targetChild);
					stack.push(sourceChild);
				}

			}

		}

		//node.traverse( (node) => {
		//	log(node.name);	
		//});

	}


	let attributes = [
		new GLBufferAttribute("position", 0, 3, gl.FLOAT, gl.FALSE, 12, 0),
		new GLBufferAttribute("color", 1, 4, gl.UNSIGNED_BYTE, gl.TRUE, 4, 12),
		//new GLBufferAttribute("random", 2, 1, gl.FLOAT, gl.FALSE, 4, 16),
		//new GLBufferAttribute("whatever", 3, 1, gl.INT, gl.FALSE, 4, 20),
	];

	let buffer = new GLBuffer();

	if(false){ 
		// CONSECUTIVE

		let position = new Float32Array(3 * numPoints);
		let color = new Uint8Array(4 * numPoints);
		let random = new Float32Array(numPoints);
		let whatever = new Int32Array(numPoints);

		verticesf32 = new Float32Array(vertices);
		verticesu8 = new Uint8Array(vertices);

		for(let j = 0; j < numPoints; j++){

			//let i = order[j];
			let i = j;

			position[3 * j + 0] = verticesf32[4 * i + 0];
			position[3 * j + 1] = verticesf32[4 * i + 1];
			position[3 * j + 2] = verticesf32[4 * i + 2];

			color[4 * j + 0] = verticesu8[16 * i + 12];
			color[4 * j + 1] = verticesu8[16 * i + 13];
			color[4 * j + 2] = verticesu8[16 * i + 14];
			color[4 * j + 3] = verticesu8[16 * i + 15];

			random[j] = Math.random();

			whatever[j] = 0;
		}

		buffer.setConsecutive([position, color, random, whatever], attributes, numPoints);
	}else{
		// INTERLEAVED

		buffer.setInterleaved(vertices, attributes, numPoints);
	}






	node.buffer = buffer;
	node.isLoading = false;
	//node.justLoaded = true;

	//log(`loaded ${node.name}, frame: $${frameCount}`);

	return;
};

"PointCloudOctree.js"