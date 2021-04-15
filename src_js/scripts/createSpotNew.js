
// { // create objects

// 	let objPath = `${resourceDir}/models/spot/spot_triangulated.obj`;
// 	let imgPath = `${resourceDir}/models/spot/spot_texture.png`;

// 	let glbuffer = ObjectLoader.loadBuffer(objPath);
// 	let image = loadImage("../../resources/models/spot/spot_texture.png");

// 	let n = 10;
// 	for(let i = 0; i < n; i++){
// 		let id = `spot_${i}`;

// 		let node = new MeshNode(id, glbuffer, material);

// 		let u = i / n;
// 		let radius = 6;

// 		let x = radius * Math.cos(2 * Math.PI * u);
// 		let y = 0.8;
// 		let z = radius * Math.sin(2 * Math.PI * u);

// 		let s = 0.7;
// 		node.position.set(x, y, z);
// 		node.scale.set(s, s, s);

// 		node.updateMatrixWorld();
// 		node.lookAt(0, 4, 0);
// 		node.updateMatrixWorld();
// 	}


// }

// {
// 	let nodes = [
// 		$(`spot_1`),
// 	];

// 	for(let node of nodes){

// 	}

// }