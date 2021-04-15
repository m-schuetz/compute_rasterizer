
{

	let obj = OBJLoader.load("../../resources/models/spot/spot_triangulated.obj");
	let image = loadImage("../../resources/models/spot/spot_texture.png");

	let texture = new GLTexture(image.width, image.height, image.data);
	let {position, uv} = obj;

	let normal = new Float32Array(position.length);
	for(let i = 0; i < obj.count / 3; i++){
		
		let p1 = new Vector3(
			position[9 * i + 0],
			position[9 * i + 1],
			position[9 * i + 2],
		);

		let p2 = new Vector3(
			position[9 * i + 3],
			position[9 * i + 4],
			position[9 * i + 5],
		);

		let p3 = new Vector3(
			position[9 * i + 6],
			position[9 * i + 7],
			position[9 * i + 8],
		);

		let v1 = new Vector3().subVectors(p2, p1);
		let v2 = new Vector3().subVectors(p3, p1);

		let n = v1.cross(v2).normalize();

		normal[9 * i + 0] = n.x;
		normal[9 * i + 1] = n.y;
		normal[9 * i + 2] = n.z;

		normal[9 * i + 3] = n.x;
		normal[9 * i + 4] = n.y;
		normal[9 * i + 5] = n.z;

		normal[9 * i + 6] = n.x;
		normal[9 * i + 7] = n.y;
		normal[9 * i + 8] = n.z;
		

	}

	let vsPath = "../../resources/shaders/mesh.vs";
	let fsPath = "../../resources/shaders/mesh.fs";

	let shader = new Shader([
		{type: gl.VERTEX_SHADER, path: vsPath},
		{type: gl.FRAGMENT_SHADER, path: fsPath},
	]);
	shader.watch();

	let material = new GLMaterial();

	material.shader = shader;
	material.texture = texture;
	material.glDrawMode = gl.TRIANGLES;

	let attributes = [
		new GLBufferAttribute("position", 0, 3, gl.FLOAT, gl.FALSE, 12, 0),
		new GLBufferAttribute("normal", 1, 3, gl.FLOAT, gl.FALSE, 12, 12),
		new GLBufferAttribute("uv", 2, 2, gl.FLOAT, gl.FALSE, 8, 24),
	];

	let buffer = new GLBuffer();
	buffer.setConsecutive([position, normal, uv], attributes, position.length / 3);

	let n = 10;
	for(let i = 0; i < n; i++){

		let id = `spot_${i}`;
		let node = $(id);

		if(!node){
			//node = new SceneNode(id);
			//node.components.push(material, buffer);

			node = new MeshNode(id, buffer, material);
			scene.root.add(node);
		}

		let u = i / n;
		let r = 6;

		let x = r * Math.cos(2 * Math.PI * u);
		let y = 0.8;
		let z = r * Math.sin(2 * Math.PI * u);

		let s = 0.7;
		node.position.set(x, y, z);
		node.scale.set(s, s, s);

		node.updateMatrixWorld();
		node.lookAt(0, 4, 0);
		node.updateMatrixWorld();
	}

	// let length = 10;
	// let width = 10;
	// for(let i = 0; i < length; i++){
	// 	for(let j = 0; j < width; j++){

	// 		let id = `spot_${i}_${j}`;
	// 		let node = $(id);

	// 		if(!node){
	// 			node = new MeshNode(id, buffer, material);
	// 			scene.root.add(node);
	// 		}

	// 		//let u = i / n;
	// 		//let r = 6;

	// 		//let x = r * Math.cos(2 * Math.PI * u);
	// 		//let y = 0.8;
	// 		//let z = r * Math.sin(2 * Math.PI * u);

	// 		//let s = 0.7;
	// 		//node.position.set(x, y, z);
	// 		//node.scale.set(s, s, s);

	// 		//node.updateMatrixWorld();
	// 		//node.lookAt(0, 4, 0);
	// 		//node.updateMatrixWorld();
	// 	}
	// }



}

