
{

	let obj = OBJLoader.load("../../resources/models/blub/blub_triangulated.obj");
	let image = loadImage("../../resources/models/blub/blub_texture.png");

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

	let material = new GLMaterial();

	let shader = new Shader([
		{type: gl.VERTEX_SHADER, path: vsPath},
		{type: gl.FRAGMENT_SHADER, path: fsPath},
	]);
	shader.watch();

	material.shader = shader;
	material.texture = texture;
	material.glDrawMode = gl.TRIANGLES;

	let attributes = [
		new GLBufferAttribute("position", 0, 3, gl.FLOAT, gl.FALSE, 12, 0),
		new GLBufferAttribute("normal", 1, 3, gl.FLOAT, gl.FALSE, 12, 12),
		new GLBufferAttribute("uv", 2, 2, gl.FLOAT, gl.FALSE, 8, 24),
	];


	//log(normal.subarray(0, 10));

	let buffer = new GLBuffer();
	buffer.setConsecutive([position, normal, uv], attributes, position.length / 3);

	let n = 10;
	for(let i = 0; i < n; i++){

		let id = `blub_${i}`;
		let node = $(id);

		if(!node){
			//node = new SceneNode(id);
			//node.components.push(material, buffer);

			node = new MeshNode(id, buffer, material);
			scene.root.add(node);
		}

		let u = i / n + 0.05;
		let r = 6;

		let x = r * Math.cos(2 * Math.PI * u);
		let y = 0.2;
		let z = r * Math.sin(2 * Math.PI * u);

		let s = 0.5;
		node.position.set(x, y, z);
		node.scale.set(s, s, s);

		let dir = new Vector3(x, y, z).normalize();
		let target = node.position.clone().add(dir);

		node.updateMatrixWorld();
		node.lookAt(target);
		node.updateMatrixWorld();
	}



}

