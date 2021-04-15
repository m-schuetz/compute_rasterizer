
{

	let position = new Float32Array([
		// back
		-1, -1, -1, 
		+1, -1, -1, 
		+1, +1, -1, 

		+1, +1, -1, 
		-1, +1, -1, 
		-1, -1, -1,

		// front
		-1, -1, +1, 
		+1, -1, +1, 
		+1, +1, +1, 

		+1, +1, +1, 
		-1, +1, +1, 
		-1, -1, +1,

		// bottom
		-1, -1, -1,
		-1, -1, +1,
		+1, -1, +1,

		+1, -1, +1,
		+1, -1, -1,
		-1, -1, -1,

		// top
		-1, +1, -1,
		-1, +1, +1,
		+1, +1, +1,

		+1, +1, +1,
		+1, +1, -1,
		-1, +1, -1,

		// right
		+1, -1, -1,
		+1, -1, +1,
		+1, +1, +1,

		+1, +1, +1,
		+1, +1, -1,
		+1, -1, -1,

		// left
		-1, -1, -1,
		-1, -1, +1,
		-1, +1, +1,

		-1, +1, +1,
		-1, +1, -1,
		-1, -1, -1,

	]);

	let attributes = [
		new GLBufferAttribute("position", 0, 3, gl.FLOAT, gl.FALSE, 12, 0),
	];

	let texture;
	{
		let handle = gl.createTexture();
		let type = gl.TEXTURE_CUBE_MAP;

		gl.bindTexture(type, handle);

		gl.texParameteri(type, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_NEAREST);
		gl.texParameteri(type, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
		gl.texParameteri(type, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
		gl.texParameteri(type, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
		gl.texParameteri(type, gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE);

		let level = 0;
		let border = 0;

		let tiles = [
			{file: "../../resources/images/skybox/nx.jpg", target: gl.TEXTURE_CUBE_MAP_NEGATIVE_X},
			{file: "../../resources/images/skybox/ny.jpg", target: gl.TEXTURE_CUBE_MAP_NEGATIVE_Y},
			{file: "../../resources/images/skybox/nz.jpg", target: gl.TEXTURE_CUBE_MAP_NEGATIVE_Z},
			{file: "../../resources/images/skybox/px.jpg", target: gl.TEXTURE_CUBE_MAP_POSITIVE_X},
			{file: "../../resources/images/skybox/py.jpg", target: gl.TEXTURE_CUBE_MAP_POSITIVE_Y},
			{file: "../../resources/images/skybox/pz.jpg", target: gl.TEXTURE_CUBE_MAP_POSITIVE_Z},
		];

		for(let tile of tiles){
			let image = loadImage(tile.file);

			let [width, height] = [image.width, image.height];

			gl.texImage2D(tile.target, 
				level, gl.RGBA, width, height, border, gl.RGBA, gl.UNSIGNED_BYTE, image.data
			);
		}

		gl.generateMipmap(type);

		gl.bindTexture(type, 0);

		texture = {
			type: gl.TEXTURE_CUBE_MAP,
			handle: handle
		};
	}

	if(!$("skybox")){
		let vsPath = "../../resources/shaders/cubemap.vs";
		let fsPath = "../../resources/shaders/cubemap.fs";

		let material = new GLMaterial();

		let shader = new Shader([
			{type: gl.VERTEX_SHADER, path: vsPath},
			{type: gl.FRAGMENT_SHADER, path: fsPath},
		]);
		shader.watch();

		material.shader = shader;
		material.texture = texture;

		let sceneNode = new SceneNode("skybox");

		let buffer = new GLBuffer();

		sceneNode.components.push(material, buffer);
		scene.root.add(sceneNode);
	}	

	let node = $("skybox");
	let buffer = node.getComponent(GLBuffer);
	let material = node.getComponent(GLMaterial);

	let s = 100000;
	node.transform.elements.set([
		s, 0, 0, 0, 
		0, s, 0, 0, 
		0, 0, s, 0, 
		0, 0, 0, 1, 
	]);
	
	material.texture = texture;

	buffer.setConsecutive([position], attributes, position.length / 3);

	material.glDrawMode = gl.TRIANGLES;
	material.depthTest = true;
	material.depthWrite = false;

	node.update = () => {

		SceneNode.prototype.update.call(node);

		let campos = new Vector3(0, 0, 0).applyMatrix4(camera.transform);

		let s = 10;
		node.world.elements.set([
			s, 0, 0, 0, 
			0, s, 0, 0, 
			0, 0, s, 0, 
			campos.x, campos.y, campos.z, 1, 
		]);

	};


}


"createSkybox.js"