
// GROUND
{

	let position = new Float32Array([
		// bottom
		-1, 0, -1,
		-1, 0, +1,
		+1, 0, +1,

		+1, 0, +1,
		+1, 0, -1,
		-1, 0, -1,
	]);

	let normal = new Float32Array([
		0, 1, 0,
		0, 1, 0,
		0, 1, 0,

		0, 1, 0,
		0, 1, 0,
		0, 1, 0,
	]);

	let uv = new Float32Array([
		0, 0,
		0, 10,
		10, 10,

		10, 10,
		10, 0,
		0, 0,
	]);

	

	let attributes = [
		new GLBufferAttribute("position", 0, 3, gl.FLOAT, gl.FALSE, 12, 0),
		new GLBufferAttribute("normal", 1, 3, gl.FLOAT, gl.FALSE, 12, 12),
		new GLBufferAttribute("uv", 2, 2, gl.FLOAT, gl.FALSE, 8, 24),
	];

	let image = loadImage("../../resources/images/rectangle.png");
	let data = new Uint8Array(image.data);
	for(let i = 0; i < data.length; i++){

		if( (i % 4) !== 3){
			data[i] = data[i] * 0.2;
		}
	}
	let texture = new GLTexture(image.width, image.height, image.data);
	

	if(!$("origin_ground")){
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

		//let sceneNode = new SceneNode("origin_ground");
		let buffer = new GLBuffer();
		buffer.setConsecutive([position, normal, uv], attributes, position.length / 3);
		let sceneNode = new MeshNode("origin_ground", buffer, material);


		//sceneNode.components.push(material, buffer);
		scene.root.add(sceneNode);
	}	

	let node = $("origin_ground");
	let buffer = node.getComponent(GLBuffer);
	let material = node.getComponent(GLMaterial);

	let s = 10;
	node.transform.elements.set([
		s, 0, 0, 0, 
		0, s, 0, 0, 
		0, 0, s, 0, 
		0, 0, 0, 1, 
	]);
	node.world = node.transform;
	
	material.texture = texture;
	

	buffer.setConsecutive([position, normal, uv], attributes, position.length / 3);

	material.glDrawMode = gl.TRIANGLES;
	material.depthTest = true;
	material.depthWrite = true;


}