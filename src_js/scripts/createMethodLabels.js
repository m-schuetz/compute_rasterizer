

createMethodLabel = function(name){
	let aspect = 16 / 9;

	let vertices = new Float32Array([
		-1, -1, 0, 0, 0,
		 1, -1, 0, 1, 0,
		 1,  1, 0, 1, 1,

		-1, -1, 0, 0, 0,
		 1,  1, 0, 1, 1,
		-1,  1, 0, 0, 1,
	]);

	let attributes = [
		new GLBufferAttribute("position", 0, 3, gl.FLOAT, gl.FALSE, 12, 0),
		new GLBufferAttribute("uv", 1, 2, gl.FLOAT, gl.FALSE, 8, 12),
	];

	let buffer = new GLBuffer();
	buffer.set(vertices, attributes, 6);

	let vsPath = "../../resources/shaders/mesh.vs";
	let fsPath = "../../resources/shaders/mesh.fs";

	let shader = new Shader([
		{type: gl.VERTEX_SHADER, path: vsPath},
		{type: gl.FRAGMENT_SHADER, path: fsPath},
	]);
	shader.watch();

	let material = new GLMaterial();
	material.glDrawMode = gl.TRIANGLES;
	material.shader = shader;

	//let image = loadImage("../../resources/models/spot/spot_texture.png");
	//let texture = new GLTexture(image.width, image.height, image.data);
	//material.texture = texture;

	let sceneNode = new SceneNode(name);
	sceneNode.components.push(material, buffer);
	sceneNode.visible = false;

	return sceneNode;
}


{

	let node = createMethodLabel("lbl_method");

	let image = loadImage("../../resources/images/method_1.png");
	let texture = new GLTexture(image.width, image.height, image.data);
	node.getComponent(GLMaterial).texture = texture;

	scene.root.add(node);

	

}
