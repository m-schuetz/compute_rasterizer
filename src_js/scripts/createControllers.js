

{

	


	let obj = OBJLoader.load("../../resources/models/steamvr/vr_controller_vive_1_5/vr_controller_vive_1_5.obj");

	let {position, uv} = obj;

	let attributes = [
		new GLBufferAttribute("position", 0, 3, gl.FLOAT, gl.FALSE, 12, 0),
		new GLBufferAttribute("uv", 2, 2, gl.FLOAT, gl.FALSE, 8, 12),
	];

	//let image = loadImage("../../resources/models/steamvr/vr_controller_vive_1_5/onepointfive_texture.png");
	//let imgLeft = loadImage("../../resources/models/steamvr/vr_controller_vive_1_5/onepointfive_texture.png");

	//let texture = new GLTexture(image.width, image.height, image.data);
	

	if(!$("vr.controller.left")){
		let vsPath = "../../resources/shaders/mesh.vs";
		let fsPath = "../../resources/shaders/mesh.fs";
		
		let shader = new Shader([
			{type: gl.VERTEX_SHADER, path: vsPath},
			{type: gl.FRAGMENT_SHADER, path: fsPath},
		]);
		shader.watch();

		{ // left
			let material = new GLMaterial();

			let imgLeft = loadImage("../../resources/models/steamvr/vr_controller_vive_1_5/left.png");
			let texLeft = new GLTexture(imgLeft.width, imgLeft.height, imgLeft.data);

			material.shader = shader;
			material.texture = texLeft;

			let snLeft = new SceneNode("vr.controller.left");

			let buffer = new GLBuffer();

			buffer.setConsecutive([position, uv], attributes, position.length / 3);

			material.glDrawMode = gl.TRIANGLES;
			material.depthTest = true;
			material.depthWrite = true;

			snLeft.components.push(material, buffer);

			scene.root.add(snLeft);
		}

		{ // right
			let material = new GLMaterial();

			let imgRight = loadImage("../../resources/models/steamvr/vr_controller_vive_1_5/right.png");
			let texRight = new GLTexture(imgRight.width, imgRight.height, imgRight.data);
			material.shader = shader;
			material.texture = texRight;

			let snRight = new SceneNode("vr.controller.right");

			let buffer = new GLBuffer();

			buffer.setConsecutive([position, uv], attributes, position.length / 3);

			material.glDrawMode = gl.TRIANGLES;
			material.depthTest = true;
			material.depthWrite = true;

			snRight.components.push(material, buffer);

			scene.root.add(snRight);
		}

		
	}

	let snLeft = $("vr.controller.left");
	let snRight = $("vr.controller.right");
	

	let s = 10;

	snRight.transform.elements.set([
		s, 0, 0, 0, 
		0, s, 0, 0, 
		0, 0, s, 0, 
		0, 0, 0, 1, 
	]);
	

}



let imgRightSelA = loadImage("../../resources/models/steamvr/vr_controller_vive_1_5/right_selected_a.png");
let imgRightSelB = loadImage("../../resources/models/steamvr/vr_controller_vive_1_5/right_selected_b.png");
let imgRightSelC = loadImage("../../resources/models/steamvr/vr_controller_vive_1_5/right_selected_c.png");

let texRightSelA = new GLTexture(imgRightSelA.width, imgRightSelA.height, imgRightSelA.data);
let texRightSelB = new GLTexture(imgRightSelB.width, imgRightSelB.height, imgRightSelB.data);
let texRightSelC = new GLTexture(imgRightSelC.width, imgRightSelC.height, imgRightSelC.data);