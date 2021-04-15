

createImageSpaceQuad = function(){

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

	let vsPath = "../../resources/shaders/imageSpaceQuad.vs";
	let fsPath = "../../resources/shaders/imageSpaceQuad.fs";

	let shader = new Shader([
		{type: gl.VERTEX_SHADER, path: vsPath},
		{type: gl.FRAGMENT_SHADER, path: fsPath},
	]);
	shader.watch();

	let material = new GLMaterial();
	material.glDrawMode = gl.TRIANGLES;
	material.shader = shader;

	let sceneNode = new SceneNode("image_space_quad");
	sceneNode.components.push(material, buffer);
	sceneNode.visible = false;

	scene.root.add(sceneNode);
}

createSpot = function(name){

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

	let buffer = new GLBuffer();
	buffer.setConsecutive([position, normal, uv], attributes, position.length / 3);


	let node = $(name);

	if(!node){
		node = new MeshNode(name, buffer, material);
		//scene.root.add(node);
	}


	//let s = 0.7;
	//node.position.set(x, y, z);
	//node.scale.set(s, s, s);

	//node.updateMatrixWorld();
	//node.lookAt(0, 4, 0);
	//node.updateMatrixWorld();

	return node;

}



loadOctreeHeidentor = function(){
	
	//let heidentor = new PointCloudOctree("heidentor", "C:/dev/pointclouds/heidentor_converted/cloud.js");
	let heidentor = new PointCloudOctree("heidentor_oct", "C:/dev/pointclouds/heidentor_averaged/cloud.js");
	//let heidentor = new PointCloudOctree("heidentor_oct", "C:/dev/pointclouds/heidentor_converted/cloud.js");
	//let heidentor = new PointCloudOctree("heidentor", "D:/dev/pointclouds/archpro/heidentor/cloud.js");
	heidentor.transform.elements.set([
		1, 0, 0, 0, 
		0, 0, -1, 0, 
		0, 1, 0, 0, 
		0, 0, 0, 1, 
	]);
	scene.root.add(heidentor);

	view.set(
		[-7.330047391982175, 7.897543503270976, -4.463023058403868],
		[4.437738969389951, 4.55472018779445, -7.284232739227429]
	);
}	


loadOctreeCA13 = function(){
	let ca13 = new PointCloudOctree("ca13", "C:/dev/pointclouds/converted/CA13_morro_area/cloud.js");
	ca13.transform.elements.set([
		1, 0, 0, 0, 
		0, 0, -1, 0, 
		0, 1, 0, 0, 
		0, 0, 0, 1, 
	]);
	scene.root.add(ca13);

	view.position.set(3507.3145044342887,458.50801523952845,-687.702439499016);
	view.lookAt(3275.280062400423,29.129026788555823,-1413.3102282310417);
}

loadOctreeRetz = function(){
	let pc = new PointCloudOctree("retz", "C:/dev/pointclouds/riegl/Retz_Airborne_Terrestrial_Combined_1cm.laz_converted/cloud.js");
	pc.transform.elements.set([
		1, 0, 0, 0, 
		0, 0, -1, 0, 
		0, 1, 0, 0, 
		0, 0, 0, 1, 
	]);
	scene.root.add(pc);

	view.position.set(628.3218112125325,55.698254179368,-812.1339171990866);
	view.lookAt(586.3603996295701,33.40059995387169,-858.3032402371352);
	
}


loadProgressiveHeidentor = function(){
	let pc = new PointCloudProgressive("heidentor", "C:/dev/pointclouds/heidentor.las");
	//let pc = new PointCloudProgressive("lion", "C:/dev/pointclouds/lion.las");

	let s = 0.6;
	pc.transform.elements.set([
		s, 0, 0, 0, 
			0, 0, s, 0, 
			0, s, 0, 0, 
			0, 0, 1, 1, 
	]);

	scene.root.add(pc);

	view.set(
		[-11.542882308346643, 7.178273756296709, 5.3013466701128715],
		[-0.7992052516487097, 3.523237342807562, 1.849253974346153]
	);
}

loadBasicHeidentor = function(){
	let pc = new PointCloudBasic("heidentor", "C:/dev/pointclouds/heidentor.las");

	let s = 3.6;
	pc.world.elements.set([
		s, 0, 0, 0, 
			0, 0, s, 0, 
			0, s, 0, 0, 
			0, 0, 1, 1, 
	]);

	scene.root.add(pc);

	view.set(
		[-11.542882308346643, 7.178273756296709, 5.3013466701128715],
		[-0.7992052516487097, 3.523237342807562, 1.849253974346153]
	);
}

loadProgressiveLion = function(){
	let pc = new PointCloudProgressive("lion", "C:/dev/pointclouds/lion.las");

	let s = 0.6;
	pc.transform.elements.set([
		s, 0, 0, 0, 
			0, 0, s, 0, 
			0, s, 0, 0, 
			0, 0, 1, 1, 
	]);
	//pc.transform.elements.set([
	//	1, 0, 0, 0, 
	//	0, 0, -1, 0, 
	//	0, 1, 0, 0, 
	//	0, 0, 0, 1, 
	//]);
	scene.root.add(pc);

	view.set(
		[3.21100208995429, 2.658987782300026, -0.2640089331301896],
		[0.4097924229397498, 0.8982746832937245, 2.6127698880080517]
	);
}


loadExpRetz = function(){
	//let pc = new PointCloudExp("lion", "C:/dev/pointclouds/lion.las");
	//let pc = new PointCloudExp("lion", "C:/dev/pointclouds/heidentor_merged_oc.bin");
	let pc = new PointCloudExp("retz", "C:/dev/pointclouds/retz_averaged.bin");

	pc.octreeSize = 1630;
	pc.spacing = 14.112011909484864;

	let s = 0.01;
	pc.transform.elements.set([
		s, 0, 0, 0, 
		0, 0, s, 0, 
		0, s, 0, 0, 
		0, 0, 1, 1, 
	]);

	scene.root.add(pc);

	view.set(
		[3.21100208995429, 2.658987782300026, -0.2640089331301896],
		[0.4097924229397498, 0.8982746832937245, 2.6127698880080517]
	);
}



loadExpAffandi_1_74_to_76 = function(){
	//let pc = new PointCloudExp("lion", "C:/dev/pointclouds/lion.las");
	//let pc = new PointCloudExp("lion", "C:/dev/pointclouds/heidentor_merged_oc.bin");
	let pc = new PointCloudExp("affandi_1_74_to_76", "C:/dev/pointclouds/affandi_1_74_to_76_averaged.bin");

	pc.octreeSize = 550;
	pc.spacing = 4.7432732582092289;

	let s = 0.01;
	pc.transform.elements.set([
		s, 0, 0, 0, 
		0, 0, s, 0, 
		0, s, 0, 0, 
		0, 0, 1, 1, 
	]);

	scene.root.add(pc);

	view.set(
		[3.21100208995429, 2.658987782300026, -0.2640089331301896],
		[0.4097924229397498, 0.8982746832937245, 2.6127698880080517]
	);
}

loadExpAffandi_6_02_to_04 = function(){
	//let pc = new PointCloudExp("lion", "C:/dev/pointclouds/lion.las");
	//let pc = new PointCloudExp("lion", "C:/dev/pointclouds/heidentor_merged_oc.bin");
	let pc = new PointCloudExp("affandi_6_02_to_04", "C:/dev/pointclouds/affandi_6_02_to_04_averaged.bin");

	pc.octreeSize = 60;
	pc.spacing = 0.5279831886291504;

	let s = 0.01;
	pc.transform.elements.set([
		s, 0, 0, 0, 
		0, 0, s, 0, 
		0, s, 0, 0, 
		0, 0, 1, 1, 
	]);

	scene.root.add(pc);

	view.set(
		[3.21100208995429, 2.658987782300026, -0.2640089331301896],
		[0.4097924229397498, 0.8982746832937245, 2.6127698880080517]
	);
}

loadExpHeidentor = function(){
	let pc = new PointCloudExp("heidentor", "C:/dev/pointclouds/heidentor_merged_oc_mm_out.bin");

	pc.octreeSize = 16;
	pc.spacing = 0.1347;
	pc.pointSize = 1;
	pc.minMilimeters = 2;

	let s = 0.6;
	pc.world.elements.set([
		s, 0, 0, 0, 
		0, 0, s, 0, 
		0, s, 0, 0, 
		0, 0, 1, 1, 
	]);

	scene.root.add(pc);

	view.set(
		[3.21100208995429, 2.658987782300026, -0.2640089331301896],
		[0.4097924229397498, 0.8982746832937245, 2.6127698880080517]
	);
}

loadExpTUP = function(){
	let pc = new PointCloudExp("tup", "D:/dev/pointclouds/tu_project/merged2.bin");

	pc.octreeSize = 370;
	pc.spacing = 3.1852848529815676;
	pc.pointSize = 1;
	pc.minMilimeters = 1;

	let s = 0.6;
	pc.world.elements.set([
		s, 0, 0, 0, 
		0, 0, s, 0, 
		0, s, 0, 0, 
		0, 0, 1, 1, 
	]);

	scene.root.add(pc);

	view.set(
		[3.21100208995429, 2.658987782300026, -0.2640089331301896],
		[0.4097924229397498, 0.8982746832937245, 2.6127698880080517]
	);
}

loadExpEndeavor = function(){
	let node = new PointCloudExp("endeavor_clod", "C:/dev/pointclouds/endeavor.bin");

	node.octreeSize = 100;
	node.spacing = 0.1347;

	let s = 0.2;
	node.world.elements.set([
		s, 0, 0, 0, 
		0, 0, -s, 0, 
		0, s, 0, 0, 
		s * 0.18, s * 5.07, s * 0, 1, 
	]);

	node.octreeSize = 190;
	node.spacing = 1.6429535150527955;
	node.minMilimeters = 1.5;
	node.pointSize = 1.0;

	scene.root.add(node);

	view.set(
		[3.21100208995429, 2.658987782300026, -0.2640089331301896],
		[0.4097924229397498, 0.8982746832937245, 2.6127698880080517]
	);
}

loadOctreeEndeavor = function(){
	
	let node = new PointCloudOctree("endeavor_oct", "D:/dev/pointclouds/NVIDIA/laserscans/sel_averaged/cloud.js");
	
	let s = 0.2;
	node.world.elements.set([
		s, 0, 0, 0, 
		0, 0, -s, 0, 
		0, s, 0, 0, 
		s * -31.35, s * -15, s * 30.7, 1, 
	]);

	node.octreeSize = 360;
	node.spacing = node.octreeSize / 200;
	node.minMilimeters = 1.5;
	node.pointSize = 1;

	scene.root.add(node);


	{
		let node = createSpot("spot_endeavor");

		let s = 0.04;

		node.position.set(-1, 1.17, 0.1);
		node.scale.set(s, s, s);

		node.updateMatrixWorld();

		scene.root.add(node);
	}

	{
		let node = createSpot("spot_endeavor_2");

		let s = 0.03;

		node.position.set(0.44, 0.57, -0.7);
		node.scale.set(s, s, s);

		node.updateMatrixWorld();
		scene.root.add(node);
	}
}

loadExpMatterhorn120 = function(){
	let node = new PointCloudExp("matterhorn120_clod", "D:/dev/pointclouds/matterhorn.bin");

	node.octreeSize = 100;
	node.spacing = 0.1347;

	let s = 0.0005;
	node.world.elements.set([
		-s, 0, 0, 0, 
		0, 0, s, 0, 
		0, s, 0, 0, 
		-s * 660.08, s * 3036, s * -2957.38, 1, 
	]);

	node.octreeSize = 1630;
	node.spacing = 60.371979713439942;
	node.minMilimeters = 4.06;
	node.pointSize = 1.3;

	scene.root.add(node);

	view.set(
		[3.21100208995429, 2.658987782300026, -0.2640089331301896],
		[0.4097924229397498, 0.8982746832937245, 2.6127698880080517]
	);
}

loadOctreeMatterhorn120 = function(){
	
	let node = new PointCloudOctree("matterhorn120_oct", "D:/dev/pointclouds/matterhorn_averaged/cloud.js");

	scene.root.add(node);

	let s = 0.0005;
	node.world.elements.set([
		-s, 0, 0, 0, 
		0, 0, s, 0, 
		0, s, 0, 0, 
		s * 2500, -s * 1500, -s * 6000, 1, 
	]);

	node.octreeSize = 5700;
	node.spacing = 65;
	node.minMilimeters = 2;
	node.pointSize = 1.2;
	
	{
		let node = $("spot_matterhorn");

		let s = 0.02;

		node.position.set(-0.75, 1.5, -1.75);
		node.scale.set(s, s, s);

		node.updateMatrixWorld();
		scene.root.add(node);
	}
}


loadExpEclepens = function(){
	//let pc = new PointCloudExp("lion", "C:/dev/pointclouds/lion.las");
	//let pc = new PointCloudExp("lion", "C:/dev/pointclouds/heidentor_merged_oc.bin");
	let pc = new PointCloudExp("eclepens", "C:/dev/pointclouds/eclepens_averaged.bin");

	pc.octreeSize = 1070;
	pc.spacing = 9.371979713439942;
	pc.minMilimeters = 4.06;

	let s = 0.003;
	pc.transform.elements.set([
		-s, 0, 0, 0, 
		0, 0, -s, 0, 
		0, s, 0, 0, 
		0.51, 0.99, 0.5, 1, 
	]);

	scene.root.add(pc);

	view.set(
		[3.21100208995429, 2.658987782300026, -0.2640089331301896],
		[0.4097924229397498, 0.8982746832937245, 2.6127698880080517]
	);
}

loadExpMatterhorn = function(){
	//let pc = new PointCloudExp("lion", "C:/dev/pointclouds/lion.las");
	//let pc = new PointCloudExp("lion", "C:/dev/pointclouds/heidentor_merged_oc.bin");
	//let pc = new PointCloudExp("matterhorn", "C:/dev/pointclouds/test.bin");
	let pc = new PointCloudExp("matterhorn", "C:/dev/pointclouds/matterhorn_averaged.bin");

	pc.octreeSize = 1630;
	pc.spacing = 60.371979713439942;
	pc.minMilimeters = 4.06;
	pc.pointSize = 1.3;

	let s = 0.003;
	pc.transform.elements.set([
		-s, 0, 0, 0, 
		0, 0, -s, 0, 
		0, s, 0, 0, 
		0.51, 0.99, 0.5, 1, 
	]);

	scene.root.add(pc);

	view.set(
		[-1.530782813643967, 2.2583109798215535, 0.6799130799883772],
		[0.03800878156636581, 1.2051155315418414, -0.8885679555907814]
	);
}

loadExpLion = function(){
	//let pc = new PointCloudExp("lion", "C:/dev/pointclouds/lion.las");
	//let pc = new PointCloudExp("lion", "C:/dev/pointclouds/heidentor_merged_oc.bin");
	let pc = new PointCloudExp("lion", "C:/dev/pointclouds/test.bin");

	let s = 0.6;
	pc.transform.elements.set([
		s, 0, 0, 0, 
		0, 0, s, 0, 
		0, s, 0, 0, 
		0, 0, 1, 1, 
	]);

	scene.root.add(pc);

	view.set(
		[3.21100208995429, 2.658987782300026, -0.2640089331301896],
		[0.4097924229397498, 0.8982746832937245, 2.6127698880080517]
	);
}

loadProgressiveCandiSari = function(){
	//let pc = new PointCloudProgressive("heidentor", "C:/dev/pointclouds/Retz_Airborne_Terrestrial_Combined_1cm.las");
	//let pc = new PointCloudProgressive("heidentor", "C:/dev/pointclouds/candi_sari_exterior.las");
	let pc = new PointCloudProgressive("candi sari", "C:/dev/pointclouds/candi_sari.las");
	//let pc = new PointCloudProgressive("heidentor", "C:/dev/pointclouds/heidentor.las");
	//let pc = new PointCloudProgressive("lion", "C:/dev/pointclouds/lion.las");
	pc.transform.elements.set([
		1, 0, 0, 0, 
		0, 0, -1, 0, 
		0, 1, 0, 0, 
		0, 0, 0, 1, 
	]);
	scene.root.add(pc);

	view.set(
		[6.025500931997474, 1.958335567729737, 100.8251832420204],
		[19.046112810740308, -10.721512781591064, 79.76248235632359]
	);
}

loadProgressiveRetz = function(){
	let pc = new PointCloudProgressive("retz_progressive", "C:/dev/pointclouds/Retz_Airborne_Terrestrial_Combined_1cm.las");
	//let pc = new PointCloudProgressive("heidentor", "C:/dev/pointclouds/heidentor.las");
	//let pc = new PointCloudProgressive("lion", "C:/dev/pointclouds/lion.las");
	pc.transform.elements.set([
		1, 0, 0, 0, 
		0, 0, -1, 0, 
		0, 1, 0, 0, 
		0, 0, 0, 1, 
	]);
	scene.root.add(pc);

	view.set(
		[608.2627542249325, 44.81856265925976, -446.82819954242206],
		[515.6532013815034, -20.49565657881057, -505.0567246324728]
	);
}

loadProgressiveEclepens = function(){
	let pc = new PointCloudProgressive("eclepens_progressive", "C:/dev/pointclouds/eclepens.las");
	//let pc = new PointCloudProgressive("heidentor", "C:/dev/pointclouds/heidentor.las");
	//let pc = new PointCloudProgressive("lion", "C:/dev/pointclouds/lion.las");
	pc.transform.elements.set([
		1, 0, 0, 0, 
		0, 0, -1, 0, 
		0, 1, 0, 0, 
		0, 0, 0, 1, 
	]);
	scene.root.add(pc);

	view.set(
		[874.6785925774539, 212.7780835491692, -69.67790692452093],
		[494.85150258814264, -51.14680708994847, -397.005395340306]
	);

	//view.set(
	//	[608.2627542249325, 44.81856265925976, -446.82819954242206],
	//	[515.6532013815034, -20.49565657881057, -505.0567246324728]
	//);
}

createMatterhornLabel = function(){

	let image = loadImage("../../resources/images/Matterhorn.png");
	//let image = loadImage("../../resources/images/starry_night.jpg");

	let texture = new GLTexture(image.width, image.height, image.data);

	let sx = 1;
	let sy = image.height / image.width;

	let vertices = new Float32Array([
		-sx, -sy, 0, 0, 0,
		 sx, -sy, 0, 1, 0,
		 sx,  sy, 0, 1, 1,

		-sx, -sy, 0, 0, 0,
		 sx,  sy, 0, 1, 1,
		-sx,  sy, 0, 0, 1,
	]);

	let attributes = [
		new GLBufferAttribute("position", 0, 3, gl.FLOAT, gl.FALSE, 12, 0),
		new GLBufferAttribute("uv", 2, 2, gl.FLOAT, gl.FALSE, 8, 12),
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
	material.texture = texture;
	material.shader = shader;

	let sceneNode = new MeshNode("matterhorn_label", buffer, material);

	let s = 0.5;
	sceneNode.transform.elements.set([
		s, 0, 0, 0, 
		0, s, 0, 0, 
		0, 0, s, 0, 
		-0.5, 2, -1.5, 1, 
	]);

	scene.root.add(sceneNode);

}

createEclepensLabel = function(){

	let image = loadImage("../../resources/images/eclepens.png");
	//let image = loadImage("../../resources/images/starry_night.jpg");

	let texture = new GLTexture(image.width, image.height, image.data);

	let sx = 1;
	let sy = image.height / image.width;

	let vertices = new Float32Array([
		-sx, -sy, 0, 0, 0,
		 sx, -sy, 0, 1, 0,
		 sx,  sy, 0, 1, 1,

		-sx, -sy, 0, 0, 0,
		 sx,  sy, 0, 1, 1,
		-sx,  sy, 0, 0, 1,
	]);

	let attributes = [
		new GLBufferAttribute("position", 0, 3, gl.FLOAT, gl.FALSE, 12, 0),
		new GLBufferAttribute("uv", 2, 2, gl.FLOAT, gl.FALSE, 8, 12),
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
	material.texture = texture;
	material.shader = shader;

	let sceneNode = new MeshNode("eclepens_label", buffer, material);

	let s = 0.5;
	sceneNode.transform.elements.set([
		-0, 0, s, 0, 
		0, s, 0, 0, 
		s, 0, 0, 0, 
		2, 1.8, 0.4, 1, 
	]);

	scene.root.add(sceneNode);

}


loadExpPlane = function(){
	let node = new PointCloudExp("planes_clod", "C:/dev/pointclouds/planes_merged.bin");


	let s = 0.1;
	node.world.elements.set([
		s, 0, 0, 0, 
		0, s, 0, 0, 
		0, 0, s, 0, 
		0, 0, 1, 1, 
	]);

	node.octreeSize = 200;
	node.spacing = node.octreeSize / 128;
	node.minMilimeters = 1.5;
	node.pointSize = 1.0;

	scene.root.add(node);

	view.set(
		[3.21100208995429, 2.658987782300026, -0.2640089331301896],
		[0.4097924229397498, 0.8982746832937245, 2.6127698880080517]
	);
}


loadOctreePlane = function(){
	
	let node = new PointCloudOctree("plane_oct", "C:/dev/pointclouds/planes_converted/cloud.js");

	scene.root.add(node);

	let s = 0.1;
	node.world.elements.set([
		s, 0, 0, 0, 
		0, s, 0, 0, 
		0, 0, s, 0, 
		0, 0, 1, 1, 
	]);

	node.octreeSize = 200;
	node.spacing = node.octreeSize / 128;
	node.minMilimeters = 1.5;
	node.pointSize = 1.0;
}


// loadExpEclepens();
// createEclepensLabel();
// loadExpMatterhorn();
// createMatterhornLabel();


// loadExpPlane();
// loadOctreePlane();


//loadExpEndeavor();
// loadOctreeEndeavor();


// loadExpMatterhorn120();
// loadOctreeMatterhorn120();


//loadExpTUP();
//loadExpHeidentor();
//loadExpLion();
//loadExpRetz();
//loadExpAffandi_1_74_to_76();
//loadExpAffandi_6_02_to_04();

//loadBasicHeidentor();
//loadProgressiveHeidentor();
//loadProgressiveLion();
//loadProgressiveCandiSari();
//loadProgressiveRetz();
//loadProgressiveEclepens();

//loadOctreeHeidentor();
//loadOctreeCA13();
//loadOctreeEndeavor();

// undefined;

"createPointCloudScene.js"