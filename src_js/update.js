
var updateVRControllers = function(){
	if(vr.isActive()){
		let snLeft = $("vr.controller.left");
		let snRight = $("vr.controller.right");

		//{ // update button states
		//	let stateLeft = vr.getControllerStateLeft();
		//	let stateRight = vr.getControllerStateRight();
		//}

		if(false){ // update pose
			let rightPose = vr.getRightControllerPose();
			let leftPose = vr.getLeftControllerPose();

			if(rightPose){
				snRight.transform.set(rightPose);
				snRight.world.set(rightPose);
				snRight.visible = true;
			}else{
				snRight.visible = false;
			}

			if(leftPose){
				snLeft.transform.set(leftPose);
				snLeft.world.set(leftPose);
				snLeft.visible = true;
			}else{
				snLeft.visible = false;
			}
		}

	}else{
		let snLeft = $("vr.controller.left");
		let snRight = $("vr.controller.right");

		if(snLeft) snLeft.visible = false;
		if(snRight) snRight.visible = false;
	}
};

var updateCamera = function(){

	let {near, far} = camera;

	//let [near, far] = [1000, 1000];

	//log(camera.fov);

	if(vr.isActive()){
		
		//let [near, far] = [1, 1000];
		let hmdPose = new Matrix4().set(vr.getHMDPose());
		let leftProj = new Matrix4().set(vr.getLeftProjection(near, far));
		let rightProj = new Matrix4().set(vr.getRightProjection(near, far));

		camera.position = new Vector3(0, 0, 0).applyMatrix4(hmdPose);
		camera.transform = hmdPose;
		camera.world = hmdPose;
		camera.updateProjectionMatrix();
		//camera.projectionMatrix = rightProj;
		camera.fov = 90;

		let size = vr.getRecommmendedRenderTargetSize();
		camera.size = size;
	}else{
		camera.updateMatrixWorld();
		controls.update(window.timeSinceLastFrame);

		camera.position.copy(view.position);
		camera.updateMatrixWorld();
		camera.lookAt(view.getPivot());
		camera.updateMatrixWorld();

		camera.size = {width: window.width, height: window.height};
	}

	camera.updateProjectionMatrix();

	//log(camera.projectionMatrix.elements);
};

var updateSpot = function(){

	return;
	
	let nodes = [
		$("spot_6")
	];

	for(let node of nodes){
		let t = now();
		let y = 1 + 0.5 * Math.sin(3 * t);

		node.position.y = y;
		node.updateMatrixWorld();
	}


	return;


};

// if($("test_brush")){

// 	let brushNode = $("test_brush");

// 	brushNode.buffer.count = 0;

// }

var lastDragPos = null;
var dragStartTime = null;
var updateControllerBrushing = function(){

	let snRight = $("vr.controller.right");

	if(!vr.isActive() || snRight == null || snRight.visible === false){
		return;
	}

	let posWorld = snRight.position.clone().applyMatrix4(snRight.world);
	let state = vr.getControllerStateRight();

	if(state.pressed[OVRButtonID.SteamVR_Trigger]){
		
		if(lastDragPos === null){
			lastDragPos = posWorld;
			return;
		}

		if(dragStartTime === null){
			dragStartTime = now();
		}

		let lineResolution = 0.01;
		let pointsPerStop = 50;
		let spreadRadius = 0.01;

		let distance = posWorld.distanceTo(lastDragPos);
		if(distance < lineResolution){
			return;
		}

		let brushNode = $("test_brush");

		if(!brushNode){
			return;
		}

		let stride = brushNode.buffer.stride;

		let data = new ArrayBuffer(pointsPerStop * stride);
		let view = new DataView(data);

		let t = now();
		let tSinceStart = now() -  dragStartTime;
		let px = posWorld.x;
		let py = posWorld.y;
		let pz = posWorld.z;

		//log(tSinceStart);
		//spreadRadius *= 5 * ( Math.abs(Math.sin(10 * tSinceStart )));

		

		let spectral = [
			new Vector3(158,1,66),
			new Vector3(213,62,79),
			new Vector3(244,109,67),
			new Vector3(253,174,97),
			new Vector3(254,224,139),
			new Vector3(255,255,191),
			new Vector3(230,245,152),
			new Vector3(171,221,164),
			new Vector3(102,194,165),
			new Vector3(50,136,189),
			new Vector3(94,79,162),
		];

		let spectralIndex = parseInt(Math.min((t % 1) * spectral.length, spectral.length - 1));
		// log(spectralIndex);
		let spectralColor = spectral[spectralIndex];

		spreadRadius = 0.000001;
		spreadRadius = 0.01;

		for(let i = 0; i < pointsPerStop; i++){

			let x = px + 2 * (Math.random() - 0.5) * spreadRadius;
			let y = py + 2 * (Math.random() - 0.5) * spreadRadius;
			let z = pz + 2 * (Math.random() - 0.5) * spreadRadius;

			let dp = Math.sqrt(
				(px - x) ** 2 + 
				(py - y) ** 2 + 
				(pz - z) ** 2);
			//let dn = dp / spreadRadius;

			size = 25;

			let [r, g, b, a] = [...spectralColor.toArray(), 255];
			//let [r, g, b, a] = [255, 0, 0, 255];
			//let [r, g, b, a] = [0, 255, 0, 255];


			view.setFloat32(stride * i + 0, x, true);
			view.setFloat32(stride * i + 4, y, true);
			view.setFloat32(stride * i + 8, z, true);

			view.setFloat32(stride * i + 12, px, true);
			view.setFloat32(stride * i + 16, py, true);
			view.setFloat32(stride * i + 20, pz, true);

			view.setFloat32(stride * i + 24,  r / 255.0, true);
			view.setFloat32(stride * i + 28,  g / 255.0, true);
			view.setFloat32(stride * i + 32,  b / 255.0, true);
			view.setFloat32(stride * i + 36,  a / 255.0, true);

			view.setFloat32(stride * i + 40, size, true);
			view.setFloat32(stride * i + 44, t, true);
			view.setFloat32(stride * i + 48, Math.random(), true);

		}


		brushNode.addData(data);
		lastDragPos = posWorld;

	}else{
		lastDragPos = null;
		dragStartTime = null;
	}

	
};


var tmDragStart = null;
var updateTriggerMove = function(){

	let snRight = $("vr.controller.right");

	if(!vr.isActive() || snRight == null || snRight.visible === false){
		return;
	}

	let nodes = [
		$("endeavor_clod"), 
		$("endeavor_oct"),
		$("matterhorn120_oct"),
		$("matterhorn120_clod"),
		$("spot_endeavor"),
		$("spot_endeavor_2"),
		$("spot_matterhorn"),
		$("tup"),
	].filter(node => node !== null);

	let triggerPosWorld = snRight.position.clone().applyMatrix4(snRight.world);

	//let nodesPosWorld = nodes.map( node => new Vector3(0, 0, 0).applyMatrix4(node.transform));

	{ // RIGHT CONTROLLER
		let state = vr.getControllerStateRight();

		if(state.pressed[OVRButtonID.Axis0]){
			let [x, y] = state.axis;

			let a = (Math.atan2(y, x) + 2 * Math.PI) % (2 * Math.PI);

			let s1 = Math.PI / 2;
			let s2 = s1 + 2 * Math.PI / 3;
			let s3 = s2 + 2 * Math.PI / 3;

			if(a > s1 && a < s2){
				//USER_STUDY_RENDER_OCTREE = true;
				//USER_STUDY_RENDER_CLOD = false;
				//USER_STUDY_OCTREE_MODE = "ADAPTIVE";
				US_setMethodA();
			}else if(a > s2 && a < s3){
				//USER_STUDY_RENDER_OCTREE = false;
				//USER_STUDY_RENDER_CLOD = true;
				US_setMethodC();
			}else{
				//USER_STUDY_RENDER_OCTREE = true;
				//USER_STUDY_RENDER_CLOD = false;
				//USER_STUDY_OCTREE_MODE = "FIXED";
				US_setMethodB();
			}


		}

		let triggerPressed = state.pressed[OVRButtonID.SteamVR_Trigger];
		if(triggerPressed && tmDragStart === null){

			let nodeTransforms = new Map(nodes.map( node => [node, node.world.clone()] ));

			tmDragStart = {
				triggerPos: triggerPosWorld,
				nodeTransforms: nodeTransforms,
				//nodePos: nodePosWorld,
				//nodeTransform: node.transform.clone()
			};

		}else if(triggerPressed){

			let diff = new Vector3().subVectors(triggerPosWorld, tmDragStart.triggerPos);

			

			//let newNodeTransform = tmDragStart.nodeTransform.multiply(diffTransform);
			//let newNodeTransform = diffTransform.multiply(tmDragStart.nodeTransform);

			for(let node of nodes){
				let diffTransform = new Matrix4().makeTranslation(diff.x, diff.y, diff.z);
				let startNodeTransform = tmDragStart.nodeTransforms.get(node);
				let newNodeTransform = diffTransform.multiply(startNodeTransform);

				node.world.copy(newNodeTransform);

				//log(node.name);
			}

		}else{
			tmDragStart = null;
		}
		
	}


	{ // LEFT CONTROLLER
		let state = vr.getControllerStateLeft();

		if(state.pressed[OVRButtonID.Axis0]){
			let [x, y] = state.axis;

			let a = (Math.atan2(y, x) + 2 * Math.PI) % (2 * Math.PI);

			let s1 = 0;
			let s2 = Math.PI;

			if(a > s1 && a < s2){
				USER_STUDY_LOD_MODIFIER += 0.05;
				USER_STUDY_LOD_MODIFIER = Math.min(USER_STUDY_LOD_MODIFIER, 1);
			}else{
				USER_STUDY_LOD_MODIFIER -= 0.05;
				USER_STUDY_LOD_MODIFIER = Math.max(USER_STUDY_LOD_MODIFIER, -1);
			}
		}
	}

	

}

var lastUpdate = now();

var update = function() {

	let start = now();

	for(let listener of listeners.update){
		listener();
	}

	updateCamera();

	scene.root.update();

	let duration = now() - start;
	let durationMS = (duration * 1000).toFixed(3);
	setDebugValue("duration.cp.update", `${durationMS}ms`);

	{

		let pos = view.position.toArray().map(v => v.toFixed(3)).join(", ");
		let target = view.getPivot().toArray().map(v => v.toFixed(3)).join(", ");

		setDebugValue("setView", 
`view.set(
	[${pos}], 
	[${target}]
);`);

	}
	
};

"update.js"