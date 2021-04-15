
if(true){
	log("run");
	let n = 100;

	let numPoints = n * n;
	let vertices = new Float32Array(numPoints * 4);
	let verticesU8 = new Uint8Array(vertices.buffer);
	let bytesPerPoint = 16;

	let vindex = 0;
	for(let i = 0; i < n; i++){
		for(let j = 0; j < n; j++){

			let u = 2 * Math.PI * i / n - Math.PI;
			let v = 2 * Math.PI * j / n - Math.PI;

			let radius = 1;
			let x = radius * Math.sin(u) * Math.sin(v); // + 0.05 * Math.cos(20 * v);
			let y = radius * Math.cos(u) * Math.sin(v); // + 0.05 * Math.cos(20 * u);
			let z = radius * Math.cos(v);

			let r = (u + Math.PI) / 2 * Math.PI;
			let g = (v + Math.PI) / 2 * Math.PI;
			let b = 0;

			[r, g, b] = [0.1, 0.22, 0.02];
			r = 255 * (x + radius) / (2 * radius);
			g = 255 * (y + radius) / (2 * radius);
			b = 255 * (z + radius) / (2 * radius);
			//b = 0;

			r = r < 200 ? 0 : r;
			g = g < 200 ? 0 : g;
			b = b < 200 ? 0 : b;

			vertices[4 * vindex + 0] = x;
			vertices[4 * vindex + 1] = y;
			vertices[4 * vindex + 2] = z;

			verticesU8[16 * vindex + 12] = r;
			verticesU8[16 * vindex + 13] = g;
			verticesU8[16 * vindex + 14] = b;
			verticesU8[16 * vindex + 15] = 255;

			vindex++;
		}
	}

	let wobblySphere = scene.root.find("originMarker");
	
	if(!wobblySphere){
		wobblySphere = new SceneNode("originMarker");
		let wobblySphereBuffer = new GLBuffer();
		wobblySphere.components.push(wobblySphereBuffer);	

		scene.root.add(wobblySphere);
	}
	let wobblySphereBuffer = wobblySphere.getComponents(GLBuffer)[0];

	wobblySphereBuffer.set(vertices, vindex);
}




if(false){
	log("run");
	let n = 300;

	let numPoints = n * n;
	let vertices = new Float32Array(numPoints * 4);
	let verticesU8 = new Uint8Array(vertices.buffer);
	let bytesPerPoint = 16;

	let vindex = 0;
	for(let i = 0; i < n; i++){
		for(let j = 0; j < n; j++){

			let u = 4 * Math.PI * i / n - Math.PI;
			let v = 4 * Math.PI * j / n - Math.PI;

			let radius = 3;
			let x = radius * Math.sin(u) * Math.sin(v); // + 0.05 * Math.cos(20 * v);
			let y = radius * Math.cos(u) * Math.sin(v); // + 0.05 * Math.cos(20 * u);
			let z = radius * Math.cos(v);

			let r = (u + Math.PI) / 2 * Math.PI;
			let g = (v + Math.PI) / 2 * Math.PI;
			let b = 0;

			[r, g, b] = [0.1, 0.22, 0.02];
			r = 255 * (x + radius) / (2 * radius);
			g = 255 * (y + radius) / (2 * radius);
			b = 255 * (z + radius) / (2 * radius);

			r = r < 200 ? 0 : r;
			g = g < 200 ? 0 : g;
			b = b < 200 ? 0 : b;

			vertices[4 * vindex + 0] = x;
			vertices[4 * vindex + 1] = y;
			vertices[4 * vindex + 2] = z;

			verticesU8[16 * vindex + 12] = r;
			verticesU8[16 * vindex + 13] = g;
			verticesU8[16 * vindex + 14] = b;
			verticesU8[16 * vindex + 15] = 255;

			vindex++;
		}
	}

	let wobblySphere = scene.root.find("wobblySphere");
	
	if(!wobblySphere){
		wobblySphere = new SceneNode("wobblySphere");
		let wobblySphereBuffer = new GLBuffer();
		wobblySphere.components.push(wobblySphereBuffer);	

		scene.root.add(wobblySphere);
	}
	let wobblySphereBuffer = wobblySphere.getComponents(GLBuffer)[0];

	wobblySphereBuffer.set(vertices, vindex);

	wobblySphere.position.set(0, 0, 0);
	wobblySphere.scale.set(1, 1, 1);
	//wobblySphere.lookAt(1, 1, -1);
}

vr.start();

{
	let position = camera.position;
	let target = camera.getDirectionWorld().multiplyScalar(view.radius).add(position);

	setDebugValue("camPos", position.toArray().map(v => v.toFixed(3)).join(", "))
	setDebugValue("target", target.toArray().map(v => v.toFixed(3)).join(", "))
}
