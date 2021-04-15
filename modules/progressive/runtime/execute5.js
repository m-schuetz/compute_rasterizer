
if(true){
	
	let las = loadLASProgressive("D:/dev/pointclouds/pix4d/matterhorn.las");
	//let las = loadBINProgressive("D:/dev/pointclouds/pix4d/matterhorn.bin");
	
	let pc = new PointCloudProgressive("testcloud", "blabla");

	let handles = las.handles;

	let attributes = [
		new GLBufferAttribute("position", 0, 3, gl.FLOAT, gl.FALSE, 12, 0),
		new GLBufferAttribute("value", 1, 4, gl.INT, gl.FALSE, 4, 12, {targetType: "int"}),
	];
	let bytesPerPoint = attributes.reduce( (p, c) => p + c.bytes, 0);

	let maxPointsPerBuffer = 134 * 1000 * 1000;
	let numPointsLeft = las.numPoints;
	let glBuffers = handles.map( (handle) => {

		let numPointsInBuffer = numPointsLeft > maxPointsPerBuffer ? maxPointsPerBuffer : numPointsLeft;
		numPointsLeft -= maxPointsPerBuffer;

		let glbuffer = new GLBuffer();

		glbuffer.attributes = attributes;

		gl.bindVertexArray(glbuffer.vao);
		glbuffer.vbo = handle;
		gl.bindBuffer(gl.ARRAY_BUFFER, glbuffer.vbo);

		for(let attribute of attributes){

			let {location, count, type, normalize, offset} = attribute;

			gl.enableVertexAttribArray(location);

			if(attribute.targetType === "int"){
				gl.vertexAttribIPointer(location, count, type, bytesPerPoint, offset);
			}else{
				gl.vertexAttribPointer(location, count, type, normalize, bytesPerPoint, offset);
			}
		}

		gl.bindVertexArray(0);

		glbuffer.count =  numPointsInBuffer;

		return glbuffer;
	});

	pc.glBuffers = glBuffers;

	let s = 0.3;
	pc.transform.elements.set([
	//pc.world.elements.set([
		s, 0, 0, 0, 
		0, 0, -s, 0, 
		0, s, 0, 0, 
		-10, 1.4, -11, 1, 
	]);


	// {// Matterhorn VR
	// 	let s = 0.003;
	// 	pc.transform.elements.set([
	// 	//pc.world.elements.set([
	// 		s, 0, 0, 0, 
	// 		0, 0, s, 0, 
	// 		0, s, 0, 0, 
	// 		-3, 0.8, -1, 1, 
	// 	]);
	// }

	scene.root.add(pc);

	listeners.update.push(() => {
		pc.numPoints = las.numPoints;
	});

}

// if($("testcloud")){
// 	let pc = $("testcloud");
// 	let s = 0.0005;
// 	pc.transform.elements.set([
// 		s, 0, 0, 0, 
// 		0, 0, s, 0, 
// 		0, s, 0, 0, 
// 		-2.5, -1.5, -2, 1, 
// 	]);

// 	log("lala");
// }

view.set(
	[1854.689, 1835.803, -1851.135], 
	[897.119, 879.788, -684.016]
);