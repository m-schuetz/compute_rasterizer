

if(typeof e4called === "undefined"){
	e4called = true;
	
	let las = loadLASProgressive("D:/dev/pointclouds/benchmark/lion/original.las");

	let pc = new PointCloudProgressive("testcloud", "blabla");
	pc.boundingBox.min.set(...las.boundingBox.min);
	pc.boundingBox.max.set(...las.boundingBox.max);

	log(pc.boundingBox);

	let handles = las.handles;

	let attributes = [
		new GLBufferAttribute("position", 0, 3, gl.FLOAT, gl.FALSE, 12, 0),
		new GLBufferAttribute("color",    1, 4, gl.UNSIGNED_BYTE, gl.TRUE, 4, 12),
		// new GLBufferAttribute("value", 1, 4, gl.INT, gl.FALSE, 4, 12, {targetType: "int"}),
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

	let s = 1;
	pc.transform.elements.set([
		s, 0, 0, 0, 
		0, 0, -s, 0, 
		0, s, 0, 0, 
		0, 0, 0, 1, 
	]);

	scene.root.add(pc);

	pc.numPoints = las.numPoints;
}

window.width = 1920;
window.height = 1080;

window.x = 2560;
window.y = 0;

view.set(
	[4.482548084301243, 3.9856851176075163, 1.9479170192704958],
	[0.6162662017196654, 1.65415456111992, -2.6637675862770447],
);
camera.fov = 60;
camera.near = 0.1;

log(`
view.set(
	[${view.position}],
	[${view.getPivot()}],
);
`);


renderDebug = renderComputeHQS_1x64bit_fast;
renderDebug = render_compute_ballot_earlyDepth;
renderDebug = renderPointCloudBasic;

// setTimestampEnabled(false);

if(typeof af032f !== "undefined"){
	listeners.update = listeners.update.filter(l => l !== af032f);
}

var dbgLast = now();
var dbgLastFC = frameCount;
var af032f = () => {

	if(now() - dbgLast > 1){
		log(frameCount - dbgLastFC);
		dbgLast = now();

		dbgLastFC = frameCount;
	}

};
listeners.update.push(af032f);