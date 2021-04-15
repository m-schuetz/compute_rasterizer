
if(!$("progressive")){
	
	let las = loadLASProgressive("D:/dev/pointclouds/ot_35121F2416A_1.las");

	let pc = new PointCloudProgressive("progressive", "blabla");

	let attributes = [
		new GLBufferAttribute("position", 0, 3, gl.FLOAT, gl.FALSE, 12, 0),
		new GLBufferAttribute("value", 1, 4, gl.INT, gl.FALSE, 4, 12, {targetType: "int"}),
	];
	let bytesPerPoint = attributes.reduce( (p, c) => p + c.bytes, 0);

	let maxPointsPerBuffer = 134 * 1000 * 1000;
	let numPointsLeft = las.numPoints;

	let glBuffers = las.handles.map( (handle) => {

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

	// flip y and z
	let s = 1.0;
	pc.transform.elements.set([
		s, 0, 0, 0, 
		0, 0, -s, 0, 
		0, s, 0, 0, 
		0, 0, 0, 1, 
	]);

	{ // set initial view position
		let min = new Vector3(...las.boundingBox.min);
		let max = new Vector3(...las.boundingBox.max);
		max = max.sub(min).applyMatrix4(pc.transform);
		min = min.sub(min).applyMatrix4(pc.transform);
		let d = min.distanceTo(max);

		const u = 0.2;
		const y = u * max.y + (1 - u) * min.y;
		const box = new Box3(min, max);
		const center = box.getCenter();
		center.y = y;

		const pos = [
			center.x + d * 0.5,
			center.y + d * 0.5,
			center.z + d * 0.5,
		];

		view.set(
			pos,
			center.toArray(),
		);
	}

	scene.root.add(pc);

	pc.numPoints = las.numPoints;

	// listeners.update.push(() => {
	// 	pc.numPoints = las.numPoints;
	// });

}else{
	log("point cloud already loaded once. Restart to load another.");
}

// view.set(
// 	[1000, 1000, 500],
// 	[500, 00, -500],
// );

window.x = window.monitors[0].width * 0.1;
window.y = window.monitors[0].height * 0.1;
window.width = window.monitors[0].width * 0.8;
window.height = window.monitors[0].height * 0.8;



MSAA_SAMPLES = 1;
EDL_ENABLED = true;
camera.near = 0.1;

