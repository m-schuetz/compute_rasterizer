

if(typeof e4called === "undefined"){
	e4called = true;
	
	let las = loadLASProgressive("D:/dev/pointclouds/tuwien_baugeschichte/candi sari/candi_sari.las");
	// let las = loadLASProgressive("D:/dev/pointclouds/tuwien_baugeschichte/candi Banyunibo/candi_banyunibo.las");
	// let las = loadLASProgressive("D:/dev/pointclouds/benchmark/retz/shuffled.las");
	// let las = loadLASProgressive("D:/dev/pointclouds/benchmark/retz/morton.las");
	// let las = loadLASProgressive("D:/dev/pointclouds/benchmark/retz/morton_shuffled.las");

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



window.width = 2560;
window.height = 1440;

window.width = 800;
window.height = 600;

window.width = 1920;
window.height = 1080;

// window.width = 3000;
// window.height = 2000;


window.x = 2560;
window.y = 0;
camera.fov = 50;

view.set(
        [840.7870481845399, 98.0626551830085, -1023.1216631614049],
        [849.0793344343361, 91.08280768857297, -1033.8072114309473],
);

log(`
view.set(
	[${view.position}],
	[${view.getPivot()}],
);
`);

renderBenchmark = render_compute_fragcount;
renderBenchmark = renderComputeHQS_1x64bit_fast;
renderBenchmark = undefined;


// {

// 	let start = [
// 		[554.5518151228893, 34.61396109170297, -863.4523300239539],
// 		[559.2976598058233, 28.489001181097635, -862.4697492083494],
// 	];

// 	let end = [
// 		[98.86893669492179, 622.7156687512672, -957.7970219083803],
// 		[559.2976598058233, 28.489001181097592, -862.4697492083494],
// 	];

// 	let tStart = now();
// 	let warmup = 2;
// 	let duration = 1;

// 	let timings = [];

// 	let move = () => {

// 		if(now() < tStart + warmup){
// 			view.set(...start);

// 			return;
// 		}

// 		let t = (now() - tStart - warmup) / duration;
// 		let w = t * t;

// 		let pos = [
// 			(1 - w) * start[0][0] + w * end[0][0],
// 			(1 - w) * start[0][1] + w * end[0][1],
// 			(1 - w) * start[0][2] + w * end[0][2],
// 		];

// 		let target = [
// 			(1 - w) * start[1][0] + w * end[1][0],
// 			(1 - w) * start[1][1] + w * end[1][1],
// 			(1 - w) * start[1][2] + w * end[1][2],
// 		];
		
// 		view.set(pos, target);

		
// 		if(now() > tStart + warmup + duration){
// 			listeners.update = listeners.update.filter(f => f !== move);
// 			log(timings.map(t => `${t.t.toFixed(3)}\t${t.duration.toFixed(3)}`).join("\n"));
// 		}else{

// 			let data = getTimings();
// 			let json = JSON.parse(data);
// 			let avg = json.timings.find(t => t.label === "frame").avg;
			
// 			timings.push({t: t, duration: avg});
// 		}


// 	};

	

// 	listeners.update.push(move);
// }