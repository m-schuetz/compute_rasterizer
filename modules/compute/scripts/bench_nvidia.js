

if(typeof e4called === "undefined"){
	e4called = true;
	
	// let las = loadLASProgressive("D:/temp/sorted/nvidia_sort_x.las");
	// let las = loadLASProgressive("D:/temp/sorted/nvidia_sort_morton.las");
	// let las = loadLASProgressive("D:/dev/pointclouds/benchmark/endeavor/original.las");
	// let las = loadLASProgressive("D:/dev/pointclouds/benchmark/endeavor/morton_shuffled.las");
	let las = loadLASProgressive("D:/dev/pointclouds/benchmark/endeavor/morton.las");
	// let las = loadLASProgressive("D:/dev/pointclouds/benchmark/endeavor/morton_shuffled_128_full.las");
	// let las = loadLASProgressive("D:/dev/pointclouds/benchmark/endeavor/morton_shuffled_2048_full.las");
	// let las = loadLASProgressive("D:/dev/pointclouds/benchmark/endeavor/morton_shuffled_4096_full.las");
	// let las = loadLASProgressive("D:/dev/pointclouds/benchmark/endeavor/morton_shuffled_8192_full.las");
	// let las = loadLASProgressive("D:/dev/pointclouds/benchmark/endeavor/morton_shuffled_16384_full.las");
	// let las = loadLASProgressive("D:/dev/pointclouds/nvidia.las");

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

	listeners.update.push(() => {

		if(pc.numPoints !== las.numPoints){
			//log(las.numPoints);
		}
		pc.numPoints = las.numPoints;
		
	});

	view.set(
		[490.3592362417345, 619.2836214329301, -553.9481303969455],
		[601.7127514620295, 509.1147309220324, -605.4171365393083],
	);

}

window.width = 1920;
window.height = 1080;

// window.width = 3000;
// window.height = 2000;

window.x = 2560;
window.y = 0;

// zoomed-in
// view.set(
// 	[602.889902834417, 508.5243830928026, -598.4410889950284],
// 	[601.7127514620295, 509.11473092203255, -605.4171365393083],
// );


// zoomed-out


camera.fov = 80;
camera.near = 0.1;

log(`
view.set(
	[${view.position}],
	[${view.getPivot()}],
);
`);

renderDebug = render_compute_ballot_earlyDepth;

// renderDebug = renderComputeHQS_1x64bit_fast;
// renderDebug = render_compute_ballot_earlyDepth;
// renderBenchmark = render_compute_fragcount;
// renderBenchmark = renderComputeHQS_1x64bit_fast;
// renderDebug = render_compute_ballot_earlyDepth;



// {

// 	let start = [
// 		[602.889902834417, 508.5243830928026, -598.4410889950284],
// 		[601.7127514620295, 509.11473092203255, -605.4171365393083],
// 	];

// 	let end = [
// 		[490.3592362417345, 619.2836214329301, -553.9481303969455],
// 		[601.7127514620295, 509.1147309220324, -605.4171365393083],
// 	];

// 	let tStart = now();
// 	let duration = 10;

// 	let timings = [];

// 	let move = () => {

// 		let t = (now() - tStart) / duration;
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

		
// 		if(now() > tStart + duration){
// 			listeners.update = listeners.update.filter(f => f !== move);

// 			// log(JSON.stringify(timings));

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


setTimestampEnabled(false);

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

renderBenchmark = render_compute_fragcount;
renderBenchmark = renderPointCloudBasic;
renderBenchmark = render_compute_ballot_earlyDepth;
renderBenchmark = renderComputeHQS_1x64bit_fast;
renderBenchmark = renderComputeHQS;
renderBenchmark = render_compute_ballot_earlyDepth_dedup;
renderBenchmark = undefined;