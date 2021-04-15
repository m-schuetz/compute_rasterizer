

if(typeof e4called === "undefined"){
	e4called = true;
	
	// let las = loadLASProgressive("D:/dev/pointclouds/benchmark/retz/shuffled.las");
	// let las = loadLASProgressive("D:/dev/pointclouds/benchmark/retz/morton.las");
	let las = loadLASProgressive("D:/dev/pointclouds/benchmark/retz/original.las");
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


window.width = 2540;
window.height = 1080;

// window.width = 3000;
// window.height = 2000;


window.x = 2560;
window.y = 0;
camera.fov = 60;

// all points outside frustum
// view.set(
// 	[576.3328212443457, 29.88865657458824, -857.0353218912022],
// 	[574.2656798968638, 25.772263202753667, -859.7262103826686],
// );

// close-up
// view.set(
// 	[601.5844144526318, 88.77716966321097, -825.3167255422129],
// 	[574.8954999244643, 53.27534573798231, -862.3046610359106],
// );


// overview
// view.set(
// 	[283.4061054620159, 698.6431442370857, -1003.85193836096],
// 	[591.6650729160136, 11.267937007744763, -922.1270100458455],
// );

view.set(
        [720.3283763028116, 307.0203057869209, -532.7018672136085],
        [600.0421166600124, 59.74790106321831, -860.3317198773238],
);

// view.set(
//         [-4637.175248600517, 4295.5194445935385, -4086.060429616287],
//         [591.6650729160137, 11.267937007744877, -922.1270100458473],
// );

log(`
view.set(
	[${view.position}],
	[${view.getPivot()}],
);
`);

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
renderBenchmark = renderComputeHQS_1x64bit_fast;
renderBenchmark = renderComputeHQS;
renderBenchmark = renderComputeBasic;
renderBenchmark = render_compute_guenther;
renderBenchmark = render_compute_ballot_earlyDepth_dedup;
renderBenchmark = render_compute_ballot_earlyDepth;
renderBenchmark = renderPointCloudBasic;
renderBenchmark = undefined;