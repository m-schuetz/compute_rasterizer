
if(!$("testcloud")){
	
	//let las = loadBINProgressive("../../morro_bay.bin");
	//let las = loadBINProgressive("../../morro_bay_1billion.bin");
	let las = loadBINProgressive("D:/dev/pointclouds/open_topography/ca13/morro_bay.bin");
	// let las = loadBINProgressive("D:/dev/pointclouds/open_topography/ca13/morro_bay_1billion.bin");

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

	let s = 1.0;
	pc.transform.elements.set([
		s, 0, 0, 0, 
		0, 0, -s, 0, 
		0, s, 0, 0, 
		0, 0, 0, 1, 
	]);

	pc.numPoints = las.numPoints;

	scene.root.add(pc);

	// listeners.update.push(() => {
	// 	pc.numPoints = las.numPoints;
	// });

}

view.set(
	[485.168, 1511.749, 21.458], 
	[1535.947, 7.916, -1492.148]
);

window.x = 0;
window.y = 0;
window.width = 1920;
window.height = 1080;
// window.width = 2560;
// window.height = 1440;

MSAA_SAMPLES = 1;
EDL_ENABLED = false;

// renderDebug = renderPointCloudBasic;
// renderDebug = renderPointCloudProgressive;

camera.near = 10;

function measurePerformance(label){
	let vProgressive = getDebugValue("gl.render.progressive");
	let vPass1 = getDebugValue("gl.render.progressive.p1_reproject");
	let vPass2 = getDebugValue("gl.render.progressive.p2_fill.render_fixed");
	let vPass3 = getDebugValue("gl.render.progressive.p3_vbo");
	let vBruteforce = getDebugValue("gl.render.basic");

	let algorithm = null;
	if(renderDebug === renderPointCloudBasic){
		algorithm = "bruteforce";
	}else if(renderDebug === renderPointCloudProgressive){
		algorithm = "progressive";
	}

	let content = `

##
## ${label}
##

EDL_ENABLED: ${EDL_ENABLED},
MSAA_SAMPLES: ${MSAA_SAMPLES},
PROGRESSIVE_BUDGET: ${PROGRESSIVE_BUDGET},
algorithm: ${algorithm}

measurements: 
`;

	if(algorithm === "bruteforce"){

		lines = [`bruteforce: ${vBruteforce}`];

		content += lines.join("\n");
	}else if(algorithm === "progressive"){

		lines = [
			`progressive: ${vProgressive}`,
			`pass1: ${vPass1}`,
			`pass2: ${vPass2}`,
			`pass3: ${vPass3}`,
		];

		content += lines.join("\n");
	}


	fsFileAppendText("./benchmark.txt", content);
}

async function runTest(){
	GLTimerQueries.enabled = true;

	for(let i = 0; i < 1000; i++){

		await sleep(1);

		let progress = getDebugValue("pointcloud_progress");
		log(`load progress: ${progress}`);

		if(progress === "100%"){
			break;
		}
	}

	log("start benchmarks");

	fsFileDelete("./benchmark.txt");

	await sleep(1);
	
	{ // TEST 1
		log("EXECUTING TEST 1");

		EDL_ENABLED = false;
		MSAA_SAMPLES = 1;
		PROGRESSIVE_BUDGET = 1 * 1000 * 1000;
		renderDebug = renderPointCloudProgressive;

		// let state settle for a bit
		await sleep(2);
		
		measurePerformance("TEST 1");
	}

	{ // TEST 2
		log("EXECUTING TEST 2");

		EDL_ENABLED = false;
		MSAA_SAMPLES = 1;
		PROGRESSIVE_BUDGET = 1 * 1000 * 1000;
		renderDebug = renderPointCloudProgressive;

		// let state settle for a bit
		await sleep(2);
		
		measurePerformance("TEST 2");
	}

	{ // TEST 3
		log("EXECUTING TEST 3");

		EDL_ENABLED = false;
		MSAA_SAMPLES = 1;
		PROGRESSIVE_BUDGET = 10 * 1000 * 1000;
		renderDebug = renderPointCloudProgressive;

		// let state settle for a bit
		await sleep(2);
		
		measurePerformance("TEST 3");
	}

	{ // TEST 4
		log("EXECUTING TEST 4");

		EDL_ENABLED = false;
		MSAA_SAMPLES = 1;
		PROGRESSIVE_BUDGET = 10 * 1000 * 1000;
		renderDebug = renderPointCloudProgressive;

		// let state settle for a bit
		await sleep(2);
		
		measurePerformance("TEST 4");
	}

	{ // TEST 5 - NOW TESTING BRUTEFORCE!!!
		log("EXECUTING TEST 5");

		EDL_ENABLED = false;
		MSAA_SAMPLES = 1;
		PROGRESSIVE_BUDGET = 1 * 1000 * 1000;
		renderDebug = renderPointCloudBasic;

		// let state settle for a bit
		await sleep(2);
		
		measurePerformance("TEST 5");
		await sleep(1);
		measurePerformance("TEST 5.1");
	}

	// back to progressive
	PROGRESSIVE_BUDGET = 1 * 1000 * 1000;
	renderDebug = renderPointCloudProgressive;

	log("Benchmark finished.");
	log("results at './bin/Release_x64/benchmark.txt'");


};

runTest();















