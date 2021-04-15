
let dataDirectory = "D:/dev/pointclouds/benchmark";

// let targetFile = "../../doc/benchmarks/data/endeavor_original_3090.json";
// let targetFile = "../../doc/benchmarks/data/endeavor_morton_3090.json";
// let targetFile = "../../doc/benchmarks/data/endeavor_shuffled_3090.json";
let targetFile = "../../doc/benchmarks/data/endeavor_shuffled_morton_3090.json";
// let targetFile = "../../doc/benchmarks/data/models_3090.json";
// let targetFile = "../../doc/benchmarks/data/viewpoints_3090.json";
// let targetFile = "../../doc/benchmarks/data/models_3090.json";
let warmupTime = 3;

let scenarios = [
	// {
	// 	model: "Lion",
	// 	view: [
	// 		[4.482548084301243, 3.9856851176075163, 1.9479170192704958],
	// 		[0.6162662017196654, 1.65415456111992, -2.6637675862770447],
	// 	],
	// 	fov: 60,
	// 	sortings: [
	// 		{name: "original"         , path: `${dataDirectory}/lion/original.las`},
	// 		{name: "morton"           , path: `${dataDirectory}/lion/morton.las`},
	// 		{name: "shuffled"         , path: `${dataDirectory}/lion/shuffled.las`},
	// 		{name: "morton-shuffled"  , path: `${dataDirectory}/lion/morton_shuffled.las`},
	// 	],
	// },


	// {
	// 	model: "Lifeboat",
	// 	view: [
	// 		[-4.192893387200137, 8.489790368140836, -14.494595950465634],
	// 		[7.49495077761617, 4.847105868878701, -8.046285665009435],
	// 	],
	// 	fov: 60,
	// 	sortings: [
	// 		{name: "original"         , path: `${dataDirectory}/lifeboat/original.las`},
	// 		{name: "morton"           , path: `${dataDirectory}/lifeboat/morton.las`},
	// 		{name: "shuffled"         , path: `${dataDirectory}/lifeboat/shuffled.las`},
	// 		{name: "morton-shuffled"  , path: `${dataDirectory}/lifeboat/morton_shuffled.las`},
	// 	],
	// },


	// {
	// 	model: "Retz",
	// 	view: [
	// 		[601.5844144526318, 88.77716966321097, -825.3167255422129],
	// 		[574.8954999244643, 53.27534573798231, -862.3046610359106],
	// 	],
	// 	fov: 80,
	// 	sortings: [
	// 		{name: "original"         , path: `${dataDirectory}/retz/original.las`},
	// 		{name: "morton"           , path: `${dataDirectory}/retz/morton.las`},
	// 		{name: "shuffled"         , path: `${dataDirectory}/retz/shuffled.las`},
	// 		{name: "morton-shuffled"  , path: `${dataDirectory}/retz/morton_shuffled.las`},
	// 	],
	// },


	// {
	// 	model: "Retz - no points",
	// 	view: [
	// 		[576.3328212443457, 29.88865657458824, -857.0353218912022],
	// 		[574.2656798968638, 25.772263202753667, -859.7262103826686],
	// 	],
	// 	fov: 80,
	// 	sortings: [
	// 		{name: "original"         , path: `${dataDirectory}/retz/original.las`},
	// 		{name: "morton"           , path: `${dataDirectory}/retz/morton.las`},
	// 		{name: "shuffled"         , path: `${dataDirectory}/retz/shuffled.las`},
	// 		{name: "morton-shuffled"  , path: `${dataDirectory}/retz/morton_shuffled.las`},
	// 	],
	// },
	// {
	// 	model: "Retz - closeup",
	// 	view: [
	// 		[601.5844144526318, 88.77716966321097, -825.3167255422129],
	// 		[574.8954999244643, 53.27534573798231, -862.3046610359106],
	// 	],
	// 	fov: 80,
	// 	sortings: [
	// 		{name: "original"         , path: `${dataDirectory}/retz/original.las`},
	// 		{name: "morton"           , path: `${dataDirectory}/retz/morton.las`},
	// 		{name: "shuffled"         , path: `${dataDirectory}/retz/shuffled.las`},
	// 		{name: "morton-shuffled"  , path: `${dataDirectory}/retz/morton_shuffled.las`},
	// 	],
	// },
	// {
	// 	model: "Retz - overview",
	// 	view: [
	// 		[283.4061054620159, 698.6431442370857, -1003.85193836096],
	// 		[591.6650729160136, 11.267937007744763, -922.1270100458455],
	// 	],
	// 	fov: 80,
	// 	sortings: [
	// 		{name: "original"         , path: `${dataDirectory}/retz/original.las`},
	// 		{name: "morton"           , path: `${dataDirectory}/retz/morton.las`},
	// 		{name: "shuffled"         , path: `${dataDirectory}/retz/shuffled.las`},
	// 		{name: "morton-shuffled"  , path: `${dataDirectory}/retz/morton_shuffled.las`},
	// 	],
	// },


	{
		model: "Endeavor",
		view: [
			[602.889902834417, 508.5243830928026, -598.4410889950284],
			[601.7127514620295, 509.11473092203255, -605.4171365393083],
		],
		fov: 80,
		sortings: [
			// {name: "original"         , path: `${dataDirectory}/endeavor/original.las`},
			// {name: "morton"           , path: `${dataDirectory}/endeavor/morton.las`},
			// {name: "shuffled"         , path: `${dataDirectory}/endeavor/shuffled.las`},
			{name: "morton-shuffled"  , path: `${dataDirectory}/endeavor/morton_shuffled.las`},

			// {name: "morton-shuffled"  , path: `${dataDirectory}/endeavor/morton_shuffled_32_full.las`},
			// {name: "morton-shuffled"  , path: `${dataDirectory}/endeavor/morton_shuffled_64_full.las`},
			// {name: "morton-shuffled"  , path: `${dataDirectory}/endeavor/morton_shuffled_128_full.las`},
			// {name: "morton-shuffled"  , path: `${dataDirectory}/endeavor/morton_shuffled_256_full.las`},
			// {name: "morton-shuffled"  , path: `${dataDirectory}/endeavor/morton_shuffled_512_full.las`},
			// {name: "morton-shuffled"  , path: `${dataDirectory}/endeavor/morton_shuffled_1024_full.las`},
			// {name: "morton-shuffled"  , path: `${dataDirectory}/endeavor/morton_shuffled_2048_full.las`},
			// {name: "morton-shuffled"  , path: `${dataDirectory}/endeavor/morton_shuffled_4096_full.las`},
			// {name: "morton-shuffled"  , path: `${dataDirectory}/endeavor/morton_shuffled_8192_full.las`},
			// {name: "morton-shuffled"  , path: `${dataDirectory}/endeavor/morton_shuffled_16384_full.las`},
		],
	},
	
];

var methods = [
	{
		stepID: "GL_POINTS",
		renderFunction: renderPointCloudBasic,
		warmupTime: warmupTime,
	},{
		stepID: "atomicMin",
		renderFunction: renderPointCloudCompute,
		warmupTime: warmupTime,
	},{
		stepID: "reduce",
		renderFunction: render_compute_ballot,
		warmupTime: warmupTime,
	},{
		stepID: "early-z",
		renderFunction: render_compute_earlyDepth,
		warmupTime: warmupTime,
	},{
		stepID: "reduce,early-z",
		renderFunction: render_compute_ballot_earlyDepth,
		warmupTime: warmupTime,
	},{
		stepID: "dedup",
		renderFunction: render_compute_ballot_earlyDepth_dedup,
		warmupTime: warmupTime,
	},{
		stepID: "just-set",
		renderFunction: renderComputeJustSet,
		warmupTime: warmupTime,
	},{
		stepID: "HQS",
		renderFunction: renderComputeHQS,
		warmupTime: warmupTime,
	},{
		stepID: "HQS1x,protected",
		renderFunction: renderComputeHQS_1x64bit_fast,
		warmupTime: warmupTime,
	},{
		stepID: "guenther",
		renderFunction: render_compute_guenther,
		warmupTime: warmupTime,
	},
];

function setupScenario(scenario, sorting){

	if($("testcloud")){
		let pc = $("testcloud");
		scene.root.remove(pc);
		pc.las.dispose();
	}

	renderBenchmark = render_compute_ballot_earlyDepth;

	log(`setting scenario: ${sorting.path}`);

	let las = loadLASProgressive(sorting.path);

	let pc = new PointCloudProgressive("testcloud", "blabla");
	pc.boundingBox.min.set(...las.boundingBox.min);
	pc.boundingBox.max.set(...las.boundingBox.max);
	pc.las = las;

	let handles = las.handles;

	let attributes = [
		new GLBufferAttribute("position", 0, 3, gl.FLOAT, gl.FALSE, 12, 0),
		new GLBufferAttribute("color",    1, 4, gl.UNSIGNED_BYTE, gl.TRUE, 4, 12),
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

	scene.root.add(pc);

	pc.numPoints = las.numPoints;

	view.set(...scenario.view);
	camera.fov = scenario.fov;

	return new Promise(resolve => {
		let doneListener = () => {
			if(las.isDone()){
				listeners.update = listeners.update.filter(l => l !== doneListener);

				resolve();
			}
		};
		listeners.update.push(doneListener);
	});
}

async function benchmarkMethods(){

	let timings = [];

	for(let method of methods){

		log(`setting method: ${method.renderFunction.name}`);
		renderBenchmark = method.renderFunction;

		await sleep(method.warmupTime);

		let data = getTimings();
		let json = JSON.parse(data);
		let result = {
			method: method.stepID,
			durations: json.timings,
		}


		timings.push(result);

	}

	return timings;
}

async function run(){

	let json = {
		scenarios: [],
	};

	for(let scenario of scenarios){
		for(let sorting of scenario.sortings){
			await setupScenario(scenario, sorting);
			let timings = await benchmarkMethods();

			let jsScenario = {
				model: scenario.model,
				sorting: sorting.name,
				timings: timings,
			};

			json.scenarios.push(jsScenario);
		}
	}

	let file = targetFile;
	let str = JSON.stringify(json, null, '\t');
	writeTextFile(file, str);

	log("done");

}

window.width = 1920;
window.height = 1080;

window.x = 2560;
window.y = 0;
// camera.fov = 80;
// camera.near = 0.1;

run();
