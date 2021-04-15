window.width = 1600;
window.height = 1137;
window.x  = 2560 + 20;
window.y = 23 + 20;

GLTimerQueries.enabled = false;
reportState(GLTimerQueries.enabled);

GLTimerQueries.history = new Map();

// log(camera.position)

// {
// 	let pc = new PointCloudLZ("test");

// 	let n = 10000;
// 	let position = new ArrayBuffer(n * 12);
// 	let color = new ArrayBuffer(n * 4);

// 	let vPosition = new DataView(position);
// 	let vColor = new DataView(color);

// 	for(let i = 0; i < n; i++){

// 		let x = Math.random() * 4 - 2;
// 		let y = Math.random() * 4 - 2;
// 		let z = Math.random() * 4 - 2;

// 		let r = Math.random() * 255;
// 		let g = Math.random() * 255;
// 		let b = Math.random() * 255;
// 		let a = 255;

// 		vPosition.setFloat32(12 * i + 0, x, true);
// 		vPosition.setFloat32(12 * i + 4, y, true);
// 		vPosition.setFloat32(12 * i + 8, z, true);

// 		vColor.setUint8(4 * i + 0, r);
// 		vColor.setUint8(4 * i + 1, g);
// 		vColor.setUint8(4 * i + 2, b);
// 		vColor.setUint8(4 * i + 3, a);

// 	}

// 	pc.setData(n, position, color);

// 	scene.root.add(pc);

// }




if(typeof LaszipVlr === "undefined"){
	LaszipVlr = class LaszipVlr{
		constructor(){
			this.compressor = 0;
			this.coder = 0;
			this.versionMajor = 0;
			this.versionMinor = 0;
			this.versionRevision = 0;
			this.options = 0;
			this.chunkSize = 0;
			this.numberOfSpecialEvlrs = 0;
			this.offsetToSpecialEvlrs = 0;
			this.numItems = 0;
			this.items = [];

		}
	};
}


parseCString = (cstring) => {

	let end = cstring.byteLength;
	let view = new DataView(cstring);

	let str = "";
	for(let i = 0; i < cstring.byteLength; i++){
		if(view.getUint8(i) === 0){
			end = i;
			break;
		}else{
			str += String.fromCharCode(view.getUint8(i));
		}
	}

	log(str);

	return str;
};

parseLaszipVlr = (buffer) => {
	let vlr = new LaszipVlr();
	let view = new DataView(buffer);

	vlr.compressor = view.getUint16(0, true);
	vlr.coder = view.getUint16(2, true);
	vlr.versionMajor = view.getUint8(4, true);
	vlr.versionMinor = view.getUint8(5, true);
	vlr.versionRevision = view.getUint16(6, true);
	vlr.options = view.getUint32(8, true);
	vlr.chunkSize = view.getUint32(12, true);
	vlr.numberOfSpecialEvlrs = view.getBigInt64(16, true);
	vlr.offsetToSpecialEvlrs = view.getBigInt64(24, true);
	vlr.numItems = view.getUint16(32, true);

	for(let i = 0; i < vlr.numItems; i++){
		let item = {
			type: view.getUint16(34 + i * 6 + 0, true),
			size: view.getUint16(34 + i * 6 + 2, true),
			version: view.getUint16(34 + i * 6 + 4, true),
		};

		vlr.items.push(item);
	}

	return vlr;
};

{
	let data = readFile(`D:/dev/pointclouds/heidentor.laz`);
	// let data = readFile(`${rootDir}/resources/models/heidentor_small.laz`);
	let view = new DataView(data);
	log(data.byteLength);

	let headerSize = view.getUint16(94, true);
	log(headerSize);

	let versionMajor = view.getUint8(24);
	let versionMinor = view.getUint8(25);
	let offsetToPointData = view.getUint32(96, true);
	let numVLRs = view.getUint32(100, true);
	let point_data_format = view.getUint8(104);
	let point_record_length = view.getUint16(105, true);
	let numPoints = view.getUint32(107, true);

	let scale = {
		x: view.getFloat64(131, true),
		y: view.getFloat64(139, true),
		z: view.getFloat64(147, true),
	};
	let offset = {
		x: view.getFloat64(155, true),
		y: view.getFloat64(163, true),
		z: view.getFloat64(171, true),
	};
	let max = {
		x: view.getFloat64(179, true),
		y: view.getFloat64(195, true),
		z: view.getFloat64(211, true),
	};
	let min = {
		x: view.getFloat64(187, true),
		y: view.getFloat64(203, true),
		z: view.getFloat64(219, true),
	};

	if(versionMajor === 1 && versionMinor >= 4){
		log("TODO");
		exit(123);
		//numPoints = view.readBigInt64LE(247);
	}

	let laszipVlr = null;

	// VARIABLE LENGTH RECORDS
	let vlrs = [];
	let byteOffset = headerSize;
	for(let i = 0; i < numVLRs; i++){
		// let vlrHeaderBuffer = Buffer.alloc(54);
		// await handle.read(vlrHeaderBuffer, 0, vlrHeaderBuffer.byteLength, byteOffset);
		let vlrHeaderBuffer = data.slice(byteOffset, byteOffset + 54);
		let vlrHeaderBufferView = new DataView(vlrHeaderBuffer);

		log("vlr");
		
		let userId = parseCString(vlrHeaderBuffer.slice(2, 18));
		let recordId = vlrHeaderBufferView.getUint16(18, true);
		let recordLength = vlrHeaderBufferView.getUint16(20, true);

		// let vlrContentBuffer = Buffer.alloc(recordLength);
		// await handle.read(vlrContentBuffer, 0, vlrContentBuffer.byteLength, byteOffset + 54);
		let vlrContentBuffer = data.slice(byteOffset + 54, byteOffset + 54 + recordLength);

		let isLaszipVlr = (userId === "laszip encoded" && recordId === 22204);
		if(isLaszipVlr){
			laszipVlr = parseLaszipVlr(vlrContentBuffer);
		}else{
			// TODO: parse regular VLR
		}

		byteOffset = byteOffset + 54 + recordLength;
	}


	{ // see lasreadpoint.cpp; read_chunk_table()

		let chunkTableStart = Number(view.getBigInt64(byteOffset, true));
		let chunkTableSize = data.byteLength - chunkTableStart;
		let chunkTableBuffer = data.slice(chunkTableStart, chunkTableStart + chunkTableSize);
		let chunkTableBufferView = new DataView(chunkTableBuffer);
		
		let version = chunkTableBufferView.getUint32(0, true);
		let numChunks = chunkTableBufferView.getUint32(4, true);

		let chunk_sizes = new Int32Array(numChunks);

		let dec = new ArithmeticDecoder(new Uint8Array(chunkTableBuffer), 8);
		let ic = new IntegerCompressor(dec, 32, 2);

		for(let i = 0; i < numChunks; i++){

			let pred = (i == 0) ? 0 : chunk_sizes[i - 1];

			let chunk_size = Number(ic.decompress(pred, 1));
			chunk_sizes[i] = chunk_size;
		}

		// header + vlrs + 8 bytes describing chunk table location
		let firstChunkOffset = byteOffset + 8;

		let chunk_starts = new Float64Array(numChunks);
		chunk_starts[0] = firstChunkOffset;
		for(let i = 1; i <= numChunks; i++){
			chunk_starts[i] = chunk_starts[i - 1] + chunk_sizes[i - 1];

			if(i < 10){
				log(chunk_starts[i]);
			}
		}

		log("chunkTableStart: " +  chunkTableStart);
		log("version: " +  version);
		log("numChunks: " +  numChunks);

		let chunkSize = laszipVlr.chunkSize;
		let pointsPerChunk = 10000;
		numChunks = 10;

		{ // TEST 

			let tStart = now();

			log(`[start]: ${now() - tStart}s`);
		
			// let pathTest = `${rootDir}/modules/laszip/init_test.cs`;
			let pathTest = `${rootDir}/modules/laszip/test.cs`;
			let csTest = new Shader([{type: gl.COMPUTE_SHADER, path: pathTest}]);

			log(`[load shader]: ${now() - tStart}s`);

			let bufferI = new ArrayBuffer(8000);
			let ssboTestI = gl.createBuffer();
			gl.namedBufferData(ssboTestI, bufferI.byteLength, bufferI, gl.DYNAMIC_DRAW);

			let bufferU = new ArrayBuffer(8000);
			let ssboTestU = gl.createBuffer();
			gl.namedBufferData(ssboTestU, bufferU.byteLength, bufferU, gl.DYNAMIC_DRAW);

			let lazbuffer = data;
			let ssboLazbuffer = gl.createBuffer();
			gl.namedBufferData(ssboLazbuffer, lazbuffer.byteLength, lazbuffer, gl.DYNAMIC_DRAW);

			let positionBuffer = new ArrayBuffer(12 * numChunks * chunkSize);
			let ssboPosition = gl.createBuffer();
			gl.namedBufferData(ssboPosition, positionBuffer.byteLength, positionBuffer, gl.DYNAMIC_DRAW);

			let colorBuffer = new ArrayBuffer(4 * numChunks * chunkSize);
			let ssboColor = gl.createBuffer();
			gl.namedBufferData(ssboColor, colorBuffer.byteLength, colorBuffer, gl.DYNAMIC_DRAW);

			let batchBuffer = new ArrayBuffer(32 * numChunks);
			let batchBufferView = new DataView(batchBuffer);
			for(let i = 0; i < numChunks; i++){
				let chunk_start = chunk_starts[i];
				let chunk_size = chunk_sizes[i];

				batchBufferView.setUint32(28 * i + 0, chunk_start, true);
				batchBufferView.setUint32(28 * i + 4, chunk_size, true);
				batchBufferView.setUint32(28 * i + 8, pointsPerChunk, true);
			}
			let ssboBatches = gl.createBuffer();
			gl.namedBufferData(ssboBatches, batchBuffer.byteLength, batchBuffer, gl.DYNAMIC_DRAW);

			// let point10s = new ArrayBuffer(4 * 200 * numChunks);
			// let ssboPoint10s = gl.createBuffer();
			// gl.namedBufferData(ssboPoint10s, point10s.byteLength, point10s, gl.DYNAMIC_DRAW);

			// let rgb12s = new ArrayBuffer(4 * 400 * numChunks);
			// let ssboRGB12s = gl.createBuffer();
			// gl.namedBufferData(ssboRGB12s, rgb12s.byteLength, rgb12s, gl.DYNAMIC_DRAW);

			let buffer_descriptor = new ArrayBuffer(12);
			let ssboBufferDescriptor = gl.createBuffer();
			gl.namedBufferData(ssboBufferDescriptor, buffer_descriptor.byteLength, buffer_descriptor, gl.DYNAMIC_DRAW);

			let buffer_uint = new ArrayBuffer(120000 * 4 * numChunks);
			let ssboBufferUint = gl.createBuffer();
			gl.namedBufferData(ssboBufferUint, buffer_uint.byteLength, buffer_uint, gl.DYNAMIC_DRAW);

			let buffer_am = new ArrayBuffer(2 * 280 * 24 * numChunks);
			let ssboBufferAM = gl.createBuffer();
			gl.namedBufferData(ssboBufferAM, buffer_am.byteLength, buffer_am, gl.DYNAMIC_DRAW);

			let buffer_sm5 = new ArrayBuffer(2 * 32 * 8 * 4 * numChunks);
			let ssboBufferSM5 = gl.createBuffer();
			gl.namedBufferData(ssboBufferSM5, buffer_sm5.byteLength, buffer_sm5, gl.DYNAMIC_DRAW);

			// log(`bufferI: ${bufferI.byteLength}`);
			// log(`bufferU: ${bufferU.byteLength}`);
			// log(`lazbuffer: ${lazbuffer.byteLength}`);
			// log(`positionBuffer: ${positionBuffer.byteLength}`);
			// log(`colorBuffer: ${colorBuffer.byteLength}`);
			// log(`batchBuffer: ${batchBuffer.byteLength}`);
			// log(`point10s: ${batchBuffer.byteLength}`);
			// log(`rgb12s: ${rgb12s.byteLength}`);
			// log(`buffer_uint: ${buffer_uint.byteLength}`);
			// log(`buffer_am: ${buffer_am.byteLength}`);
			// log(`buffer_sm5: ${buffer_sm5.byteLength}`);

			log(`[create buffers]: ${now() - tStart}s`);

			gl.useProgram(csTest.program);

			gl.bindBufferBase(gl.SHADER_STORAGE_BUFFER, 10, ssboTestI);
			gl.bindBufferBase(gl.SHADER_STORAGE_BUFFER, 11, ssboTestU);

			gl.bindBufferBase(gl.SHADER_STORAGE_BUFFER, 0, ssboLazbuffer);
			gl.bindBufferBase(gl.SHADER_STORAGE_BUFFER, 1, ssboPosition);
			gl.bindBufferBase(gl.SHADER_STORAGE_BUFFER, 2, ssboColor);
			gl.bindBufferBase(gl.SHADER_STORAGE_BUFFER, 3, ssboBatches);
			// gl.bindBufferBase(gl.SHADER_STORAGE_BUFFER, 4, ssboPoint10s);
			gl.bindBufferBase(gl.SHADER_STORAGE_BUFFER, 5, ssboBufferDescriptor);
			gl.bindBufferBase(gl.SHADER_STORAGE_BUFFER, 6, ssboBufferUint);
			gl.bindBufferBase(gl.SHADER_STORAGE_BUFFER, 7, ssboBufferAM);
			gl.bindBufferBase(gl.SHADER_STORAGE_BUFFER, 8, ssboBufferSM5);
			// gl.bindBufferBase(gl.SHADER_STORAGE_BUFFER, 9, ssboRGB12s);

			gl.uniform1i(csTest.uniforms.uNumChunks, numChunks);
			gl.uniform1i(csTest.uniforms.uPointsPerChunk, pointsPerChunk);
			// gl.uniform1i(csTest.uniforms.uBatchOffset, 0);
			//gl.uniform1i(csTest.uniforms.uNumChunks, numChunks);

			log(`[before dispatch]: ${now() - tStart}s`);

			// GLTimerQueries.mark("laszip-dispatch-start");


			// let query = GLTimerQueries.start("dispatch laszip");
			// // gl.dispatchCompute(50, 1, 1);
			// query.end();

			// gl.memoryBarrier(gl.ALL_BARRIER_BITS);

			// gl.memoryBarrier(gl.ALL_BARRIER_BITS);
			gl.memoryBarrier(gl.ALL_BARRIER_BITS);

			{
				let query = GLTimerQueries.start("dispatch laszip 1");
				gl.dispatchCompute(numChunks, 1, 1);
				query.end();
			}
			gl.memoryBarrier(gl.ALL_BARRIER_BITS);

			{
				let query = GLTimerQueries.start("dispatch laszip 2");
				gl.dispatchCompute(numChunks, 1, 1);
				query.end();
			}
			// gl.memoryBarrier(gl.ALL_BARRIER_BITS);

			// {
			// 	let query = GLTimerQueries.start("dispatch laszip 3");
			// 	gl.dispatchCompute(numChunks, 1, 1);
			// 	query.end();
			// }

			gl.memoryBarrier(gl.ALL_BARRIER_BITS);

			// query.end().then((result) => {
			// 	log(`dispatch: ${result.duration / 1000}ms`);
			// });

			// GLTimerQueries.mark("laszip-dispatch-end");
			// GLTimerQueries.measure("laszip-dispatch", "laszip-dispatch-start", "laszip-dispatch-end");

			log(`[after dispatch]: ${now() - tStart}s`);

			// { // INT DEBUG ARRAY
			// 	let target = new ArrayBuffer(12 * numChunks);
			// 	gl.getNamedBufferSubData(ssboTestI, 0, 4 * numChunks, target);
			// 	let u32 = new Int32Array(target);
			// 	let msg = "";
			// 	for(let i = 0; i < 30; i++){
			// 		log(`${i}: ${u32[i]}`);
			// 		// msg += `${i}: ${u32[i]}`.padEnd(15) + "| ";
			// 	}
			// 	log("int");
			// 	log(msg);
			// }

			// { // UINT DEBUG ARRAY
			// 	let target = new ArrayBuffer(12 * numChunks);
			// 	gl.getNamedBufferSubData(ssboTestU, 0, 4 * numChunks, target);
			// 	let u32 = new Uint32Array(target);
			// 	let msg = "";
			// 	for(let i = 0; i < 10; i++){
			// 		// log(`${i}: ${u32[i]}`);
			// 		msg += `${i}: ${u32[i]}`.padEnd(15) + "| ";
			// 	}
			// 	log("uint");
			// 	log(msg);
			// }

			// { // BUFFER DESCRIPTORS
			// 	let target = new ArrayBuffer(12);
			// 	gl.getNamedBufferSubData(ssboBufferDescriptor, 0, target.byteLength, target);
			// 	let u32 = new Uint32Array(target);
			// 	log(`#uints: ${u32[0]}`);
			// 	log(`#AMs: ${u32[1]}`);
			// 	log(`#SM5s: ${u32[2]}`);
			// }

			// let query = GLTimerQueries.start("dispatch laszip");

			// // gl.dispatchCompute(50, 1, 1);

			// query.end();

			// gl.memoryBarrier(gl.ALL_BARRIER_BITS);

			// printSSBO(ssboTestI, Int32Array, 0, 10);
			// printSSBO(ssboTestU, Uint32Array, 0, 10);
			// printSSBO(ssboBufferDescriptor, Uint32Array, 0, 3);
			// printSSBO(ssboPosition, Float32Array, 0, 12);

			log(`[end]: ${now() - tStart}s`);

			{
				let pc = new PointCloudLZ("test");
				let s = 1.0;
				pc.world.elements.set([
					s, 0, 0, 0, 
					0, 0, -s, 0, 
					0, s, 0, 0, 
					0, 0, 1, 1, 
				]);

				pc.setVboData(numPoints, ssboPosition, ssboColor);

				scene.root.add(pc);
			}

			// [ 9] xyz:  -7.725, 2.133, 1.831   rgb:   58,  55,  50


			// [ 0] XYZ:  -7.817, 2.194, 1.816   rgb:  154, 146, 135
			// [ 1] xyz:  -7.795, 2.185, 1.813   rgb:  102,  92,  86
			// [ 2] xyz:  -7.777, 2.109, 1.819   rgb:  104, 103,  98
			// [ 3] xyz:  -7.765, 2.185, 1.815   rgb:   86,  83,  78
			// [ 4] xyz:  -7.757, 2.052, 1.799   rgb:  120, 107,  92
			// [ 5] xyz:  -7.75,  2.176, 1.897   rgb:  126, 115,  98
			// [ 6] xyz:  -7.743, 2.149, 1.874   rgb:   84,  77,  66
			// [ 7] xyz:  -7.738, 2.042, 1.807   rgb:  115, 111, 104
			// [ 8] xyz:  -7.732, 2.142, 1.827   rgb:   43,  43,  38
			// [ 9] xyz:  -7.725, 2.133, 1.831   rgb:   58,  55,  50
			// [10] xyz:  -7.722, 2.14,   1.85   rgb:   65,  60,  55
			// [11] xyz:  -7.714, 2.119, 1.863   rgb:   83,  80,  67
			// [12] xyz:  -7.711, 2.135, 1.861   rgb:   74,  70,  59
			// [13] xyz:  -7.706, 2.102, 1.819   rgb:   51,  49,  45
			// [14] xyz:  -7.703, 2.1,   1.813   rgb:   98,  95,  88
			// [15] xyz:  -7.701, 2.135, 1.891   rgb:  103,  93,  78
			// [16] xyz:  -7.699, 2.177, 1.984   rgb:   90,  86,  79
			// [17] xyz:  -7.698, 2.177, 1.926   rgb:   86,  80,  67
			// [18] xyz:  -7.691, 2.166, 1.945   rgb:  105,  94,  85
			// [19] xyz:  -7.69,  2.192, 1.998   rgb:  104, 102,  98

		}
	}
}
