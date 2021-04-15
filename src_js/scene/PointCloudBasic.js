


if(typeof PointCloudBasic === "undefined"){

	PointCloudBasic = class PointCloudBasic extends SceneNode{

		constructor(name, path){
			super(name);

			this.path = path;

			{ // normal point cloud material 
				let vsPath = "../../resources/shaders/pointcloud_basic.vs";
				let fsPath = "../../resources/shaders/pointcloud.fs";
				
				let shader = new Shader([
					{type: gl.VERTEX_SHADER, path: vsPath},
					{type: gl.FRAGMENT_SHADER, path: fsPath},
				]);
				shader.watch();

				let material = new GLMaterial();
				material.shader = shader;
				this.components.push(material);
			}

			this.load();
		}

		async load(){

			let loadStart = now();

			let file = openFile(this.path);

			if(!file){
				log(`could not open file: ${this.path}`);

				return;
			}

			let headerSize = 227;

			let headerBuffer = await file.readBytes(headerSize);

			let headerView = new DataView(headerBuffer);

			// 4	
			// 2	
			// 2	
			// 4	
			// 2	
			// 2	
			// 8	
			// 1	
			// 1	
			// 32	
			// 32	
			// 2	
			// 2	
			// 2	 Header Size
			// 4	 Offset to Point Data

			let offsetToPointData = headerView.getUint32(96, true);

			// 4	 num var length records
			// 1	 point data format
			let pointDataFormat = headerView.getUint8(104);

			// 2
			let pointDataRecordLength = headerView.getUint16(105, true);

			// 4
			let numPoints = headerView.getUint32(107, true);
			numPoints = Math.min(numPoints, 120 * 1000 * 1000);

			// 20

			// 3x8 scale factors
			let sx = headerView.getFloat64(131, true);
			let sy = headerView.getFloat64(139, true);
			let sz = headerView.getFloat64(147, true);

			// 3x8 offsets
			let ox = headerView.getFloat64(155, true);
			let oy = headerView.getFloat64(163, true);
			let oz = headerView.getFloat64(171, true);

			//let bytesPerPoint = 16;
			//let vbo = new ArrayBuffer(numPoints * bytesPerPoint);
			//let vboF32 = new Float32Array(vbo);
			//let vboU8 = new Uint8Array(vbo);


			let bytesPerPoint = 0;


			let glbuffer = new GLBuffer();
			{
				let attributes = [
					new GLBufferAttribute("position", 0, 3, gl.FLOAT, gl.FALSE, 12, 0),
					new GLBufferAttribute("color_orig", 1, 4, gl.UNSIGNED_BYTE, gl.TRUE, 4, 12),
					//new GLBufferAttribute("random", 2, 1, gl.FLOAT, gl.FALSE, 4, 16),
					//new GLBufferAttribute("color_avg", 2, 4, gl.UNSIGNED_BYTE, gl.TRUE, 4, 16),
					//new GLBufferAttribute("color", 1, 4, gl.UNSIGNED_BYTE, gl.TRUE, 4, 12),
					//new GLBufferAttribute("acc", 3, 4, gl.FLOAT, gl.FALSE, 16, 20),
				];

				glbuffer.attributes = attributes;

				bytesPerPoint = attributes.reduce( (p, c) => p + c.bytes, 0);

				gl.bindVertexArray(glbuffer.vao);
				gl.bindBuffer(gl.ARRAY_BUFFER, glbuffer.vbo);
				gl.bufferData(gl.ARRAY_BUFFER, numPoints * bytesPerPoint, 0, gl.DYNAMIC_DRAW);

				let stride = attributes.reduce( (a, v) => a + v.bytes, 0);

				for(let attribute of attributes){
					gl.enableVertexAttribArray(attribute.location);
					gl.vertexAttribPointer(
						attribute.location, 
						attribute.count, 
						attribute.type, 
						attribute.normalize, 
						stride, 
						attribute.offset);	
				}

				gl.bindVertexArray(0);

				glbuffer.count = numPoints;
			}
			this.components.push(glbuffer);

			let rgbOffset = pointDataFormat == 2 ? 20 : 28;

			let start = now();

			await file.setReadLocation(offsetToPointData);

			let pointsPerChunk = 0.1 * 1000 * 1000;
			let source = await file.readBytes(pointsPerChunk * pointDataRecordLength);
			let sourceU8 = new Uint8Array(source);
			let sourceView = new DataView(source);

			let vboChunk = new ArrayBuffer(pointsPerChunk * bytesPerPoint);
			let vboChunkU8 = new Uint8Array(vboChunk);
			let vboChunkF32 = new Float32Array(vboChunk);
			let vboChunkView = new DataView(vboChunk);


			let tmp = new ArrayBuffer(4);
			let tmpU8 = new Uint8Array(tmp);
			let tmpU16 = new Uint16Array(tmp);
			let tmpI32 = new Int32Array(tmp);

			let i_local = 0;
			for(let i = 0; i < numPoints; i++){

				let offsetSource = i_local * pointDataRecordLength;

				// USING DATA VIEW
				// let ux = sourceView.getInt32(offsetSource + 0, true);
				// let uy = sourceView.getInt32(offsetSource + 4, true);
				// let uz = sourceView.getInt32(offsetSource + 8, true);

				// USING U8 ARRAY HACK
				tmpU8[0] = sourceU8[offsetSource + 0];
				tmpU8[1] = sourceU8[offsetSource + 1];
				tmpU8[2] = sourceU8[offsetSource + 2];
				tmpU8[3] = sourceU8[offsetSource + 3];
				let ux = tmpI32[0];

				tmpU8[0] = sourceU8[offsetSource + 4];
				tmpU8[1] = sourceU8[offsetSource + 5];
				tmpU8[2] = sourceU8[offsetSource + 6];
				tmpU8[3] = sourceU8[offsetSource + 7];
				let uy = tmpI32[0];

				tmpU8[0] = sourceU8[offsetSource + 8];
				tmpU8[1] = sourceU8[offsetSource + 9];
				tmpU8[2] = sourceU8[offsetSource + 10];
				tmpU8[3] = sourceU8[offsetSource + 11];
				let uz = tmpI32[0];


				// USING DATA VIEW
				//let ur = sourceView.getUint16(offsetSource + rgbOffset + 0, true);
				// let ug = sourceView.getUint16(offsetSource + rgbOffset + 2, true);
				// let ub = sourceView.getUint16(offsetSource + rgbOffset + 4, true);

				// USING U8 ARRAY HACK
				tmpU8[0] = sourceU8[offsetSource + rgbOffset + 0];
				tmpU8[1] = sourceU8[offsetSource + rgbOffset + 1];
				let ur = tmpU16[0];

				tmpU8[0] = sourceU8[offsetSource + rgbOffset + 2];
				tmpU8[1] = sourceU8[offsetSource + rgbOffset + 3];
				let ug = tmpU16[0];

				tmpU8[0] = sourceU8[offsetSource + rgbOffset + 4];
				tmpU8[1] = sourceU8[offsetSource + rgbOffset + 5];
				let ub = tmpU16[0];

				let x = ux * sx;
				let y = uy * sy;
				let z = uz * sz;

				let r = ur / 256;
				let g = ug / 256;
				let b = ub / 256;

				// USING ARRAY HACK
				vboChunkF32[i_local * bytesPerPoint / 4 + 0] = x;
				vboChunkF32[i_local * bytesPerPoint / 4 + 1] = y;
				vboChunkF32[i_local * bytesPerPoint / 4 + 2] = z;

				// USING DATA VIEW
				//vboChunkView.setFloat32(bytesPerPoint * i_local + 0, x, true);
				//vboChunkView.setFloat32(bytesPerPoint * i_local + 4, y, true);
				//vboChunkView.setFloat32(bytesPerPoint * i_local + 8, z, true);

				vboChunkU8[bytesPerPoint * i_local + 12] = r;
				vboChunkU8[bytesPerPoint * i_local + 13] = g;
				vboChunkU8[bytesPerPoint * i_local + 14] = b;
				vboChunkU8[bytesPerPoint * i_local + 15] = 255;

				i_local++;

				if(i_local === pointsPerChunk || (i + 1) === numPoints){

					let end = now();
					let duration = end - start;
					log(`load chunk: ${duration.toFixed(3)}s`);

					let vboOffset = (i + 1 - i_local) * bytesPerPoint;
					let vboChunkSize = i_local * bytesPerPoint;
					gl.namedBufferSubData(glbuffer.vbo, vboOffset, vboChunkSize, vboChunk);

					//log(`======`);
					//log(`i: ${i}`);
					//log(`i_local: ${i_local}`);
					//log(`vboOffset: ${vboOffset}`);
					//log(`vboChunkSize: ${vboChunkSize}`);

					let moreToRead = (i + 1) < numPoints;
					if(moreToRead){
						let rbStart = now();
						source = await file.readBytes(pointsPerChunk * pointDataRecordLength);
						let rbEnd = now();
						let rbDuration = rbEnd - rbStart;
						log(`rbDuration: ${rbDuration.toFixed(3)}s`);

						sourceView = new DataView(source);
						sourceU8 = new Uint8Array(source);

						i_local = 0;

						start = now();
					}

				}
			}

			

			
			log(10);

			file.close();
			log(20);

			let loadEnd = now();
			let loadDuration = loadEnd - loadStart;
			log(`loadDuration: ${loadDuration.toFixed(3)}s`);
			
		}

	}
}


"PointCloudBasic.js"
