

if(typeof e4called === "undefined"){
	e4called = true;
	
	//let las = loadLASProgressive("D:/dev/pointclouds/archpro/heidentor.las");
	//let las = loadLASProgressive("D:/dev/pointclouds/NVIDIA/photogrammetry/test.las");

	//let las = loadBINProgressive("D:/dev/pointclouds/riegl/niederweiden_400m.bin");
	//let las = loadLASProgressive("D:/dev/pointclouds/tuwien_baugeschichte/Museum Affandi_las export/batch_1.las");
	//let las = loadLASProgressive("D:/dev/pointclouds/tuwien_baugeschichte/Kirchenburg Arbegen/batch_0.las");
	//let las = loadLASProgressive("D:/dev/pointclouds/NVIDIA/laserscans/merged.las");
	//let las = loadLASProgressive("D:/dev/pointclouds/hofbibliothek/HB_64.las");
	//let las = loadBINProgressive("D:/dev/pointclouds/test.bin");

	//let las = loadLASProgressive("D:/dev/pointclouds/archpro/heidentor.las");
	//let las = loadLASProgressive("D:/dev/pointclouds/mschuetz/lion.las");
	//let las = loadLASProgressive("D:/dev/pointclouds/tu_photogrammetry/wienCity_v5_250k.las");
	//let las = loadLASProgressive("D:/dev/pointclouds/tu_photogrammetry/wienCity_v3.las");
	//let las = loadLASProgressive("D:/dev/pointclouds/weiss/pos7_Subsea_equipment.las");

	//let las = loadLASProgressive("D:/dev/pointclouds/tuwien_baugeschichte/Candi Sari_las export/candi_sari.las");


	let pc = new PointCloudProgressive("testcloud", "blabla");
	//pc.boundingBox.min.set(...las.boundingBox.min);
	//pc.boundingBox.max.set(...las.boundingBox.max);

	log(pc.boundingBox);

	let handles = las.handles;

	let attributes = [
		new GLBufferAttribute("position", 0, 3, gl.FLOAT, gl.FALSE, 12, 0),
		//new GLBufferAttribute("color",    1, 4, gl.UNSIGNED_BYTE, gl.TRUE, 4, 12),
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
		s, 0, 0, 0, 
		0, 0, -s, 0, 
		0, s, 0, 0, 
		-10, 1.4, -11, 1, 
	]);

	scene.root.add(pc);

	listeners.update.push(() => {

		if(pc.numPoints !== las.numPoints){
			//log(las.numPoints);
		}
		pc.numPoints = las.numPoints;
		
	});

}

view.set(
	[-12.362381738151846, 4.05480067990359, -15.238182318876296 ],
	[-8.091633605894913, 1.9830766937417055, -14.241735502512483],
);

camera.fov = 100;

// log($("testcloud"));

// if($("testcloud")){
// 	let pc = $("testcloud");
// 	let s = 0.03;
// 	pc.transform.elements.set([
// 		s, 0, 0, 0, 
// 		0, 0, s, 0, 
// 		0, s, 0, 0, 
// 		-0.5, 0.7, 1, 1, 
// 	]);

// 	log("lala");

// }