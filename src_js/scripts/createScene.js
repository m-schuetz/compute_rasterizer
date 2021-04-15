

{

	let brushNode = new BrushNode("test_brush");
	let stride = brushNode.buffer.stride;

	let n = 100;
	let data = new ArrayBuffer(n * stride);
	let view = new DataView(data);

	for(let i = 0; i < n; i++){

		let x = Math.random();
		let y = Math.random();
		let z = Math.random();

		let size = 5.0;

		view.setFloat32(stride * i + 0, x, true);
		view.setFloat32(stride * i + 4, y, true);
		view.setFloat32(stride * i + 8, z, true);

		view.setFloat32(stride * i + 24, size, true);

	}


	brushNode.addData(data);

	scene.root.add(brushNode);

}

