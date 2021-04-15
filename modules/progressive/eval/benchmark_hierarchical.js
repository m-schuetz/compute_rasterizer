
loadOctree = function(){
	
	let pc = new PointCloudOctree("octree", "D:/dev/pointclouds/archpro/heidentor.las_converted/cloud.js");
	//let pc = new PointCloudOctree("octree", "D:/dev/pointclouds/converted/affandi_batch_1/cloud.js");
	
	//heidentor.transform.elements.set([
	//	1, 0, 0, 0, 
	//	0, 0, -1, 0, 
	//	0, 1, 0, 0, 
	//	0, 0, 0, 1, 
	//]);
	let s = 1.0;
	pc.transform.elements.set([
		s, 0, 0, 0, 
		0, 0, -s, 0, 
		0, s, 0, 0, 
		0, 0, 0, 1, 
	]);
	scene.root.add(pc);

	view.set(
		[318.4550957504863, 99.92744790028979, -363.3676528807173 ],
		[359.86661645472486, 67.79371994307411, -365.0374069252646]
	);
}

loadOctree();

// {
// 	let node = $("heidentor_oct");

// 	log(node.root.children[0]);

// }