
const fs = require('fs');
const fpath = require("path");
const {LASHeader, parseHeader} = require("./LASHeader");


const dir = "D:/dev/pointclouds/open_topography/ca13/laz";

const files = fs.readdirSync(dir);
const paths = files.map(f => `${dir}/${f}`);

function sumarize(paths){

	let totalPoints = 0;
	let min = [Infinity, Infinity, Infinity];
	let max = [-Infinity, -Infinity, -Infinity];

	for(const path of paths){	
		const header = parseHeader(path);

		min = [
			Math.min(min[0], header.min[0]),
			Math.min(min[1], header.min[1]),
			Math.min(min[2], header.min[2]),
		];

		max = [
			Math.max(max[0], header.max[0]),
			Math.max(max[1], header.max[1]),
			Math.max(max[2], header.max[2]),
		];

		totalPoints += header.numberOfPoints;
	}

	console.log(`numPoints: ${totalPoints.toLocaleString()}`);
	console.log(min);
	console.log(max);
}

sumarize(paths);


function filesInBox(paths, box){

	const inside = [];

	const intersects = (box1, box2) => {

		const isLeft = box1.max[0] < box2.min[0];
		const isRight = box1.min[0] > box2.max[0];
		const isDown = box1.max[1] < box2.min[1];
		const isUp = box1.min[1] > box2.max[1];

		//console.log("===");
		//console.log(1, box1, 2, box2);
		//console.log(isLeft, isRight, isUp, isDown);

		const isOutside = isLeft || isRight || isDown || isUp;

		return !isOutside;
	};

	for(const path of paths){
		const header = parseHeader(path);
		const lasBox = {
			min: header.min,
			max: header.max,
		};

		if(intersects(lasBox, box)){
			inside.push(path);
		}

	}

	console.log(inside.length);

	sumarize(inside);

	let filenames = inside.map(p => fpath.basename(p));
	const exceptions = ["ot_35120D7408C_1.laz", "ot_35120D7413A_1.laz"];
	filenames = filenames.filter(file => !exceptions.includes(file));


	let filePart = filenames.join(" ");
	let command = `lasinfo -i ${filePart} -gui`;
	console.log(command);

}


{
	let box = {
		min: [ 692947 - 200, 3914489 - 200, -2.72 ],
		max: [ 699903 + 1000, 3921011 + 500, 286.06 ],
	};

	filesInBox(paths, box);
}
