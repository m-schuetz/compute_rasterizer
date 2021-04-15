
if(typeof OBJLoader === "undefined"){
	OBJLoader = class OBJLoader{

		constructor(){

		}

	};
}

// poor mans OBJ loader
OBJLoader.load = function(file){
	let objtxt = readTextFile(file);

	let lines = objtxt.split("\n");


	let sPosition = [];
	let sUV = [];
	let sNormal = [];

	let tPosition = [];
	let tUV = [];
	let tNormal = [];

	for(let line of lines){
		if(line.startsWith("v ")){
			let tokens = line.split(" ");

			let x = parseFloat(tokens[1]);
			let y = parseFloat(tokens[2]);
			let z = parseFloat(tokens[3]);

			sPosition.push(x, y, z);
		}else if(line.startsWith("vt ")){
			let tokens = line.split(" ");

			let u = parseFloat(tokens[1]);
			let v = parseFloat(tokens[2]);

			sUV.push(u, v);
		}else if(line.startsWith("vn ")){
			let tokens = line.split(" ");

			let x = parseFloat(tokens[1]);
			let y = parseFloat(tokens[2]);
			let z = parseFloat(tokens[3]);

			sNormal.push(x, y, z);
		}else if(line.startsWith("f ")){
			let tokens = line.split(" ");

			let ti = [1, 2, 3];
			if(tokens.length > 4){
				for(let i = 4; i < tokens.length; i++){
					ti.push(1, i - 1, i);
				}
			}

			for(let i of ti){
				let face = tokens[i];
				let faceIndices = face.split("/");

				let iPosition = parseInt(faceIndices[0]) - 1;
				let iUV = parseInt(faceIndices[1]) - 1;
				let iNormal = parseInt(faceIndices[2]) - 1;

				let x = sPosition[3 * iPosition + 0];
				let y = sPosition[3 * iPosition + 1];
				let z = sPosition[3 * iPosition + 2];

				let u = sUV[2 * iUV + 0];
				let v = sUV[2 * iUV + 1];

				let nx = sNormal[3 * iNormal + 0];
				let ny = sNormal[3 * iNormal + 1];
				let nz = sNormal[3 * iNormal + 2];

				tPosition.push(x, y, z);
				tUV.push(u, v);
				tNormal.push(nx, ny, nz);
			}
		}
	}

	let position = new Float32Array(tPosition);
	let uv = new Float32Array(tUV);
	let normal = new Float32Array(tNormal);

	return {
		position: position,
		uv: uv,
		normal: normal,
		count: position.length / 3,
	}

};

"OBJLoader.js"