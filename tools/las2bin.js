
const fs = require('fs');



//let file = "D:/dev/pointclouds/Riegl/niederweiden.las";
//let targetFile = "D:/dev/pointclouds/Riegl/test.bin";

// let file = "D:/dev/pointclouds/archpro/heidentor.las";
// let targetFile = "D:/dev/pointclouds/archpro/heidentor.bin";
// let offsetXYZ = 0;
// let offsetRGB = 20;

//RETZ
// let file = "D:/dev/pointclouds/riegl/Retz_Airborne_Terrestrial_Combined_1cm.las";
// let targetFile = "D:/dev/pointclouds/riegl/retz.bin";
// let offsetXYZ = 0;
// let offsetRGB = 28;

//Wien v6 250M
// let file = "D:/dev/pointclouds/tu_photogrammetry/wienCity_v6_250M.las";
// let targetFile = "D:/dev/pointclouds/tu_photogrammetry/wienCity_v6_250M.bin";
// let offsetXYZ = 0;
// let offsetRGB = 30;

// Wien v6 125
// let file = "D:/dev/pointclouds/tu_photogrammetry/wienCity_v6_125M.las";
// let targetFile = "D:/dev/pointclouds/tu_photogrammetry/wien_v6_125.bin";
// let offsetXYZ = 0;
// let offsetRGB = 30;

// Wien v6 350 (405M)
// let file = "E:/pointclouds/tuwien_photogrammetry/wienCity_v6_350M.las";
// let targetFile = "D:/dev/pointclouds/tu_photogrammetry/wien_v6_350M.bin";
// let offsetXYZ = 0;
// let offsetRGB = 30;

// Wien v6 500 
// let file = "D:/dev/pointclouds/tu_photogrammetry/wienCity_v6_500M.las";
// let targetFile = "D:/dev/pointclouds/tu_photogrammetry/wienCity_v6_500M.bin";
// let offsetXYZ = 0;
// let offsetRGB = 30;

// MATTERHORN
// let file = "D:/dev/pointclouds/pix4d/matterhorn.las";
// let targetFile = "D:/dev/pointclouds/pix4d/matterhorn.bin";
// let offsetXYZ = 0;
// let offsetRGB = 20;

// MATTERHORN
// let file = "D:/dev/pointclouds/pix4d/eclepens.las";
// let targetFile = "D:/dev/pointclouds/pix4d/eclepens.bin";
// let offsetXYZ = 0;
// let offsetRGB = 20;

// MATTERHORN
let file = "D:\\dev\\pointclouds\\tuwien_baugeschichte\\Museum Affandi_las export\\batch_0.las";
let targetFile = "D:\\dev\\pointclouds\\tuwien_baugeschichte\\Museum Affandi_las export\\batch_0.bin";
let offsetXYZ = 0;
let offsetRGB = 20;

// CANDI SARI
// let file = "D:/dev/pointclouds/tuwien_baugeschichte/Candi Sari_las export/candi_sari.las";
// let targetFile = "D:/dev/pointclouds/tuwien_baugeschichte/Candi Sari_las export/candi_sari.bin";
// let offsetXYZ = 0;
// let offsetRGB = 20;

// MORRO BAY
// let file = "D:/dev/pointclouds/open_topography/ca13/morro_bay.las";
// let targetFile = "D:/dev/pointclouds/open_topography/ca13/morro_bay.bin";
// let offsetXYZ = 0;
// let offsetRGB = 28;

// MORRO BAY 1 BILLION
// let file = "D:/dev/pointclouds/open_topography/ca13/morro_bay_1billion.las";
// let targetFile = "D:/dev/pointclouds/open_topography/ca13/morro_bay_1billion.bin";
// let offsetXYZ = 0;
// let offsetRGB = 28;

class LASHeader{

	constructor(){
		this.versionMajor = 0;
		this.versionMinor = 0;

		this.headerSize = 0;
		this.offsetToPointData = 0;
		this.numberOfVariableLengthRecords = 0;
		this.pointDataFormat = 0;
		this.pointDataRecordLength = 0;
		this.numberOfPoints = 0;
		this.scale = null;
		this.offset = null;
		this.min = null;
		this.max = null;
	}

};

function parseHeader(file){
	let source = Buffer.alloc(375);
	let fd = fs.openSync(file);
	fs.readSync(fd, source, 0, source.length, 0);

	const header = new LASHeader();

	header.versionMajor = source.readUInt8(24);
	header.versionMinor = source.readUInt8(25);
	header.headerSize = source.readUInt16LE(94);
	header.offsetToPointData = source.readUInt32LE(96);
	header.numberOfVariableLengthRecords = source.readUInt32LE(100);
	header.pointDataFormat = source.readUInt8(104);
	header.pointDataRecordLength = source.readUInt16LE(105);

	if(header.versionMajor === 1 && header.versionMinor < 4){
		header.numberOfPoints = source.readUInt32LE(107);
	}else{
		header.numberOfPoints = parseInt(source.readBigUInt64LE(247));
	}

	header.scale = [
		source.readDoubleLE(131),
		source.readDoubleLE(139),
		source.readDoubleLE(147),
	];

	header.offset = [
		source.readDoubleLE(155),
		source.readDoubleLE(163),
		source.readDoubleLE(171),
	];

	header.min = [
		source.readDoubleLE(187),
		source.readDoubleLE(203),
		source.readDoubleLE(219),
	];

	header.max = [
		source.readDoubleLE(179),
		source.readDoubleLE(195),
		source.readDoubleLE(211),
	];


	return header;
}

const header = parseHeader(file);

let fd = fs.openSync(file);
let fo = fs.openSync(targetFile, "w");

//let pointsLeft = 200 * 1000 * 1000;
let pointsLeft = Math.min(header.numberOfPoints, 300_000_000);
let batchSize = 1000 * 1000;
let pointsRead = 0;
const bytesPerPoint = header.pointDataRecordLength;
const offsetToPointData = header.offsetToPointData;
const {scale, offset, min} = header;

let colorFactor = null;

while(pointsLeft > 0){

	let batch = Math.min(batchSize, pointsLeft);

	let source = Buffer.alloc(batch * bytesPerPoint);
	let target = Buffer.alloc(batch * 16);

	fs.readSync(fd, source, 0, source.length, offsetToPointData + pointsRead * bytesPerPoint);

	if(!colorFactor){

		colorFactor = 1;

		for(let i = 0; i < batch; i++){
			let pOffset = i * bytesPerPoint;

			let r = parseInt(source.readUInt16LE(pOffset + offsetRGB + 0));
			let g = parseInt(source.readUInt16LE(pOffset + offsetRGB + 2));
			let b = parseInt(source.readUInt16LE(pOffset + offsetRGB + 4));

			if(r > 255 || g > 255 || b > 255){
				colorFactor = 256;
			}
		}
	}

	for(let i = 0; i < batch; i++){
		let pOffset = i * bytesPerPoint;

		let ux = source.readInt32LE(pOffset + offsetXYZ + 0);
		let uy = source.readInt32LE(pOffset + offsetXYZ + 4);
		let uz = source.readInt32LE(pOffset + offsetXYZ + 8);

		let x = ux * scale[0] + offset[0] - min[0];
		let y = uy * scale[1] + offset[1] - min[1];
		let z = uz * scale[2] + offset[2] - min[2];

		// let r = parseInt(source.readUInt16LE(pOffset + offsetRGB + 0));
		// let g = parseInt(source.readUInt16LE(pOffset + offsetRGB + 2));
		// let b = parseInt(source.readUInt16LE(pOffset + offsetRGB + 4));

		let r = parseInt(source.readUInt16LE(pOffset + offsetRGB + 0) / colorFactor);
		let g = parseInt(source.readUInt16LE(pOffset + offsetRGB + 2) / colorFactor);
		let b = parseInt(source.readUInt16LE(pOffset + offsetRGB + 4) / colorFactor);

	   target.writeFloatLE(x, 16 * i + 0);
	   target.writeFloatLE(y, 16 * i + 4);
	   target.writeFloatLE(z, 16 * i + 8);
	   target.writeUInt8(r, 16 * i + 12);
	   target.writeUInt8(g, 16 * i + 13);
	   target.writeUInt8(b, 16 * i + 14);

	}
	
	fs.writeSync(fo, target);

	console.log(`pointsLeft: ${pointsLeft}`);
	pointsLeft -= batch;
	pointsRead += batch;
}



