const fs = require('fs');

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
};

exports.LASHeader = LASHeader;
exports.parseHeader = parseHeader;