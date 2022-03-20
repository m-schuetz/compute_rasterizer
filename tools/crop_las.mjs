
import * as fs from "fs";


let path = "F:/temp/wgtest/niederweiden/pointcloud.las";
let targetPath = "F:/temp/wgtest/niederweiden/cropped.las";
let croppedSize = 1_000_000_000;

let fd_source = fs.openSync(path, "r");
let fd_target = fs.openSync(targetPath, "w");

let headerBuffer = Buffer.alloc(375);
fs.readSync(fd_source, headerBuffer, 0, 375, 0);

let versionMinor = headerBuffer.readUInt8(25);
let offsetToPointData = headerBuffer.readUInt32LE(96);
let pointRecordLength = headerBuffer.readUInt16LE(105);

let numPoints;
if(versionMinor < 4){
	numPoints = Math.min(headerBuffer.readUInt32LE(107), croppedSize);
	headerBuffer.writeUint32LE(croppedSize, 107);
}else{
	numPoints = Math.min(headerBuffer.readBigUInt64LE(247), croppedSize);
	headerBuffer.writeBigUInt64LE(croppedSize, 247);
}

fs.writeSync(fd_target, headerBuffer);


let MAX_BYTES_PER_BATCH = 10_000_000;
let numBytesLeft = offsetToPointData + numPoints * pointRecordLength;
let numBytesProcessed = headerBuffer.byteLength;

let buffer = Buffer.alloc(MAX_BYTES_PER_BATCH);
while(numBytesLeft > 0){
	let numBytesInBatch = Math.min(numBytesLeft, MAX_BYTES_PER_BATCH);

	fs.readSync(fd_source, buffer, 0, numBytesInBatch, numBytesProcessed);

	fs.writeSync(fd_target, buffer, 0, numBytesInBatch);

	numBytesLeft -= numBytesInBatch;
	numBytesProcessed += numBytesInBatch;
}
