
const fs = require("fs");




let fd = fs.openSync("F:/temp/wgtest/retz/pointcloud.las");


{
	let target = Buffer.alloc(400);
	fs.readSync(fd, target, 0, 400, 0);

	// console.log(target.readUInt8(0));
	// console.log(target.readUInt8(1));
	// console.log(target.readUInt8(2));
	// console.log(target.readUInt8(3));
	console.log(target.readInt32LE(1));
}




// {
// 	let target = Buffer.alloc(4);
// 	fs.readSync(fd, target, 0, 4, 0);

// 	console.log(target.readInt32LE(0));
// }

// {
// 	let target = Buffer.alloc(4);
// 	fs.readSync(fd, target, 0, 4, 1);

// 	console.log(target.readInt32LE(0));
// }

// {
// 	let target = Buffer.alloc(4);
// 	fs.readSync(fd, target, 0, 4, 313);

// 	console.log(target.readInt32LE(0));
// }

