

export class Bitstream{

	constructor(bytesize){
		this.buffer = Buffer.alloc(bytesize);

		this.bitsWritten = 0;
		this.currentValue = 0;

	}

	write(value, bits){

		if(bits < 0 || bits > 32){
			throw "invalid bit size";
		}

		let bitOffset = this.bitsWritten;

		let blockIndex = Math.floor(bitOffset / 32);
		let blockBitsWritten =  (bitOffset % 32);
		let blockBitsLeft = 32 - blockBitsWritten;

		let allowance = Math.min(blockBitsLeft, bits);
		let remainingBits = bits - allowance;
		let writeStartsNewBlock = blockBitsWritten + bits >= 32;

		let mask = (2 ** allowance) - 1;
		let newBlockValue = this.currentValue | ((value & mask) << blockBitsWritten);
		this.buffer.writeInt32LE(newBlockValue, 4 * blockIndex);
		this.currentValue = newBlockValue;
		this.bitsWritten += allowance;

		// new block started, reset current value
		if(writeStartsNewBlock){
			this.currentValue = 0;
		}

		if(remainingBits > 0){

			if(remainingBits >= bits){
				throw "wat";
			}

			let remainingValue = value >> allowance;
			this.write(remainingValue, remainingBits);
		}
	}

	read(bitOffset, bitSize){

		let blockIndex = Math.floor(bitOffset / 32);
		let blockLocalOffset = bitOffset % 32;
		let blockBitsRemaining = 32 - blockLocalOffset;
		let valueBitsInBlock = Math.min(bitSize, blockBitsRemaining);
		let valueBitsRemaining = bitSize - valueBitsInBlock;

		let blockValue = this.buffer.readInt32LE(4 * blockIndex);
		let mask = (2 ** valueBitsInBlock) - 1;
		let value = (blockValue >> blockLocalOffset) & mask;

		if(valueBitsRemaining > 0){
			// let remainingValue = this.read(bitOffset + valueBitsInBlock, valueBitsRemaining);
			let remainderMask = (2 ** valueBitsRemaining) - 1;
			let remainingValue = this.buffer.readInt32LE(4 * blockIndex + 4) & remainderMask;

			value = (remainingValue << valueBitsInBlock) | value;
		}

		return value;
	}
}


// function test1(){
// 	let n = 8;
// 	let stream = new Bitstream(4 * n);

// 	for(let i = 0; i < n; i++){

// 		// let value = Math.random() * (2 ** 32);

// 		// value = value & 0x0FFFFFFF;

// 		let value = 1;

// 		stream.write(value, 27);

// 	}


// 	let str = "";
// 	for(let i = 0; i < 4 * n; i++){
// 		str = str + stream.buffer[i].toString(2).padStart(8, "0") + " ";
// 	}
// 	console.log(str)
// }

// function test2(){

// 	let n = 1000;
// 	let stream = new Bitstream(10000);

// 	let values = [];
// 	for(let i = 0; i < n; i++){
// 		let bits = Math.floor(Math.random() * 32) + 1;
// 		let mask = (2 ** bits) - 1;
// 		let value = Math.floor(Math.random() * (2 ** 32)) & mask;

// 		stream.write(value, bits);
// 		values.push({value, bits});
// 	}

// 	// let values = [
// 	// 	{value: 13, bits: 10},
// 	// 	{value: 7, bits: 5},
// 	// 	{value: 11, bits: 6},
// 	// 	{value: 234, bits: 17},
// 	// 	{value: 65, bits: 12},
// 	// ];

// 	// for(let item of values){
// 	// 	stream.write(item.value, item.bits);
// 	// }


// 	let bitOffset = 0;
// 	for(let i = 0; i < values.length; i++){
// 		let {bits} = values[i];

// 		let value = stream.read(bitOffset, bits);
// 		let referenceValue = values[i].value;

// 		if(value !== referenceValue){
// 			throw "damn";
// 		}

// 		console.log(value, referenceValue, bits);

// 		bitOffset += bits;
// 	}


// }


// test2();






