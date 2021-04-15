

const AC__MinLength = 0x01000000;
const AC__MaxLength = 0xFFFFFFFF;

const BM__LengthShift = 13;
const BM__MaxCount    = 1 << BM__LengthShift; 

const DM__LengthShift = 15;
const DM__MaxCount    = 1 << DM__LengthShift;

const I32_MIN = -2147483648
const I32_MAX =  2147483647

let asufeg = undefined;

 class ArithmeticModel{
	constructor(symbols, compress){

		this.symbols = symbols;
		this.compress = compress;
		this.distribution = null;

		this.symbol_count = null;
		this.last_symbol = null;
		this.decoder_table = null;
		this.table_size = null;
		this.table_shift = null;
		this.total_count = null;
		this.update_cycle = null;
		this.symbols_until_update = null;

		this.init();
	}

	init(){

		if(this.distribution == null){
			this.last_symbol = this.symbols - 1;

			if(!this.compress && this.symbols > 16){
				this.table_size = 1 << (Math.ceil(Math.log2(this.symbols)) - 2);
				let table_bits = Math.log2(this.table_size);

				this.table_shift = DM__LengthShift - table_bits;
				let distributionSize = 2 * this.symbols + this.table_size + 2;
				this.distribution = new Uint32Array(distributionSize);
				this.decoder_table = 2 * this.symbols;
			}else{
				this.decoder_table = 0;
				this.table_size = this.table_shift = 0;
				this.distribution = new Uint32Array(2 * this.symbols);
			}

			if(typeof asufeg === "undefined"){
				asufeg = new Map();
			}
			if(!asufeg.has(this.symbols)){
				
				console.log("symbols", this.symbols, "distribution", this.distribution.length);
				asufeg.set(this.symbols, this.distribution.length);
			}

			this.symbol_count = this.symbols;
		}else{
			debugger;
		}

		this.total_count = 0;
		this.update_cycle = this.symbols;

		for (let k = 0; k < this.symbols; k++){
			this.distribution[this.symbol_count + k] = 1;
		}

		this.update();
		this.update_cycle = (this.symbols + 6) >>> 1;
		this.symbols_until_update = this.update_cycle;
	}

	update(){
		// halve counts when a threshold is reached
		if ((this.total_count += this.update_cycle) > DM__MaxCount){
			this.total_count = 0;
			for (let n = 0; n < this.symbols; n++){
				let index = this.symbol_count + n;
				let newDist = (this.distribution[index] + 1) >>> 1;
				this.distribution[index] = newDist;
				this.total_count += newDist;
			}
		}
		
		// compute cumulative distribution, decoder table
		let sum = 0, s = 0;
		let scale = parseInt(0x80000000 / this.total_count);

		if (this.compress || (this.table_size == 0)){
			for (let k = 0; k < this.symbols; k++){
				this.distribution[k] = (scale * sum) >>> (31 - DM__LengthShift);
				sum += this.distribution[this.symbol_count + k];
			}
		}else{
			for (let k = 0; k < this.symbols; k++){
				this.distribution[k] = (scale * sum) >>> (31 - DM__LengthShift);
				sum += this.distribution[this.symbol_count + k];
				let w = this.distribution[k] >>> this.table_shift;
				while (s < w){
					s++;
					this.distribution[this.decoder_table + s] = k - 1;
					// this.decoder_table[++s] = k - 1;
				}
			}

			this.distribution[this.decoder_table] = 0;
			//this.decoder_table[0] = 0;

			while (s <= this.table_size){
				s++;
				this.distribution[this.decoder_table + s] = this.symbols - 1;
				//this.decoder_table[++s] = this.symbols - 1;
			}
		}
		
		// set frequency of model updates
		this.update_cycle = (5 * this.update_cycle) >>> 2;
		let max_cycle = ((this.symbols + 6) << 3) >>> 0;
		if (this.update_cycle > max_cycle){
			this.update_cycle = max_cycle;
		}
		this.symbols_until_update = this.update_cycle;
	}
}

 class ArithmeticBitModel{
	constructor(){
		this.init();
	}

	init(){
		this.bit_0_count = 1;
		this.bit_count   = 2;
		this.bit_0_prob  = 1 << (BM__LengthShift - 1);

		this.update_cycle = 4;
		this.bits_until_update = 4;
	}

	update(){

		this.bit_count = this.bit_count + this.update_cycle;

		if(this.bit_count > BM__MaxCount){
			this.bit_count = (this.bit_count + 1) >>> 1;
			this.bit_0_count = (this.bit_0_count + 1) >>> 1;
			if(this.bit_0_count === this.bit_count){
				this.bit_count++;
			}
		}

		let scale = parseInt(0x80000000 / this.bit_count);
		this.bit_0_prob = (this.bit_0_count * scale) >>> (31 - BM__LengthShift);

		this.update_cycle = (5 * this.update_cycle) >>> 2;
		if(this.update_cycle > 64){
			this.update_cycle = 64;
		}

		this.bits_until_update = this.update_cycle;

	}
}

 class ArithmeticDecoder{

	constructor(buffer, offset = 0){
		this.buffer = buffer;
		this.bufferOffset = offset;

		this.length = AC__MaxLength;
		this.value = 0;

		// big endian
		// end with >>> 0 so that JS interprets this.value as uint32
		// see https://stackoverflow.com/questions/6798111/bitwise-operations-on-32-bit-unsigned-ints
		this.value = (this.getByte() << 24) >>> 0;
		this.value = (this.value | (this.getByte() << 16)) >>> 0;
		this.value = (this.value | (this.getByte() << 8)) >>> 0;
		this.value = (this.value | (this.getByte() )) >>> 0;

		return true;
	}

	getByte(){
		let value = this.buffer[this.bufferOffset];

		this.bufferOffset++;

		return value;
	}

	decodeSymbol(m){

		let n, sym, x, y = this.length;

		if(m.decoder_table != 0){
			let dv = parseInt(this.value / (this.length >>>= DM__LengthShift));
			let t = dv >>> m.table_shift;

			sym = m.distribution[m.decoder_table + t];
			n = m.distribution[m.decoder_table + t + 1] + 1;

			while (n > sym + 1) {
				let k = (sym + n) >>> 1;
				if (m.distribution[k] > dv){
					n = k;
				}else{ 
					sym = k;
				}
			}

			x = m.distribution[sym] * this.length;
			if (sym != m.last_symbol){
				y = m.distribution[sym + 1] * this.length;
			}
		}else{
			x = 0;
			sym = 0;

			this.length >>>= DM__LengthShift;
			n = m.symbols;

			let k = n >>> 1;

			do {
				let z = this.length * m.distribution[k];
				if (z > this.value) {
					n = k;
					y = z;
				}
				else {
					sym = k;
					x = z;
				}

				k = (sym + n) >>> 1;
			} while (k != sym);
		}

		this.value = this.value - x;
		this.length = y - x;

		if(this.length < AC__MinLength){
			this.renorm_dec_interval();
		}

		m.distribution[m.symbol_count + sym]++;
		m.symbols_until_update--;

		if(m.symbols_until_update === 0){
			m.update();
		}

		return sym;
	}

	decodeBit(m){
		let x = m.bit_0_prob * (this.length >>> BM__LengthShift);
		let sym = (this.value >= x) ? 1 : 0;

		if(sym == 0n){
			this.length = x;
			m.bit_0_count++;
		}else{
			this.value = this.value - x;
			this.length = this.length - x;
		}

		if(this.length < AC__MinLength){
			this.renorm_dec_interval();
		}

		m.bits_until_update--;

		if(m.bits_until_update === 0){
			m.update();
		}

		return sym;
	}

	readBits(bits){

		if(bits > 19){
			throw "not implemented";
		}else{
			let sym = parseInt(this.value / (this.length >>>= bits));
			this.value -= this.length * sym;

			if(this.length < AC__MinLength){
				this.renorm_dec_interval();
			}

			if(sym >= ((1 << bits) >>> 0)){
				throw 4711;
			}

			return sym;
		}

	}

	renorm_dec_interval(){
		
		do{
			this.value = (this.value << 8) >>> 0;
			this.value = (this.value | this.getByte()) >>> 0;

			this.length = (this.length << 8) >>> 0;
		}while(this.length < AC__MinLength);
	}

}

 class IntegerCompressor{

	constructor(dec, bits, contexts, bits_high = 8, range = 0){
		this.dec = dec;
		this.bits = bits;
		this.contexts = contexts;
		this.bits_high = bits_high;

		if(range !== 0){
			this.corr_bits = 0;
			this.corr_range = range;

			while(range !== 0){
				range = range >>> 1;
				this.corr_bits++;
			}

			let cmp = (1 << (this.corr_bits - 1)) >>> 0;
			if (this.corr_range == cmp){
				this.corr_bits--;
			}

			this.corr_min = -parseInt(this.corr_range / 2);
			this.corr_max = this.corr_min + this.corr_range - 1;
		}else if (bits && bits < 32){
			this.corr_bits = bits;
			this.corr_range = (1 << bits) >>> 0;
			this.corr_min = -parseInt(this.corr_range / 2);
			this.corr_max = this.corr_min + this.corr_range - 1;
		}else{
			this.corr_bits = 32;
			this.corr_range = 0;
			this.corr_min = I32_MIN;
			this.corr_max = I32_MAX;
		}

		this.k = 0n;

		this.mBits = new Array(this.contexts);
		for(let i = 0; i < this.contexts; i++){
			this.mBits[i] = new ArithmeticModel(this.corr_bits + 1, false);
		}

		this.mCorrector = new Array(this.corr_bits + 1);
		this.mCorrector[0] = new ArithmeticBitModel();
		for(let i = 1; i < this.corr_bits + 1; i++){
			if (i <= bits_high){
				this.mCorrector[i] = new ArithmeticModel((1 << i) >>> 0, false);
			}else{
				this.mCorrector[i] = new ArithmeticModel((1 << bits_high) >>> 0, false);
			}
		}

		// log("contexts", this.contexts, "#bits", this.mBits.length, "#correctors", this.mCorrector.length);

	}

	decompress(pred, context){

		let real = pred + this.readCorrector(this.mBits[context]);

		if(real < 0){
			real += this.corr_range;
		}else if(real > this.corr_range){
			real -= this.corr_range;
		}

		return real;
	}

	readCorrector(mBits){
		let c = null;

		this.k = this.dec.decodeSymbol(mBits);

		if(this.k !== 0){
			if(this.k < 32){
				if(this.k <= this.bits_high){
					c = this.dec.decodeSymbol(this.mCorrector[this.k]);
				}else{
					let k1 = this.k - this.bits_high;

					c = this.dec.decodeSymbol(this.mCorrector[this.k]);

					let c1 = this.dec.readBits(k1);

					c = ((c << k1) | c1) >>> 0;
				}

				let cmp = (1 << (this.k - 1)) >>> 0;
				if(c >= cmp){
					c++;
				}else{
					let tmp = (1 << this.k) >>> 0;
					c -= (tmp - 1);
				}
			}else{
				this.c = this.corr_min;
			}
		}else{
			c = this.dec.decodeBit(this.mCorrector[0]);
		}

		return c;
	}

	getK(){
		return this.k;
	}

}