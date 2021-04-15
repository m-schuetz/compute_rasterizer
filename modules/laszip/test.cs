#version 450

#define I8_MIN             -128
#define I8_MAX              127 
#define U8_MIN             0x00       //   0
#define U8_MAX             0xff       // 255
#define U8_MAX_MINUS_ONE   0xFE       // 254
#define U8_MAX_PLUS_ONE    0x0100     // 256

#define I32_MIN            -2147483648
#define I32_MAX             2147483647

#define AC_MinLength  0x01000000u
#define AC_MaxLength  0xFFFFFFFFu

#define BM_LengthShift  13
#define BM_MaxCount     8192          // 1 << BM__LengthShift; 

#define DM_LengthShift  15
#define DM_MaxCount     32768         // 1 << DM_LengthShift;

uint number_return_map[64] = uint[](
	15, 14, 13, 12, 11, 10,  9,  8,
	14,  0,  1,  3,  6, 10, 10,  9,
	13,  1,  2,  4,  7, 11, 11, 10,
	12,  3,  4,  5,  8, 12, 12, 11,
	11,  6,  7,  8,  9, 13, 13, 12,
	10, 10, 11, 12, 13, 14, 14, 13,
	 9, 10, 11, 12, 13, 14, 15, 14,
	 8,  9, 10, 11, 12, 13, 14, 15
);

uint number_return_level[64] = uint[](
	0,  1,  2,  3,  4,  5,  6,  7,
	1,  0,  1,  2,  3,  4,  5,  6,
	2,  1,  0,  1,  2,  3,  4,  5,
	3,  2,  1,  0,  1,  2,  3,  4,
	4,  3,  2,  1,  0,  1,  2,  3,
	5,  4,  3,  2,  1,  0,  1,  2,
	6,  5,  4,  3,  2,  1,  0,  1,
	7,  6,  5,  4,  3,  2,  1,  0
);

uint U32_ZERO_BIT_0(uint n){
	return n & 0xFFFFFFFEu;
}

uint U32_ZERO_BIT_0_1(uint n){
	return n & 0xFFFFFFFC;
}

uint U8_FOLD(uint n){
	if(n < U8_MIN){
		return n + U8_MAX_PLUS_ONE;
	}else if(n > U8_MAX){
		return (n - U8_MAX_PLUS_ONE);
	}else{
		return n;
	}
}

uint U8_CLAMP(uint n){

	if(n <= U8_MIN){
		return U8_MIN;
	}else if(n >= U8_MAX){
		return U8_MAX;
	}else{
		return n;
	}
}

uint getNumberReturnMap(uint n, uint r){
	uint index = n * 8 + r;

	return number_return_map[index];
}

uint getNumberReturnLevel(uint n, uint r){
	uint index = n * 8 + r;

	return number_return_level[index];
}

struct List{
	uint offset;
	uint size;
};

struct Batch {
	uint chunk_start;
	uint chunk_size;
	uint num_points;
	uint points_read;
	uint dec_value;
	uint dec_length;
	uint dec_offset;
};

struct ArithmeticModel {
	uint symbols;
	uint last_symbol;
	uint table_size;
	uint table_shift;

	uint total_count;
	uint update_cycle;
	uint symbols_until_update;

	uint symbol_count;
	uint decoder_table;

	List distribution; // [uint]
};

struct ArithmeticBitModel{
	uint bit_0_count;
	uint bit_count;
	uint bit_0_prob;
	uint update_cycle;
	uint bits_until_update;
};

struct StreamingMedian5{
	int[5] values;
	bool high;
};

struct IntegerCompressor{
	uint bits;
	uint contexts;
	uint bits_high;
	uint range;

	uint corr_bits;
	uint corr_range;
	uint corr_min;
	uint corr_max;

	uint k;

	ArithmeticBitModel mBitCorrector;

	// actual array sizes depend on bits and contexts
	// this takes ~5 seconds to compile
	//ArithmeticModel[32] mBits;
	//ArithmeticModel[32] mCorrector;

	// points to the region that stores the respective arrays
	// Unfortunately, creating fixed-size arrays has an enormous impact 
	// on compile times up to tens of seconds
	List mBits;      // [ArithmeticModel]
	List mCorrector; // [ArithmeticModel]
};

struct Point10{
	ivec4 last_xyz;

	List last_x_diff_median5; // StreamingMedian5
	List last_y_diff_median5; // StreamingMedian5

	int[8] last_height;

	ArithmeticModel changed_values;
	IntegerCompressor ic_dx;
	IntegerCompressor ic_dy;
	IntegerCompressor ic_z;
};

struct RGB12{
	uint last_R;
	uint last_G;
	uint last_B;

	ArithmeticModel byte_used;
	ArithmeticModel rgb_diff_0;
	ArithmeticModel rgb_diff_1;
	ArithmeticModel rgb_diff_2;
	ArithmeticModel rgb_diff_3;
	ArithmeticModel rgb_diff_4;
	ArithmeticModel rgb_diff_5;
};

layout(local_size_x = 1, local_size_y = 1) in;

layout(std430, set = 0, binding = 0) buffer LazData {
	uint lazdata[];
};

layout(std430, set = 0, binding = 1) buffer Positions {
	float positions[];
};

layout(std430, set = 0, binding = 2) buffer Colors {
	uint colors[];
};

layout(std430, set = 0, binding = 3) buffer Batches {
	Batch batches[];
};

layout(std430, set = 0, binding = 5) buffer BufferDescriptors {
	uint buffer_uint_size;
	uint buffer_am_size;
	uint buffer_sm5_size;
};

layout(std430, set = 0, binding = 6) buffer BufferUint {
	uint buffer_uint[];
};

layout(std430, set = 0, binding = 7) buffer BufferAM {
	ArithmeticModel buffer_am[];
};

layout(std430, set = 0, binding = 8) buffer BufferSM5 {
	StreamingMedian5 buffer_sm5[];
};



layout (std430, binding = 10) buffer testI {
	int ssTestI[];
};

layout (std430, binding = 11) buffer testU {
	uint ssTestU[];
};


uniform int uNumChunks;
uniform int uPointsPerChunk;

Point10 point10;
RGB12 rgb12;
Batch batch;
IntegerCompressor ic;


void assert(bool value){
	return;
}


int getMedian5(uint offset){
	return buffer_sm5[offset].values[2];
}

void addMedian5(uint offset, int v){

	StreamingMedian5 smedian = buffer_sm5[offset];

	if (smedian.high){
		if (v < smedian.values[2]){

			smedian.values[4] = smedian.values[3];
			smedian.values[3] = smedian.values[2];

			if (v < smedian.values[0]){
				smedian.values[2] = smedian.values[1];
				smedian.values[1] = smedian.values[0];
				smedian.values[0] = v;
			}else if (v < smedian.values[1]){
				smedian.values[2] = smedian.values[1];
				smedian.values[1] = v;
			}else{
				smedian.values[2] = v;
			}
		} else {
			if (v < smedian.values[3]){
				smedian.values[4] = smedian.values[3];
				smedian.values[3] = v;
			}else{
				smedian.values[4] = v;
			}
			smedian.high = false;
		}
	}else{
		if (smedian.values[2] < v){
			smedian.values[0] = smedian.values[1];
			smedian.values[1] = smedian.values[2];

			if (smedian.values[4] < v){
				smedian.values[2] = smedian.values[3];
				smedian.values[3] = smedian.values[4];
				smedian.values[4] = v;
			} else if (smedian.values[3] < v){
				smedian.values[2] = smedian.values[3];
				smedian.values[3] = v;
			} else{
				smedian.values[2] = v;
			}
		}else{
			if (smedian.values[1] < v){
				smedian.values[0] = smedian.values[1];
				smedian.values[1] = v;
			}else{
				smedian.values[0] = v;
			}

			smedian.high = true;
		}
	}

	buffer_sm5[offset] = smedian;
}


uint readU32(uint byteOffset){

	uint b0SourceIndex = byteOffset / 4u;

	uint result = 0u;
	
	uint overlapCase = byteOffset % 4u;

	if(overlapCase == 0){
		result = lazdata[b0SourceIndex];
	}else if(overlapCase == 1){
		result = lazdata[b0SourceIndex] >> 8;
		result = result | (lazdata[b0SourceIndex + 1] << 24);
	}else if(overlapCase == 2){
		result = lazdata[b0SourceIndex] >> 16;
		result = result | (lazdata[b0SourceIndex + 1] << 16);
	}else if(overlapCase == 3){
		result = lazdata[b0SourceIndex] >> 24;
		result = result | (lazdata[b0SourceIndex + 1] << 8);
	}

	return result;
}

int readI32(uint byteOffset){
	return int(readU32(byteOffset));
}

uint readU16(uint byteOffset){
	uint result = readU32(byteOffset);

	result = result & 0xFFFF;

	return result;
}

// TODO: this can be done without detour over readU32
uint readU8(uint byteOffset){
	uint result = readU32(byteOffset);

	result = result & 0xFF;

	return result;
}

float normalizeColor(uint value){

	float n = float(value);

	if(n > 255.0){
		n = n / 256.0;
	}

	if(n > 1.0){
		n = n / 256.0;
	}

	return n;
}

void initDecoder(){
	uint offset = batch.chunk_start + 26;

	uint value = readU8(offset + 0) << 24;
	value = value | (readU8(offset + 1) << 16);
	value = value | (readU8(offset + 2) <<  8);
	value = value | (readU8(offset + 3) <<  0);

	batch.dec_value = value;
	batch.dec_length = AC_MaxLength;
	batch.dec_offset = 30;
}

ArithmeticBitModel create_ABM(){

	ArithmeticBitModel abm;

	abm.bit_0_count = 1;
	abm.bit_count   = 2;
	abm.bit_0_prob  = 1 << (BM_LengthShift - 1);

	abm.update_cycle = 4;
	abm.bits_until_update = 4;

	return abm;
}

void update_ABM(inout ArithmeticBitModel model){

	if ((model.bit_count += model.update_cycle) > BM_MaxCount){

		model.bit_count = (model.bit_count + 1) >> 1;
		model.bit_0_count = (model.bit_0_count + 1) >> 1;

		if (model.bit_0_count == model.bit_count){
			++model.bit_count;
		}
	}
	
	// compute scaled bit 0 probability
	uint scale = 0x80000000u / model.bit_count;
	model.bit_0_prob = (model.bit_0_count * scale) >> (31 - BM_LengthShift);

	// set frequency of model updates
	model.update_cycle = (5 * model.update_cycle) >> 2;
	if (model.update_cycle > 64){
		model.update_cycle = 64;
	}
	model.bits_until_update = model.update_cycle;
}

void update_AM(inout ArithmeticModel model){

	List ref_distribution = model.distribution;

	// halve counts when a threshold is reached
	if ((model.total_count += model.update_cycle) > DM_MaxCount){
		model.total_count = 0;
		for (uint n = 0; n < model.symbols; n++){
			uint index = model.symbol_count + n;
			
			//uint newDist = (model.distribution[index] + 1) >> 1;
			uint newDist = (buffer_uint[ref_distribution.offset + index] + 1) >> 1;

			//model.distribution[index] = newDist;
			buffer_uint[ref_distribution.offset + index] = newDist;

			model.total_count += newDist;
		}
	}

	uint sum = 0;
	uint s = 0;
	uint scale = 0x80000000u / model.total_count;

	if (model.table_size == 0){
		for (uint k = 0; k < model.symbols; k++){
			buffer_uint[ref_distribution.offset + k] = (scale * sum) >> (31 - DM_LengthShift);

			sum += buffer_uint[ref_distribution.offset + model.symbol_count + k];
		}
	}else{
		for (uint k = 0; k < model.symbols; k++){
			buffer_uint[ref_distribution.offset + k] = (scale * sum) >> (31 - DM_LengthShift);

			sum += buffer_uint[ref_distribution.offset + model.symbol_count + k];

			uint w = buffer_uint[ref_distribution.offset + k] >> model.table_shift;

			while (s < w){
				s++;
				buffer_uint[ref_distribution.offset + model.decoder_table + s] = k - 1;
			}
		}

		buffer_uint[ref_distribution.offset + model.decoder_table] = 0;

		while (s <= model.table_size){
			s++;
			buffer_uint[ref_distribution.offset + model.decoder_table + s] = model.symbols - 1;
		}
	}

	// set frequency of model updates
	model.update_cycle = (5 * model.update_cycle) >> 2;
	uint max_cycle = (model.symbols + 6) << 3;
	if (model.update_cycle > max_cycle){
		model.update_cycle = max_cycle;
	}
	model.symbols_until_update = model.update_cycle;

}

ArithmeticModel create_AM(uint symbols){

	ArithmeticModel model;

	model.symbols = symbols;
	model.last_symbol = symbols - 1;

	uint table_size;
	uint table_shift;
	uint decoder_table;
	uint distribution_size;

	if(symbols > 16){
		table_size = 1 << uint(ceil(log2(symbols)) - 2);
		uint table_bits = uint(log2(table_size));

		table_shift = DM_LengthShift - table_bits;
		distribution_size = 2 * symbols + table_size + 2;
		decoder_table = 2 * symbols;
	}else{
		decoder_table = 0;
		table_size = 0;
		table_shift = 0;
		distribution_size = 2 * symbols;
	}

	model.table_size = table_size;
	model.table_shift = table_shift;
	model.decoder_table = decoder_table;
	model.symbol_count = symbols;

	model.total_count = 0;
	model.update_cycle = symbols;

	// "allocate" memory
	uint distribution_offset = atomicAdd(buffer_uint_size, distribution_size);
	model.distribution.offset = distribution_offset;
	model.distribution.size = distribution_size;

	for(int k = 0; k < symbols; k++){
		buffer_uint[distribution_offset + symbols + k] = 1;
	}

	update_AM(model);

	model.update_cycle = (symbols + 6) >> 1;
	model.symbols_until_update = model.update_cycle;

	return model;
}

IntegerCompressor create_IC(uint bits, uint contexts){

	IntegerCompressor ic;
	ic.bits = bits;
	ic.contexts = contexts;
	ic.bits_high = 8;

	// if(range == 0){ ...
	// } else if(bits < 32){ // always 32 for current test data
	ic.corr_bits = 32;
	ic.corr_range = 0;
	ic.corr_min = I32_MIN;
	ic.corr_max = I32_MAX;
	// }

	ic.k = 0;

	ic.mBits.size = contexts;
	ic.mBits.offset = atomicAdd(buffer_am_size, ic.mBits.size);

	ic.mBitCorrector = create_ABM();

	ic.mCorrector.size = ic.corr_bits;
	ic.mCorrector.offset = atomicAdd(buffer_am_size, ic.mCorrector.size);

	for(int i = 0; i < contexts; i++){
		uint symbols = ic.corr_bits + 1;

		ArithmeticModel am = create_AM(symbols);
		buffer_am[ic.mBits.offset + i] = am;
	}

	for(int i = 0; i < ic.corr_bits; i++){
		uint symbols = 1 << min(i, ic.bits_high);

		ArithmeticModel am = create_AM(symbols);
		buffer_am[ic.mCorrector.offset + i] = am;
	}

	return ic;
}

StreamingMedian5 create_SM5(){
	StreamingMedian5 sm;
	sm.high = true;
	sm.values[0] = 0;
	sm.values[1] = 0;
	sm.values[2] = 0;
	sm.values[3] = 0;
	sm.values[4] = 0;

	return sm;
}

void initPoint10(int X, int Y, int Z){

	ArithmeticModel changed_values = create_AM(64);

	IntegerCompressor ic_dx = create_IC(32, 2);
	IntegerCompressor ic_dy = create_IC(32, 22);
	IntegerCompressor ic_z = create_IC(32, 20);

	uint sm_offset = atomicAdd(buffer_sm5_size, 32);
	point10.last_x_diff_median5.size = 16;
	point10.last_y_diff_median5.size = 16;
	point10.last_x_diff_median5.offset = sm_offset;
	point10.last_y_diff_median5.offset = sm_offset + 16;

	for(int i = 0; i < 16; i++){
		buffer_sm5[sm_offset + i] = create_SM5();
	}

	for(int i = 0; i < 8; i++){
		point10.last_height[i] = 0;
	}

	point10.changed_values = changed_values;
	point10.ic_dx = ic_dx;
	point10.ic_dy = ic_dy;
	point10.ic_z = ic_z;
	point10.last_xyz = ivec4(X, Y, Z, 0);

	return;
}

void renorm_dec_interval(){

	do {
		uint offset = batch.chunk_start + batch.dec_offset;
		batch.dec_value = (batch.dec_value << 8) | readU8(offset);
		batch.dec_offset++;
		batch.dec_length = batch.dec_length << 8;
	} while (batch.dec_length < AC_MinLength);

	return;
}

uint decodeSymbol(inout ArithmeticModel model){

	uint n = batch.dec_length;
	uint sym = batch.dec_length;
	uint x = batch.dec_length;
	uint y = batch.dec_length;

	uint dt = model.decoder_table;

	if(model.decoder_table != 0){

		batch.dec_length = batch.dec_length >> DM_LengthShift;
		uint dv = batch.dec_value / batch.dec_length;
		uint t = dv >> model.table_shift;

		sym = buffer_uint[model.distribution.offset + model.decoder_table + t];
		n = buffer_uint[model.distribution.offset + model.decoder_table + t + 1] + 1;

		while(n > sym + 1){
			uint k = (sym + n) >> 1;

			if(buffer_uint[model.distribution.offset + k] > dv){
				n = k;
			}else{
				sym = k;
			}
		}

		x = buffer_uint[model.distribution.offset + sym] * batch.dec_length;

		if(sym != model.last_symbol){
			y = buffer_uint[model.distribution.offset + sym + 1] * batch.dec_length;
		}
	}else{
		x = 0;
		sym = 0;

		batch.dec_length = batch.dec_length >> DM_LengthShift;
		n = model.symbols;

		uint k = n >> 1;

		do{
			uint z = batch.dec_length * buffer_uint[model.distribution.offset + k];

			if (z > batch.dec_value) {
				n = k;
				y = z;
			} else {
				sym = k;
				x = z;
			}

			k = (sym + n) >> 1;
		} while (k != sym);
	}

	batch.dec_value = batch.dec_value - x;
	batch.dec_length = y - x;

	if(batch.dec_length < AC_MinLength){
		renorm_dec_interval();
	}

	buffer_uint[model.distribution.offset + model.symbol_count + sym]++;
	model.symbols_until_update--;

	if(model.symbols_until_update == 0){
		update_AM(model);
	}

	return sym;
}

uint decodeBit(inout ArithmeticBitModel model){

	uint x = model.bit_0_prob * (batch.dec_length >> BM_LengthShift);
	uint sym = (batch.dec_value >= x) ? 1 : 0;

	if(sym == 0){
		batch.dec_length = x;
		model.bit_0_count++;
	}else{
		batch.dec_value = batch.dec_value - x;
		batch.dec_length = batch.dec_length - x;
	}

	if(batch.dec_length < AC_MinLength){
		renorm_dec_interval();
	}
	
	model.bits_until_update--;

	if(model.bits_until_update == 0){
		update_ABM(model);
	}

	return sym;
}

uint readBitsRec1(uint bits){

	if(bits > 19){

		// TODO handle exception
		
		return 1234;
	}

	batch.dec_length = batch.dec_length >> bits;
	uint sym = batch.dec_value / batch.dec_length;
	batch.dec_value = batch.dec_value - batch.dec_length * sym;

	if(batch.dec_length < AC_MinLength){
		renorm_dec_interval();
	}

	return sym;
}

uint readBits(uint bits){

	if(bits > 19){
		uint offset = batch.chunk_start + batch.dec_offset;
		uint tmp = readU16(offset);
		batch.dec_offset += 2;

		bits = bits - 16;
		uint tmp1 = readBitsRec1(bits) << 16;
		//uint tmp1 = readBits(batch, bits) << 16;

		return tmp1 | tmp;
	}

	batch.dec_length = batch.dec_length >> bits;
	uint sym = batch.dec_value / batch.dec_length;
	batch.dec_value = batch.dec_value - batch.dec_length * sym;

	if(batch.dec_length < AC_MinLength){
		renorm_dec_interval();
	}

	return sym;
}

int readCorrector(inout ArithmeticModel mBits){
	ssTestU[4]++;

	int c = 0;

	ic.k = decodeSymbol(mBits);

	if(ic.k != 0){
		if(ic.k < 32){
			if(ic.k <= ic.bits_high){
				c = int(decodeSymbol(buffer_am[ic.mCorrector.offset + ic.k]));
			}else{
				int k1 = int(ic.k - ic.bits_high);

				c = int(decodeSymbol(buffer_am[ic.mCorrector.offset + ic.k]));

				int c1 = int(readBits(k1));

				c = (c << k1) | c1;
			}

			int cmp = 1 << (ic.k - 1);
			if(c >= cmp){
				c++;
			}else{
				int tmp = (1 << ic.k);
				c -= (tmp - 1);
			}
		}
	}else{

		c = int(decodeBit(ic.mBitCorrector));
	}

	return c;
}

int decompress(int pred, uint context){

	int real = pred + readCorrector(buffer_am[ic.mBits.offset + context]);

	if(real < 0){
		real += int(ic.corr_range);
	}else if(real > ic.corr_range){
		real -= int(ic.corr_range);
	}

	return real;
}

void readPoint10(uint batch_index){

	uint index = batch.points_read;

	uint changed_values = decodeSymbol(point10.changed_values);

	uint r, n, m, l = 0;

	if(changed_values != 0){
		// TODO not implemented
	}else{
		r = 0;
		n = 0;
		m = getNumberReturnMap(n, r);
		l = getNumberReturnLevel(n, r);
	}

	// DECOMPRESS X
	int median = getMedian5(point10.last_x_diff_median5.offset + m);
	uint v = buffer_am[point10.ic_dx.mBits.offset].decoder_table;
	ic = point10.ic_dx;
	int diff = decompress(median, (n == 1) ? 1 : 0);
	point10.ic_dx = ic;
	int X = point10.last_xyz.x + diff;
	addMedian5(point10.last_x_diff_median5.offset + m, diff);

	// DECOMPRESS Y
	median = getMedian5(point10.last_y_diff_median5.offset + m);
	uint k_bits = point10.ic_dx.k;
	uint dy_context = ((n == 1) ? 1 : 0) 
		+ (k_bits < 20 ? U32_ZERO_BIT_0(k_bits) : 20);
	ic = point10.ic_dy;
	diff = decompress(median, dy_context);
	point10.ic_dy = ic;
	int Y = point10.last_xyz.y + diff;
	addMedian5(point10.last_y_diff_median5.offset + m, diff);

	// DECOMPRESS Z
	k_bits = (point10.ic_dx.k + point10.ic_dy.k) / 2;
	uint z_context = 
		((n == 1) ? 1 : 0) 
		+ (k_bits < 18 ? U32_ZERO_BIT_0(k_bits) : 18);
	ic = point10.ic_z;
	int Z = decompress(point10.last_height[l], z_context);
	point10.ic_z = ic;
	point10.last_height[l] = Z;

	point10.last_xyz = ivec4(X, Y, Z, 0);

	uint point_index = batch_index * uPointsPerChunk + batch.points_read;
	positions[3 * point_index + 0] = float(X) / 1000.0;
	positions[3 * point_index + 1] = float(Y) / 1000.0;
	positions[3 * point_index + 2] = float(Z) / 1000.0;

}

void initRGB12(uint R, uint G, uint B){

	rgb12.byte_used = create_AM(128);
	rgb12.rgb_diff_0 = create_AM(256);
	rgb12.rgb_diff_1 = create_AM(256);
	rgb12.rgb_diff_2 = create_AM(256);
	rgb12.rgb_diff_3 = create_AM(256);
	rgb12.rgb_diff_4 = create_AM(256);
	rgb12.rgb_diff_5 = create_AM(256);

	rgb12.last_R = R;
	rgb12.last_G = G;
	rgb12.last_B = B;

	return;
}

uint packColor(uint R, uint G, uint B){
	
	uint c = 0;

	c |= ((R > 255 ? R / 255 : R) & 0xFF) <<  0;
	c |= ((G > 255 ? G / 255 : G) & 0xFF) <<  8;
	c |= ((B > 255 ? B / 255 : B) & 0xFF) << 16;

	return c;
}

void readRGB12(uint batch_index){

	uint corr = 0;
	int diff = 0;
	uint sym = decodeSymbol(rgb12.byte_used);
	uint R, G, B = 0;

	if((sym & 1) != 0){
		corr = decodeSymbol(rgb12.rgb_diff_0);
		R = U8_FOLD(corr + (rgb12.last_R & 0xFF));
	}else{
		R = rgb12.last_R & 0xFF;
	}

	if((sym & 2) != 0){
		corr = decodeSymbol(rgb12.rgb_diff_1);
		R |= U8_FOLD(corr + (rgb12.last_R >> 8)) << 8;
	}else{
		R |= rgb12.last_R & 0xFF00;
	}

	if((sym & 64) != 0){
			
		diff = int(R & 0x00FF) - int(rgb12.last_R & 0x00FF);

		if((sym & 4) != 0){
			corr = decodeSymbol(rgb12.rgb_diff_2);
			G = U8_FOLD(corr + U8_CLAMP(diff + (rgb12.last_G & 255)));
		}else{
			G = rgb12.last_G & 0xFF;
		}

		if((sym & 16) != 0){
			corr = decodeSymbol(rgb12.rgb_diff_4);
			diff = (diff + (int(G & 0x00FF) - int(rgb12.last_G & 0x00FF))) / 2;
			B = U8_FOLD(corr + U8_CLAMP(diff + (rgb12.last_B & 255)));
		}else{
			B = rgb12.last_B & 0xFF;
		}

		diff = int(R >> 8) - int(rgb12.last_R >> 8);

		if((sym & 8) != 0){
			corr = decodeSymbol(rgb12.rgb_diff_3);
			G |= U8_FOLD(corr + U8_CLAMP(diff + (rgb12.last_G >> 8))) << 8;
		}else{
			G |= rgb12.last_G & 0xFF00;
		}

		if((sym & 32) != 0){
			corr = decodeSymbol(rgb12.rgb_diff_5);
			diff = (diff + (int(G >> 8) - int(rgb12.last_G >> 8))) / 2;
			B |= U8_FOLD(corr + U8_CLAMP(diff + (rgb12.last_B >> 8))) << 8;
		}else{
			B |= rgb12.last_B & 0xFF00;
		}

	}else{
		G = R;
		B = R;
	}


	rgb12.last_R = R;
	rgb12.last_G = G;
	rgb12.last_B = B;

	uint point_index = batch_index * uPointsPerChunk + batch.points_read;
	colors[point_index] = packColor(R, G, B);
}

void main(){

	uint batch_index = gl_GlobalInvocationID.x;
	batch = batches[batch_index];

	uint byte_offset = batch.chunk_start;

	if(batch_index >= uNumChunks){
		return;
	}


	int X = readI32(byte_offset + 0);
	int Y = readI32(byte_offset + 4);
	int Z = readI32(byte_offset + 8);

	uint point_index = batch_index * uPointsPerChunk;

	positions[3 * point_index + 0] = float(X) / 1000.0;
	positions[3 * point_index + 1] = float(Y) / 1000.0;
	positions[3 * point_index + 2] = float(Z) / 1000.0;

	uint R = readU16(byte_offset + 20);
	uint G = readU16(byte_offset + 22);
	uint B = readU16(byte_offset + 24);

	colors[point_index] = packColor(R, G, B);

	batch.points_read = 1;

	{
		initDecoder();
		initPoint10(X, Y, Z);
		initRGB12(R, G, B);

		uint numPoints = batch.num_points;

		for(int i = 1; i < numPoints; i++){
			readPoint10(batch_index);
			readRGB12(batch_index);
			batch.points_read++;
		}

	}



}

