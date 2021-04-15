#version 450

#extension GL_NV_gpu_shader5 : enable

#define I8_MIN             -128
#define I8_MAX              127 
#define U8_MIN             0x00       //   0
#define U8_MAX             0xff       // 255
#define U8_MAX_MINUS_ONE   0xFE       // 254
#define U8_MAX_PLUS_ONE    0x0100     // 256

#define AC_MinLength  0x01000000u
#define AC_MaxLength  0xFFFFFFFFu

#define BM_LengthShift  13
#define BM_MaxCount     8192          // 1 << BM__LengthShift; 

#define DM_LengthShift  15
#define DM_MaxCount     32768         // 1 << DM__LengthShift;

// TODO, hardcoded to 500kb now
#define sizeof_P10_dynamics 500000



struct Batch {
	uint chunk_start;
	uint chunk_size;
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

	uint distribution_size;

	// dynamic object size data
	uint offset;
	uint size;
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

	ArithmeticBitModel mBitCorrector;

	// actual array sizes depend on bits and contexts
	ArithmeticModel[32] mBits;
	ArithmeticModel[32] mCorrector;

	// dynamic object size data
	uint offset;
	uint size;

};

struct Point10{
	ivec4 last_xyz;
	StreamingMedian5[16] last_x_diff_median5;
	StreamingMedian5[16] last_y_diff_median5;
	int[8] last_height;

	ArithmeticModel changed_values;
	IntegerCompressor ic_dx;
	IntegerCompressor ic_dy;
	IntegerCompressor ic_z;

};

struct Color{
	float r;
	float g;
	float b;
	float a;
};


layout(local_size_x = 32, local_size_y = 1) in;

layout(std430, set = 0, binding = 0) buffer LazData {
	uint lazdata[];
};

layout(std430, set = 0, binding = 1) buffer Positions {
	float positions[];
};

layout(std430, set = 0, binding = 2) buffer Colors {
	Color colors[];
};

layout(std430, set = 0, binding = 3) buffer Batches {
	Batch batches[];
};

layout(std430, set = 0, binding = 4) buffer Point10s {
	Point10 point10s[];
};

layout(std430, set = 0, binding = 5) buffer StructBuffers {
	uint struct_buffers[];
};



layout (std430, binding = 10) buffer framebuffer_data {
	int32_t ssTest[];
};





uniform int uNumChunks;


void assert(bool value){
	return;
}


int getMedian5(inout StreamingMedian5 smedian){
	return smedian.values[2];
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

void initDecoder(uint batch_index){
	uint offset = batches[batch_index].chunk_start + 26;

	uint value = readU8(offset + 0) << 24;
	value = value | (readU8(offset + 1) << 16);
	value = value | (readU8(offset + 2) <<  8);
	value = value | (readU8(offset + 3) <<  0);
	batches[batch_index].dec_value = value;
	batches[batch_index].dec_length = AC_MaxLength;
	batches[batch_index].dec_offset = 26;
}

// void update_AM(inout ArithmeticModel model){

// 	// halve counts when a threshold is reached
// 	if ((model.total_count += model.update_cycle) > DM_MaxCount){
// 		model.total_count = 0;
// 		for (uint n = 0; n < model.symbols; n++){
// 			uint index = model.symbol_count + n;
			
// 			//uint newDist = (model.distribution[index] + 1) >> 1;
// 			uint newDist = (struct_buffers[model.offset + index] + 1) >> 1;

// 			//model.distribution[index] = newDist;
// 			struct_buffers[model.offset + index] = newDist;

// 			model.total_count += newDist;
// 		}
// 	}

// 	uint sum = 0;
// 	uint s = 0;
// 	uint scale = 0x80000000u / model.total_count;

// 	if (model.table_size == 0){
// 		for (uint k = 0; k < model.symbols; k++){
// 			//model.distribution[k] = (scale * sum) >> (31 - DM_LengthShift);
// 			struct_buffers[model.offset + k] = (scale * sum) >> (31 - DM_LengthShift);

// 			//sum += model.distribution[model.symbol_count + k];
// 			sum += struct_buffers[model.offset + model.symbol_count + k];
// 		}
// 	}else{
// 		for (uint k = 0; k < model.symbols; k++){
// 			//model.distribution[k] = (scale * sum) >>> (31 - DM__LengthShift);
// 			struct_buffers[model.offset + k] = (scale * sum) >>> (31 - DM__LengthShift);

// 			//sum += model.distribution[model.symbol_count + k];
// 			sum += struct_buffers[model.offset + model.symbol_count + k];

// 			//uint w = model.distribution[k] >>> model.table_shift;
// 			uint w = struct_buffers[model.offset + k] >> model.table_shift;

// 			while (s < w){
// 				s++;
// 				//model.distribution[model.decoder_table + s] = k - 1;
// 				struct_buffers[model.offset + model.decoder_table + s] = k - 1;
// 			}
// 		}

// 		//model.distribution[model.decoder_table] = 0;
// 		struct_buffers[model.offset + model.decoder_table] = 0;

// 		while (s <= model.table_size){
// 			s++;
// 			//model.distribution[model.decoder_table + s] = model.symbols - 1;
// 			struct_buffers[model.offset + model.decoder_table + s] = model.symbols - 1;
// 		}
// 	}

// 	// set frequency of model updates
// 	model.update_cycle = (5 * model.update_cycle) >> 2;
// 	uint max_cycle = (model.symbols + 6) << 3;
// 	if (model.update_cycle > max_cycle){
// 		model.update_cycle = max_cycle;
// 	}
// 	model.symbols_until_update = model.update_cycle;

// }

ArithmeticModel create_AM(uint symbols, uint offset){

	ArithmeticModel model;

	// model.symbols  symbols;
	// model.last_symbol = symbols - 1;

	// uint table_size;
	// uint table_shift;
	// uint decoder_table;
	// uint distribution_size;

	// if(symbols > 16){
	// 	table_size = 1 << uint(ceil(log2(symbols)) - 2);
	// 	uint table_bits = log2(table_size);

	// 	table_shift = DM_LengthShift - table_bits;
	// 	distribution_size = 2 * symbols * table_size + 2;
	// 	decoder_table = 2 * symbols;
	// }else{
	// 	decoder_table = 0;
	// 	table_size = 0;
	// 	table_shift = 0;
	// 	distribution_size = 2 * symbols;
	// }

	// model.table_size = table_size;
	// model.table_shift = table_shift;
	// model.decoder_table = decoder_table;
	// model.distribution_size = distribution_size;
	// model.symbol_count = symbols;

	// model.total_count = 0;
	// model.update_cycle = symbols;

	// for(int k = 0; k < symbols; k++){
	// 	struct_buffers[offset + symbols + k] = 1;
	// }

	// // update_AM(model);

	// model.update_cycle = (symbols + 6) >> 1;
	// model.symbols_until_update = model.update_cycle;

	return model;
}

IntegerCompressor create_IC(uint bits, uint contexts, uint offset){
	IntegerCompressor ic;

	// TODO

	return ic;
}

void initPoint10(uint batch_index, int X, int Y, int Z){

	uint offset = batch_index * sizeof_P10_dynamics;

	ArithmeticModel changed_values = create_AM(64, offset);
	offset += changed_values.size;


	IntegerCompressor ic_dx = create_IC(32, 2, offset);
	offset += ic_dx.size;

	IntegerCompressor ic_dy = create_IC(32, 22, offset);
	offset += ic_dy.size;

	IntegerCompressor ic_z = create_IC(32, 20, offset);
	offset += ic_z.size;


	// sanity check
	assert(offset == sizeof_P10_dynamics);



	Point10 point10;

	point10.changed_values = changed_values;
	point10.ic_dx = ic_dx;
	point10.ic_dy = ic_dy;
	point10.ic_z = ic_z;
	point10.last_xyz = ivec4(X, Y, Z, 0);

	point10s[batch_index] = point10;

	return;
}







void main(){

	uint batch_index = gl_GlobalInvocationID.x;
	uint byte_offset = batches[batch_index].chunk_start;

	if(batch_index >= uNumChunks){
		return;
	}

	int X = readI32(byte_offset + 0);
	int Y = readI32(byte_offset + 4);
	int Z = readI32(byte_offset + 8);

	positions[3 * batch_index + 0] = float(X) / 1000.0;
	positions[3 * batch_index + 1] = float(Y) / 1000.0;
	positions[3 * batch_index + 2] = float(Z) / 1000.0;

	uint R = readU16(byte_offset + 20);
	uint G = readU16(byte_offset + 22);
	uint B = readU16(byte_offset + 24);

	colors[batch_index].r = normalizeColor(R);
	colors[batch_index].g = normalizeColor(G);
	colors[batch_index].b = normalizeColor(B);
	colors[batch_index].a = 1.0;

	ssTest[3 * batch_index + 0] = X;
	ssTest[3 * batch_index + 1] = Y;
	ssTest[3 * batch_index + 2] = Z;

	{
		initDecoder(batch_index);
		initPoint10(batch_index, X, Y, Z);
	}



}

