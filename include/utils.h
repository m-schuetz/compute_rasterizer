
#pragma once

#include <cstdint>

namespace morton{

	// from: https://www.forceflow.be/2013/10/07/morton-encodingdecoding-through-bit-interleaving-implementations/
	// license: Creative Commons Attribute-NonCommercial Sharealike 3.0 Unported license.
	// also: https://github.com/Forceflow/libmorton


	// method to seperate bits from a given integer 3 positions apart
	inline uint64_t splitBy3(unsigned int a){
		uint64_t x = a & 0x1fffff; // we only look at the first 21 bits
		x = (x | x << 32) & 0x1f00000000ffff; // shift left 32 bits, OR with self, and 00011111000000000000000000000000000000001111111111111111
		x = (x | x << 16) & 0x1f0000ff0000ff; // shift left 32 bits, OR with self, and 00011111000000000000000011111111000000000000000011111111
		x = (x | x << 8) & 0x100f00f00f00f00f; // shift left 32 bits, OR with self, and 0001000000001111000000001111000000001111000000001111000000000000
		x = (x | x << 4) & 0x10c30c30c30c30c3; // shift left 32 bits, OR with self, and 0001000011000011000011000011000011000011000011000011000100000000
		x = (x | x << 2) & 0x1249249249249249;
		return x;
	}

	inline uint64_t encode(unsigned int x, unsigned int y, unsigned int z){
		uint64_t answer = 0;
		answer |= (splitBy3(x) << 2) | (splitBy3(y) << 1) | (splitBy3(z) << 0);
		return answer;
	}
}