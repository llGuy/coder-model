#pragma once

#include <stdint.h>

struct BitVector {
    BitVector(uint32_t num_bits);
    BitVector(BitVector &&o);
    BitVector &operator=(BitVector &&o);
    ~BitVector();

    void reset();
    bool getBit(uint32_t bit_idx);
    void setBit(uint32_t bit_idx, uint32_t value);

    uint32_t numBytes;
    uint8_t *bytes;
};

inline uint32_t popCount(uint8_t bits) 
{
#ifndef __GNUC__
    return __popcnt(bits);
#else
    return __builtin_popcount(bits);
#endif
}
