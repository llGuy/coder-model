#include "bitv.h"
#include <stdlib.h>
#include <string.h>

BitVector::BitVector(uint32_t num_bits) 
    : numBytes((num_bits + 7) / 8),
      bytes((uint8_t *)malloc(sizeof(uint8_t) * numBytes))
{
    memset(bytes, 0, sizeof(uint8_t) * numBytes);
}

BitVector::BitVector(BitVector &&o)
{
    if (bytes) {
        free(bytes);
    }

    numBytes = o.numBytes;
    bytes = o.bytes;

    o.numBytes = 0;
    o.bytes = nullptr;
}

BitVector &BitVector::operator=(BitVector &&o)
{
    if (bytes) {
        free(bytes);
    }

    numBytes = o.numBytes;
    bytes = o.bytes;

    o.numBytes = 0;
    o.bytes = nullptr;

    return *this;
}

BitVector::~BitVector()
{
    if (bytes) {
        free(bytes);
    }
}

void BitVector::reset()
{
    memset(bytes, 0, sizeof(uint8_t) * numBytes);
}

bool BitVector::getBit(uint32_t bit_idx)
{
    uint32_t byte_idx = bit_idx / 8;
    uint32_t bit_idx_mod = bit_idx % 8;
    return 1 & (bytes[byte_idx] >> bit_idx_mod);
}

void BitVector::setBit(uint32_t bit_idx, uint32_t value)
{
    uint32_t byte_idx = bit_idx / 8;
    uint32_t bit_idx_mod = bit_idx % 8;

    if (value) {
        bytes[byte_idx] |= (1 << bit_idx_mod);
    }
    else {
        bytes[byte_idx] &= ~(1 << bit_idx_mod);
    }
}
