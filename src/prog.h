#pragma once

#include <stdint.h>
#include <assert.h>

inline constexpr uint32_t kNumIOPairs = 32;

inline constexpr uint32_t kMaxProgramSize = 2048;
inline constexpr uint32_t kNumRegisters = 3;


// This is the original setup
#if 0
inline constexpr uint32_t kMinInputs = 2;
inline constexpr uint32_t kMaxInputs = 3;

inline constexpr uint32_t kMinOutputs = 2;
inline constexpr uint32_t kMaxOutputs = 3;

inline constexpr uint32_t kProgramNumInstructions = 5;

#else
// New setup
inline constexpr uint32_t kMinInputs = 2;
inline constexpr uint32_t kMaxInputs = 2;

inline constexpr uint32_t kMinOutputs = 2;
inline constexpr uint32_t kMaxOutputs = 2;

inline constexpr uint32_t kProgramNumInstructions = 3;
#endif



inline constexpr uint32_t kLiteralRange = 20;
inline constexpr uint32_t kLiteralOffset = 0;

#define PROGRAM_NUM_INSTR kProgramNumInstructions
#define PROGRAM_INSTR_LEN_BYTES (sizeof(float) * (5 + 6 + 26))
#define PROGRAM_SIZE_BYTES (PROGRAM_NUM_INSTR * PROGRAM_INSTR_LEN_BYTES)

#define INSTR_PROB_VECTOR_LEN ((int) Operation::None)
#define LVAL_PROB_VECTOR_LEN (kMaxInputs + kNumRegisters)
#define RVAL_PROB_VECOTR_LEN (kMaxInputs + kNumRegisters + kLiteralRange)

enum class Operation {
    Add,
    Sub,
    Mul,
    Div,
    Mov,
    None, // None must go before Nop since Nop isn't an explicit instruction.
    Nop
};

inline constexpr uint32_t kNumOperations = (uint32_t)Operation::None;

enum class Operand {
    Input,
    Literal,
    Register,
    None
};

inline constexpr uint32_t kNumOperands = (uint32_t)Operand::None;

struct OperandData {
    Operand op;

    /* if op == Input: value takes on [0, numInputs]
     * if op == Register: value takes on [0, numRegisters]
     * if op == Literal: value takes on any value */
    int value;

    bool operator==(const OperandData &other)
    {
        return op == other.op && value == other.value;
    }
};

struct OperationData {
    Operation op;
    OperandData left;
    OperandData right;
};

extern char const *kOperationNames[];
char *serializeString(char const *str, char *mem);
char *serializeOpCode(Operation op, char *mem);
char *serializeSpace(char *mem);
char *serializeNewline(char *mem);
void serializeNull(char *mem);
char *serializeOperandData(const OperandData &operand, char *mem);

uint64_t argmaxIdx(float *vec, uint64_t num_elem);
float *deserializeInstruction(float *reader, OperationData &op);
char *getReadableInstruction(OperationData &op, char *instr);

static inline uint8_t toToken(Operation op)
{
    return (uint8_t)op;
}

static inline uint8_t toToken(Operand operand, uint8_t no)
{
    if (operand == Operand::Input) {
        return (uint8_t)Operation::None + no;
    }
    else if (operand == Operand::Register) {
        return (uint8_t)Operation::None + kMaxInputs + no;
    }
    else if (operand == Operand::Literal) {
        return (uint8_t)Operation::None + kMaxInputs + kNumRegisters + no;
    }
    else {
        assert(false);
    }
}

static inline uint8_t operationToByte(Operation op)
{
    return (uint8_t)op;
}

static inline uint8_t leftOperandToByte(Operand operand, uint8_t no)
{
    if (operand == Operand::Input) {
        return no;
    } 
    else if (operand == Operand::Register) {
        return no + (uint8_t)kMaxInputs;
    }
    else {
        assert(false);
    }
}

static inline OperandData byteToLeftOperand(uint8_t byte)
{
    if (byte < kMaxInputs) {
        return { Operand::Input, byte };
    }
    else {
        return { Operand::Register, byte - (uint8_t)kMaxInputs };
    }
}

static inline uint8_t rightOperandToByte(Operand operand, uint8_t no)
{
    if (operand == Operand::Input) {
        return no;
    } 
    else if (operand == Operand::Register) {
        return no + (uint8_t)kMaxInputs;
    }
    else if (operand == Operand::Literal) {
        return no + (uint8_t)kMaxInputs + (uint8_t)kNumRegisters;
    }
    else {
        assert(false);
    }
}

static inline OperandData byteToRightOperand(uint8_t byte)
{
    if (byte < kMaxInputs) {
        return { Operand::Input, byte };
    }
    else if (byte < kMaxInputs + kNumRegisters) {
        return { Operand::Register, byte - (uint8_t)kMaxInputs };
    }
    else {
        return { Operand::Literal,
            byte - (uint8_t)kMaxInputs - (uint8_t)kNumRegisters };
    }
}
