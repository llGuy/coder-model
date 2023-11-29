#include <vector>
#include <random>
#include <stdio.h>
#include <fstream>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <filesystem>
#include <assert.h>

#include "prog.h"

struct Rand {
    std::random_device dev;
    std::mt19937 mt;
    std::uniform_int_distribution<> dist;

    int next() 
    { 
        return dist(mt); 
    }

    int next(int min, int max)
    {
        if (min == max) return min;
        return min + dist(mt) % (max-min);
    }
};

struct ProgramState {
    char *programMemory;
    uint32_t numInputs;
    uint32_t usedRegisters;
    bool incrementUsedRegisters;
};

/* Generate a storeable operand */
OperandData generateLValue(ProgramState &state, Rand &rnd, Operation opType) 
{
    if (rnd.next() % 2 == 0) {
        uint32_t regIdx = rnd.next() % 
            std::min(kNumRegisters, state.usedRegisters + 1);

        if (regIdx >= state.usedRegisters) {
            // Move operation has to be the first thing that happens if using register
            // as L value.
            if (opType != Operation::Mov) goto reject;

            state.incrementUsedRegisters = true;
        }

        return { Operand::Register, (int) regIdx };
    }

reject:
    uint32_t inputIdx = rnd.next() % state.numInputs;
    return {Operand::Input, (int)inputIdx};
}

OperandData generateRValueImpl(ProgramState &state, Operation opType,
                               const OperandData &left, Rand &rnd) 
{
    Operand operandType = (Operand)(rnd.next() % kNumOperands);

    if (operandType == Operand::Register && state.usedRegisters > 0) {
        return {operandType, (int)(rnd.next() % state.usedRegisters)};
    }

    operandType = (Operand)(rnd.next() % 2);

    if (operandType == Operand::Input) { 
        return {operandType, (int)(rnd.next() % state.numInputs)}; 
    }

    else { 
        return {operandType, (int)(rnd.next() % kLiteralRange - kLiteralOffset)}; 
    }
}

OperandData generateRValue(ProgramState &state, Operation opType,
                           const OperandData &left, Rand &rnd) 
{
    OperandData tmp = generateRValueImpl(state, opType, left, rnd);

    if (opType == Operation::Mul && tmp.op == Operand::Literal && tmp.value == 0) {
        return generateRValue(state, opType, left, rnd);
    }

    if (opType == Operation::Div && tmp.op == Operand::Literal && tmp.value == 0) {
        return generateRValue(state, opType, left, rnd);
    }

    if (opType == Operation::Sub && tmp == left) {
        return generateRValue(state, opType, left, rnd);
    }

    if (opType == Operation::Mov && tmp == left) {
        return generateRValue(state, opType, left, rnd);
    }

    if (opType == Operation::Mov && tmp.op == Operand::Literal && tmp.value == 0) {
        return generateRValue(state, opType, left, rnd);
    }

    return tmp;
}

Operation generateOperation(Rand &rnd)
{
    return (Operation) (rnd.next() % 5);
}

float *serializeFloats(float *dst, float *src, uint64_t numFloats)
{
    for (uint64_t i = 0; i < numFloats; ++i) {
        memcpy(dst + i, src + i, sizeof(float));
    }
    return dst + numFloats;
}

float *serializeInstr(OperationData opData, float *progMem)
{
    // If operation is nop/none, assume progMem points to 0'd mem, advance.
    if (opData.op == Operation::None) {
        return (float *) ((char *) progMem + PROGRAM_INSTR_LEN_BYTES);
    }

    // Else, construct the appropriate float vector, and memcpy.
    float instr_probs[INSTR_PROB_VECTOR_LEN] = {0};
    float lval_probs[LVAL_PROB_VECTOR_LEN] = {0}; // ordered {inputs..., registers...}
    float rval_probs[RVAL_PROB_VECOTR_LEN] = {0}; // ordered {inputs..., registers..., literals...}

    instr_probs[(uint32_t) opData.op] = 1.0f;

    switch (opData.left.op)
    {
        case Operand::Input: {
                lval_probs[opData.left.value] = 1.0f;
            } break;
        case Operand::Register: {
                lval_probs[kMaxInputs + opData.left.value] = 1.0f;
            } break;
        default: {
                // Malformed input, fuss.
                printf("Found an Lvalue not equal to register or input, aborting\n");
                assert(0);
            } break;
    }

    switch (opData.right.op) {
        case Operand::Input: {
                rval_probs[opData.right.value] = 1.0f;
            } break;
        case Operand::Register: {
                rval_probs[kMaxInputs + opData.right.value] = 1.0f;
            } break;
        case Operand::Literal: {
                rval_probs[kMaxInputs + kNumRegisters + kLiteralOffset + opData.right.value] = 1.0f;
            } break;
        default: {
                printf("Improper rvalue requested, aborting.\n");
                assert(0);
            } break;
    }

    // memcpy instr vector, then lval vector, then rval vector.
    float *save_ptr;
    save_ptr = serializeFloats(progMem, instr_probs, INSTR_PROB_VECTOR_LEN);
    save_ptr = serializeFloats(save_ptr, lval_probs, LVAL_PROB_VECTOR_LEN);
    save_ptr = serializeFloats(save_ptr, rval_probs, RVAL_PROB_VECOTR_LEN);
    return save_ptr;
}

char *pushInstruction(ProgramState &state,
                      Rand &rnd)
{
    /* Generate a random instruction. */
    Operation instruction = generateOperation(rnd);

    /* Generate instruction information. */
    OperationData opData = { instruction };
    switch (instruction) {
        case Operation::Add: case Operation::Sub:
        case Operation::Mul: case Operation::Div:
        case Operation::Mov: {
                opData.left = generateLValue(state, rnd, instruction);
                opData.right = generateRValue(state, instruction, opData.left, rnd);
            } break;
        default: {} break;
    }

    if (state.incrementUsedRegisters) {
        state.usedRegisters++;
        state.incrementUsedRegisters = false;
    }

    /* Serialize instruction as probability vectors. */
    return (char *) serializeInstr(opData, (float *) state.programMemory);
}

/* Generates a 30-entry program where each entry is a 37-vector
 * with 16-bit float entries.
 */
float *makeProgram(uint32_t numInputs, Rand &rnd)
{
    float *progMem = (float *) calloc(1, PROGRAM_SIZE_BYTES);
    ProgramState programState = {
        .programMemory = (char *) progMem,
        .numInputs = numInputs,
        .usedRegisters = 0
    };

    uint32_t instr_left = kProgramNumInstructions;
    while (instr_left--) {
        programState.programMemory = pushInstruction(programState, rnd);
    }
    return progMem;
}

struct Program {
    uint32_t numInputs;
    uint32_t numOutputs;
    /* Source code */
    float *src;
    int *inputs;
};

int *executeProgram(const Program &program)
{
    int inputs[kMaxInputs] = {};
    int registers[kNumRegisters] {};
    int currentLiteral = 0;

    for (int i = 0; i <program.numInputs; ++i) {
        inputs[i] = program.inputs[i];
    }

    float *currentInstr = program.src;

    for (int i = 0; i < kProgramNumInstructions; ++i) {
        OperationData instr;
        currentInstr = deserializeInstruction(currentInstr, instr);
        if (instr.op == Operation::Nop) continue;

        int *leftContainer;
        if (instr.left.op == Operand::Input) {
            leftContainer = &inputs[instr.left.value];
        }
        else if (instr.left.op == Operand::Register) {
            leftContainer = &registers[instr.left.value];
        }
        else {
            assert(false);
        }

        int *rightContainer;
        if (instr.right.op == Operand::Input) {
            rightContainer = &inputs[instr.right.value];
        }
        else if (instr.right.op == Operand::Register) {
            rightContainer = &registers[instr.right.value];
        }
        else if (instr.right.op == Operand::Literal) {
            currentLiteral = instr.right.value;
            rightContainer = &currentLiteral;
        }

        switch (instr.op) {
            case Operation::Add: {
                    *leftContainer += *rightContainer;
                } break;

            case Operation::Sub: {
                    *leftContainer -= *rightContainer;
                } break;

            case Operation::Mul: {
                    *leftContainer *= *rightContainer;
                } break;

            case Operation::Div: {
                    if (*rightContainer == 0) {
                        break;
                    }
                    *leftContainer /= *rightContainer;
                } break;

            case Operation::Mov: {
                    *leftContainer = *rightContainer;
                } break;

            default: {} break;
        }
    }

    int *outputs = (int *)malloc(sizeof(int) * program.numOutputs);

    for (int i = 0; i < program.numOutputs; ++i) {
        outputs[i] = inputs[i];
    }

    return outputs;
}

void generateSet(const char *dirPath, uint32_t numExamples)
{
    namespace fs = std::filesystem;

    /* Initialize random number generator. */
    Rand rnd = {
        .dev = std::random_device(),
        .mt = std::mt19937(rnd.dev()),
        .dist = std::uniform_int_distribution<>(0, 1024)
    };

    for (int i = 0; i < numExamples; ++i) {
        uint8_t numInputs = (uint8_t) rnd.next(1, kMaxInputs+1);
        uint8_t numOutputs = (uint8_t) rnd.next(1, numInputs+1);

        /* Generate the program */
        float *src = makeProgram(numInputs, rnd);

        { /* Save the program to a file. */
            std::string programFileName = "src-" + std::to_string(i);
            std::fstream stream(fs::path(dirPath) / programFileName, std::ios::binary | std::ios::out);
            assert(stream.is_open());
            stream.write(reinterpret_cast<const char *>(src), PROGRAM_SIZE_BYTES);
        }

        Program prog = {
            .numInputs = numInputs,
            .numOutputs = numOutputs,
            .src = src,
            .inputs = nullptr
        };

        /* Write the input output pair file. */
        std::string ioFileName = "io-pair-" + std::to_string(i);
        std::fstream stream(fs::path(dirPath) / ioFileName, std::ios::binary | std::ios::out);
        assert(stream.is_open());

        /* Generate the input output pairs. */
        for (int ioPair = 0; ioPair < 1000; ++ioPair) {
            int inputs[kMaxInputs] = {};
            for (int j = 0; j < kMaxInputs; ++j) {
                inputs[j] = rnd.next() % 64;
            }

            prog.inputs = inputs;

            int *outputs = executeProgram(prog);

            /* Write the input output pair to a file. */
            signed int filler = 0xBAD;
            for (int j = 0; j < numInputs; ++j) { 
                stream.write(reinterpret_cast<char const *>(inputs + j), sizeof(int)); 
            }

            for (int j = 0; j < kMaxInputs - numInputs; ++j) { 
                stream.write(reinterpret_cast<char const *>(&filler), sizeof(int)); 
            }

            for (int j = 0; j < numOutputs; ++j) { 
                stream.write(reinterpret_cast<char const *>(outputs + j), sizeof(int)); 
            }

            for (int j = 0; j < kMaxOutputs - numOutputs; ++j) { 
                stream.write(reinterpret_cast<char const *>(&filler), sizeof(int)); 
            }

            free(outputs);
        }

        free(src);
    }
}

/* Generate data set */
int main()
{
    char tmpBuffer[1024] = {};

    namespace fs = std::filesystem;

    const fs::path kDatasetDir = fs::path(PROJECT_DIR) / "dataset";
    const fs::path kTrainDir = kDatasetDir / "train";
    const fs::path kValidationDir = kDatasetDir / "validation";

    std::cout << "dataset direction at: " << kDatasetDir << std::endl;

    /* Create dataset directory if needed. */
    if (!fs::exists(kDatasetDir)) {
        fs::create_directory(kDatasetDir);
        fs::create_directory(kTrainDir);
        fs::create_directory(kValidationDir);
    }
    else {
        fs::remove_all(kDatasetDir);

        fs::create_directory(kDatasetDir);
        fs::create_directory(kTrainDir);
        fs::create_directory(kValidationDir);
    }

    generateSet(kTrainDir.c_str(), 1000);
    generateSet(kValidationDir.c_str(), 100);
}
