#include <vector>
#include <random>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Maximum number of characters a program can have. */
inline constexpr uint32_t kMaxProgramSize = 2048;
inline constexpr uint32_t kNumRegisters = 3;
inline constexpr uint32_t kMaxInputs = 3;

enum class Operation {
  Add,
  Sub,
  Mul,
  Div,
  Mod,

  /* Takes in register and number of lines to skip. */
  BranchLess,
  BranchGreater,
  BranchEqual,
  None
};

char *kOperationNames[] = {
  "add",
  "sub",
  "mul",
  "div",
  "mod",
  "bl",
  "bg",
  "be"
};

char *serializeString(char *str, char *mem)
{
  strcpy(mem, str);
  return mem + strlen(mem);
}

char *serializeOpCode(Operation op, char *mem)
{
  return serializeString(kOperationNames[(uint32_t)op], mem);
}

char *serializeSpace(char *mem)
{
  mem[0] = ' ', mem[1] = 0;
  return mem + 1;
}

char *serializeNewline(char *mem)
{
  mem[0] = '\n', mem[1] = 0;
  return mem + 1;
}

inline constexpr uint32_t kNumOperations =
  (uint32_t)Operation::None;

inline constexpr uint32_t kMinLinesBranch = 2;
inline constexpr uint32_t kMaxLinesBranch = 5;

enum class Operand {
  Input,
  Literal,
  Register,
  None
};

inline constexpr uint32_t kNumOperands =
  (uint32_t)Operand::None;

inline constexpr uint32_t kLiteralRange = 20;
inline constexpr uint32_t kLiteralOffset = 10;

struct OperandData {
  Operand op;

  /* if op == Input: value takes on [0, numInputs]
   * if op == Register: value takes on [0, numRegisters]
   * if op == Literal: value takes on any value */
  int value;
};

char *serializeOperandData(const OperandData &operand, char *mem)
{
  switch (operand.op) {
    case Operand::Input: {
      char str[] = "x0";
      str[1] = operand.value + '0';
      return serializeString(str, mem);
    } break;

    case Operand::Register: {
      char str[] = "r0";
      str[1] = operand.value + '0';
      return serializeString(str, mem);
    } break;

    case Operand::None: case Operand::Literal: {
      char str[16];
      sprintf(str, "%d", operand.value);
      return serializeString(str, mem);
    } break;
  }

  return nullptr;
}

struct OperationData {
  Operation op;
  OperandData left;
  OperandData right;
};

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
OperandData generateLValue(ProgramState &state,
                           Rand &rnd)
{
  if (rnd.next() % 2 == 0) {
    uint32_t regIdx = rnd.next() % 
      std::min(kNumRegisters, state.usedRegisters + 1);

    if (regIdx >= state.usedRegisters)
      state.incrementUsedRegisters = true;

    return { Operand::Register, (int)regIdx };
  }

  uint32_t inputIdx = rnd.next() % state.numInputs;
  return {Operand::Input, (int)inputIdx};
}

/* For branch operations. Don't want to branch off of a register
 * which hasn't been written to yet. */
OperandData generateRestrictedLValue(ProgramState &state,
                                     Rand &rnd)
{
  if (rnd.next() % 2 == 0 && state.usedRegisters) {
    uint32_t regIdx = rnd.next() % state.usedRegisters;
    return { Operand::Register, (int)regIdx };
  }

  uint32_t inputIdx = rnd.next() % state.numInputs;
  return {Operand::Input, (int)inputIdx};
}

OperandData generateRValue(ProgramState &state,
                           Rand &rnd)
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

Operation generateOperation(Rand &rnd)
{
  if (rnd.next() % 4 == 0) {
    return (Operation)((int)Operation::BranchLess + rnd.next() % 3);
  }
  else {
    return (Operation)(rnd.next() % 5);
  }
}

char *pushInstructions(ProgramState &state,
                      Rand &rnd)
{
  /* Generate a random instruction. */
  Operation instruction = generateOperation(rnd);

  /* Generate instruction information. */
  OperationData opData = { instruction };
  switch (instruction) {
    case Operation::Add: case Operation::Sub:
    case Operation::Mul: case Operation::Div: {
      opData.left = generateLValue(state, rnd);
      opData.right = generateRValue(state, rnd);
    } break;

    case Operation::BranchLess:
    case Operation::BranchGreater:
    case Operation::BranchEqual: {
      opData.left = generateLValue(state, rnd);
      opData.right = { Operand::None, rnd.next(kMinLinesBranch, 
                                               kMaxLinesBranch) };
    } break;
  }

  /* Serialize instruction as string. */
  char *s = serializeOpCode(opData.op, state.programMemory);
  s = serializeSpace(s);
  s = serializeOperandData(opData.left, s);
  s = serializeSpace(s);
  s = serializeOperandData(opData.right, s);
  s = serializeNewline(s);

  if (state.incrementUsedRegisters) {
    state.usedRegisters++;
    state.incrementUsedRegisters = false;
  }

  return s;
}

char *makeProgram(uint32_t numInputs,
                  Rand &rnd)
{
  char *programMemory = (char *)malloc(sizeof(char) * kMaxProgramSize);
  ProgramState programState = {
    .programMemory = programMemory,
    .numInputs = numInputs,
    .usedRegisters = 0
  };

  int32_t instructionsLeft = 10 + (rnd.next() % 20);

  while (instructionsLeft--) {
    programState.programMemory = pushInstructions(programState, rnd);
  }

  return programMemory;
}

struct Program {
  uint32_t numInputs;
  uint32_t numOutputs;
  /* Source code */
  char *src;

  int *inputs;
};

char *deserializeInstruction(char *reader, OperationData &op)
{
  Operation opCode;

  bool isBranch = false;

  /* First, get instruction type. */
  if (reader[0] == 'a') {
    opCode = Operation::Add;
    reader += 4;
  }
  else if (reader[0] == 's') {
    opCode = Operation::Sub;
    reader += 4;
  }
  else if (reader[0] == 'm' && reader[1] == 'u') {
    opCode = Operation::Mul;
    reader += 4;
  }
  else if (reader[0] == 'd') {
    opCode = Operation::Div;
    reader += 4;
  }
  else if (reader[0] == 'm' && reader[1] == 'o') {
    opCode = Operation::Mod;
    reader += 4;
  }
  else if (reader[0] == 'b') {
    isBranch = true;

    if (reader[1] == 'l') {
      opCode = Operation::BranchLess;
      reader += 3;
    }
    else if (reader[1] == 'g') {
      opCode = Operation::BranchGreater;
      reader += 3;
    }
    else if (reader[1] == 'e') {
      opCode = Operation::BranchEqual;
      reader += 3;
    }
    else {
      assert(false);
    }
  }
  else {
    assert(false);
  }

  OperandData left;

  /* Get the left operand. */
  if (reader[0] == 'x') {
    left.op = Operand::Input;
    left.value = reader[1] - '0';
    reader += 3;
  }
  else if (reader[0] == 'r') {
    left.op = Operand::Register;
    left.value = reader[1] - '0';
    reader += 3;
  }
  else {
    assert(false);
  }

  OperandData right = {};

  if (reader[0] == 'x') {
    right.op = Operand::Input;
    right.value = reader[1] - '0';
    reader += 2;
  }
  else if (reader[0] == 'r') {
    right.op = Operand::Register;
    right.value = reader[1] - '0';
    reader += 2;
  }
  else {
    right.op = Operand::Literal;

    /* Literal */
    bool negative = false;
    if (reader[0] == '-') {
      negative = true;
      reader += 1;
    }

    /* Parse the number. */
    while (*reader != '\n' && *reader != 0) {
      right.value = right.value * 10 + (reader[0] - '0');
      reader += 1;
    }

    if (negative)
      right.value *= -1;
  }

  op.op = opCode;
  op.left = left;
  op.right = right;

  return reader;
}

int *executeProgram(const Program &program)
{
  int inputs[kMaxInputs] = {};
  int registers[kNumRegisters] {};
  int currentLiteral = 0;

  for (int i = 0; i <program.numInputs; ++i) {
    inputs[i] = program.inputs[i];
  }

  char *currentChar = program.src;

  while (*currentChar != 0) {
    OperationData instr;
    currentChar = deserializeInstruction(currentChar, instr);
    currentChar += 1;

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
    else {
      assert(instr.op == Operation::BranchLess || instr.op == Operation::BranchGreater ||
             instr.op == Operation::BranchEqual);

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
        *leftContainer /= *rightContainer;
      } break;

      case Operation::Mod: {
        *leftContainer %= *rightContainer;
      } break;

      case Operation::BranchLess: {
        if (*leftContainer < 0) {
          int numLines = *rightContainer;
          while (*currentChar != 0 && numLines) {
            currentChar = deserializeInstruction(currentChar, instr);
            currentChar += 1;
            --numLines;
          }
        }
      } break;

      case Operation::BranchGreater: {
        if (*leftContainer > 0) {
          int numLines = *rightContainer;
          while (*currentChar != 0 && numLines) {
            currentChar = deserializeInstruction(currentChar, instr);
            currentChar += 1;
            --numLines;
          }
        }
      } break;

      case Operation::BranchEqual: {
        if (*leftContainer == 0) {
          int numLines = *rightContainer;
          while (*currentChar != 0 && numLines) {
            currentChar = deserializeInstruction(currentChar, instr);
            currentChar += 1;
            --numLines;
          }
        }
      } break;
    }
  }

  int *outputs = (int *)malloc(sizeof(int) * program.numOutputs);

  for (int i = 0; i < program.numOutputs; ++i) {
    outputs[i] = inputs[i];
  }

  return outputs;
}

/* Example of how to use. */
#if 1
int main()
{
  char *outputFile = "test.asm";

  Rand rnd = {
    .dev = std::random_device(),
    .mt = std::mt19937(rnd.dev()),
    .dist = std::uniform_int_distribution<>(0, 1024)
  };

  char *src = makeProgram(3, rnd);

  int inputs[] = { 12, 103, 42 };
  Program prog = {
    .numInputs = 3,
    .numOutputs = 2,
    .src = src,
    .inputs = inputs
  };

  int *outputs = executeProgram(prog);

  printf("{%d,%d,%d} -> {%d,%d}\n", inputs[0], inputs[1], inputs[2], outputs[0], outputs[1]);

  FILE *file = fopen(outputFile, "w");
  assert(file);

  fputs(src, file);

  fclose(file);

  return 0;
}
#endif

/* Generate data set */
