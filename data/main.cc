#include <vector>
#include <random>
#include <stdio.h>
#include <fstream>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <filesystem>

#include <assert.h>

/* Maximum number of characters a program can have. */
inline constexpr uint32_t kMaxProgramSize = 2048;
inline constexpr uint32_t kNumRegisters = 3;
inline constexpr uint32_t kMaxInputs = 3;

// 30 instructions, each with 5-float, 6-float, and 26-float components.
#define PROGRAM_NUM_INSTR 30
#define PROGRAM_INSTR_LEN_BYTES (sizeof(float) * (5 + 6 + 26))
#define PROGRAM_SIZE_BYTES (PROGRAM_NUM_INSTR * PROGRAM_INSTR_LEN_BYTES)

enum class Operation {
  Add,
  Sub,
  Mul,
  Div,
  Mov,
	Nop, 
  None
};

enum class Operand {
  Input,
  Literal,
  Register,
  None
};

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

char const *kOperationNames[] = {
  "add",
  "sub",
  "mul",
  "div",
  "mov",
	"nop"
};

char *serializeString(char const *str, char *mem)
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

void serializeNull(char *mem)
{
	*mem = 0;
}

inline constexpr uint32_t kNumOperations =
  (uint32_t)Operation::None;

inline constexpr uint32_t kNumOperands =
  (uint32_t)Operand::None;

inline constexpr uint32_t kLiteralRange = 20;
inline constexpr uint32_t kLiteralOffset = 0;

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
      snprintf(str, 16, "%d", operand.value);
      return serializeString(str, mem);
    } break;
  }

  return nullptr;
}


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
    if (min == max)
      return min;

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
                           Rand &rnd,
                           Operation opType)
{
  if (rnd.next() % 2 == 0) {
    uint32_t regIdx = rnd.next() % 
      std::min(kNumRegisters, state.usedRegisters + 1);

    if (regIdx >= state.usedRegisters) {
      // Move operation has to be the first thing that happens if using register
      // as L value.
      if (opType != Operation::Mov)
        goto reject;

      state.incrementUsedRegisters = true;
    }

    return { Operand::Register, (int)regIdx };
  }

reject:
  uint32_t inputIdx = rnd.next() % state.numInputs;
  return {Operand::Input, (int)inputIdx};
}

OperandData generateRValueImpl(ProgramState &state,
                               Operation opType,
                               const OperandData &left,
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

OperandData generateRValue(ProgramState &state,
                           Operation opType,
                           const OperandData &left,
                           Rand &rnd)
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

#define INSTR_PROB_VECTOR_LEN ((int) Operation::Nop)
#define LVAL_PROB_VECTOR_LEN (kMaxInputs + kNumRegisters)
#define RVAL_PROB_VECOTR_LEN (kMaxInputs + kNumRegisters + kLiteralRange)

float *serializeFloats(float *dst, float *src, uint64_t numFloats)
{
	for (uint64_t i = 0; i < numFloats; ++i)
	{
		memcpy(dst + i, src + i, sizeof(float));
	}
	return dst + numFloats;
}

float *serializeInstr(OperationData opData, float *progMem)
{
	// If operation is nop/none, assume progMem points to 0'd mem, advance.
	if (opData.op == Operation::None)
	{
		return (float *) ((char *) progMem + PROGRAM_INSTR_LEN_BYTES);
	}
	
	// Else, construct the appropriate float vector, and memcpy.
	float instr_probs[INSTR_PROB_VECTOR_LEN] = {0};
	float lval_probs[LVAL_PROB_VECTOR_LEN] = {0}; // ordered {inputs..., registers...}
	float rval_probs[RVAL_PROB_VECOTR_LEN] = {0}; // ordered {inputs..., registers..., literals...}
	
	instr_probs[(uint32_t) opData.op] = 1.0f;

	switch (opData.left.op)
	{
		case Operand::Input: 
		{
			lval_probs[opData.left.value] = 1.0f;
		} break;
		case Operand::Register:
		{
			lval_probs[kMaxInputs + opData.left.value] = 1.0f;
		} break;
		default:
		{
			// Malformed input, fuss.
			printf("Found an Lvalue not equal to register or input, aborting\n");
			assert(0);
		} break;
	}

	switch (opData.right.op)
	{
		case Operand::Input:
		{
			rval_probs[opData.right.value] = 1.0f;
		} break;
		case Operand::Register:
		{
			rval_probs[kMaxInputs + opData.right.value] = 1.0f;
		} break;
		case Operand::Literal:
		{
			rval_probs[kMaxInputs + kNumRegisters + kLiteralOffset + opData.right.value] = 1.0f;
		} break;
		default: 
		{
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
float *makeProgram(uint32_t numInputs,
                  Rand &rnd)
{
	float *progMem = (float *) calloc(1, PROGRAM_SIZE_BYTES);
	ProgramState programState = 
	{
		.programMemory = (char *) progMem,
		.numInputs = numInputs,
		.usedRegisters = 0
	};
	uint32_t instr_left = 10 + rnd.next() % 20;
	while (instr_left--)
	{
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

uint64_t argmax_idx(float *vec, uint64_t num_elem)
{
	assert(num_elem > 0);
	uint64_t max_idx = 0;
	for (uint64_t i = 1; i < num_elem; ++i)
	{
		if (vec[i] > vec[max_idx]) max_idx = i;
	}
	return max_idx;
}

float *deserializeInstruction(float *reader, OperationData &op)
{
	// printf("DESERIALIZING NEW INSTRUCITON FROM IN STREAM\n");
	uint64_t argmax;
	
	Operation opCode;
	OperandData left;
	OperandData right;

	/* Extract and parse operation probabilities. Select operation with the highest
	 * probability. */

	// Unload instr probs from input stream.
	float instr_probs[INSTR_PROB_VECTOR_LEN];
	memcpy(instr_probs, reader, sizeof(instr_probs));
	reader += INSTR_PROB_VECTOR_LEN;

	// If all instruction values equal to 0, return nop.
	int nop = 1;
	for (int i = 0; i < INSTR_PROB_VECTOR_LEN; ++i) 
	{
		if (instr_probs[i] != 0.0f) { nop = 0; break; }
	}
	if (nop) { opCode = Operation::Nop; goto fill_instr; }

	// Determine the index of the highest probability.
	argmax = argmax_idx(instr_probs, INSTR_PROB_VECTOR_LEN);
	opCode = (Operation) argmax;

	/* Extract and parse lvalue probabilities. Select lvalue with the highest
	 * probability. */

	// Unload lval probs from input stream.
	float lval_probs[LVAL_PROB_VECTOR_LEN];
	memcpy(lval_probs, reader, sizeof(lval_probs));
	reader += LVAL_PROB_VECTOR_LEN;

	// Determine index of highest probability.
	argmax = argmax_idx(lval_probs, LVAL_PROB_VECTOR_LEN);	
	if (argmax < kMaxInputs)
	{
		left.op = Operand::Input;
		left.value = argmax;
	}
	else
	{
		left.op = Operand::Register;
		left.value = argmax - kMaxInputs;
	}

	/* Extract and parse rvalue probabilities. Select rvalue with the highest
	 * probability. */

	// Unload rval probs from input stream.
	float rval_probs[RVAL_PROB_VECOTR_LEN];
	memcpy(rval_probs, reader, sizeof(rval_probs));
	reader += RVAL_PROB_VECOTR_LEN;

	// Determine the index of highest probability.
	argmax = argmax_idx(rval_probs, RVAL_PROB_VECOTR_LEN);
	if (argmax < kMaxInputs)
	{
		right.op = Operand::Input;
		right.value = argmax;
	}
	else if (argmax < kMaxInputs + kNumRegisters)
	{
		right.op = Operand::Register;
		right.value = argmax - kMaxInputs;
	}
	else
	{
		right.op = Operand::Literal;
		right.value = argmax - kMaxInputs - kNumRegisters - kLiteralOffset;
	}

fill_instr:
	op.op = opCode;
	op.left = left;
	op.right = right;

#if 1 // !!!!!!!!!!!Uncomment this to dump the probability vectors and the instructions
			// they correspond to!!!!!!!!!!!!!
	// Compare generated instruction with probability arrays.
	/*
	printf("read arrays:\n");
	printf("instr probs: ");
	for (int i = 0; i < INSTR_PROB_VECTOR_LEN; ++i)
	{
		std::cout << std::fixed << std::setprecision(2) << instr_probs[i] << " ";
	}
	std::cout << std::endl;
	printf("instr probs: ");
	for (int i = 0; i < LVAL_PROB_VECTOR_LEN; ++i)
	{
		std::cout << std::fixed << std::setprecision(2) << lval_probs[i] << " ";
	}
	std::cout << std::endl;
	printf("lvalue probs: ");
	for (int i = 0; i < RVAL_PROB_VECOTR_LEN; ++i)
	{
		std::cout << std::fixed << std::setprecision(2) << rval_probs[i] << " ";
	}
	std::cout << std::endl;
	*/

  // Serialize instruction as string.
	char instr[50];
	char *s = (char *) instr;
	char *save = s;
  s = serializeOpCode(op.op, s);
  s = serializeSpace(s);
  s = serializeOperandData(op.left, s);
  s = serializeSpace(s);
  s = serializeOperandData(op.right, s);
  s = serializeNewline(s);
	*s = 0;
	printf("%s", save);
#endif 

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

  float *currentInstr = program.src;

	for (int i = 0; i < PROGRAM_NUM_INSTR; ++i)
	{
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

/* Generate data set */
int main()
{
  char tmpBuffer[1024] = {};

  namespace fs = std::filesystem;

  /* Create dataset directory if needed. */
  if (!fs::exists(fs::path("dataset"))) {
    fs::create_directory(fs::path("dataset"));
  }
  else {
    fs::remove_all(fs::path("dataset"));
    fs::create_directory(fs::path("dataset"));
  }

  uint32_t numTrainingExamples = 1;

  /* Initialize random number generator. */
  Rand rnd = {
    .dev = std::random_device(),
    .mt = std::mt19937(rnd.dev()),
    .dist = std::uniform_int_distribution<>(0, 1024)
  };

  std::fstream metadataStream(fs::path("dataset") / "metadata.txt", std::ios::out);
  assert(metadataStream.is_open());

  for (int i = 0; i < numTrainingExamples; ++i) {
    uint32_t numInputs = rnd.next(1, kMaxInputs+1);
    uint32_t numOutputs = rnd.next(1, numInputs+1);

    /* Generate the program */
    float *src = makeProgram(numInputs, rnd);

		{ /* Save the program to a file. */
      std::string programFileName = "src-" + std::to_string(i) + ".asm";
      std::fstream stream(fs::path("dataset") / programFileName, std::ios::binary | std::ios::out);
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
    std::string ioFileName = "io-pair-" + std::to_string(i) + ".txt";
    std::fstream streamIO(fs::path("dataset") / ioFileName, std::ios::out);
    assert(streamIO.is_open());

    /* Generate the input output pairs. */
    for (int ioPair = 0; ioPair < 1; ++ioPair) {
      int inputs[kMaxInputs] = {};
      for (int j = 0; j < kMaxInputs; ++j) {
        inputs[j] = (rnd.next() % 2048) - 1024;
      }

      prog.inputs = inputs;

      int *outputs = executeProgram(prog);

      /* Write the input output pair to a file. */
      std::string inputString = "{";
      for (int j = 0; j < numInputs; ++j) {
        inputString += std::to_string(inputs[j]);
        if (j < numInputs-1)
          inputString += ',';
      }
      inputString += "}->{";
      for (int j = 0; j < numOutputs; ++j) {
        inputString += std::to_string(outputs[j]);
        if (j < numOutputs-1)
          inputString += ',';
      }
      inputString += "}\n";

      streamIO << inputString;

      free(outputs);
    }

    streamIO.close();

    free(src);

    metadataStream << numInputs << "->" << numOutputs << "\n";
  }

  metadataStream.close();
}
