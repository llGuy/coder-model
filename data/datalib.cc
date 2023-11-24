#include "datalib.h"

#include <string.h>
#include <stdio.h>
#include <assert.h>

char const *kOperationNames[] =
{
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


char *serializeOperandData(const OperandData &operand, char *mem)
{
	switch (operand.op) {
		case Operand::Input:
		{
			char str[] = "x0";
			str[1] = operand.value + '0';
			return serializeString(str, mem);
	 	} break;

		case Operand::Register:
		{
			char str[] = "r0";
			str[1] = operand.value + '0';
			return serializeString(str, mem);
		} break;

		case Operand::None: case Operand::Literal:
		{
			 char str[16];
			 snprintf(str, 16, "%d", operand.value);
			 return serializeString(str, mem);
		} break;
	}
	return nullptr;
}

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

#if 0
	// !!!!!!!!!!!Uncomment this to dump the probability vectors and the instructions
	// they correspond to!!!!!!!!!!!!!
	// Compare generated instruction with probability arrays.
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
#endif

	return reader;
}

char *getReadableInstruction(OperationData &op, char *instr)
{
	char *s = (char *) instr;
	s = serializeOpCode(op.op, s);
	s = serializeSpace(s);
	s = serializeOperandData(op.left, s);
	s = serializeSpace(s);
	s = serializeOperandData(op.right, s);
	s = serializeNewline(s);
	*s = 0;
	return s + 1;
}
