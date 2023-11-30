#include "sim.h"
#include "prog.h"

#include <string>
#include <fstream>
#include <filesystem>

namespace nb = nanobind;

struct ExecutionStatus {
    bool ioMatch;
    uint32_t diffs[kMaxOutputs];
};

/* This encapsulates the "instructions" of the output program */
struct Program {
    static inline constexpr uint32_t kNumTokensPerProg = 
        kProgramNumInstructions * 3;

    /* Each instruction has 3 entries (op leftOperand rightOperand) */
    uint8_t tokens[kProgramNumInstructions][3];
};

struct SimManager::Impl {
    /* Gets incremented after each call to step() */
    uint32_t globalTimeStep;

    /* Number of worlds / agents trying to generate programs */
    uint32_t numWorlds;

    /* This buffer contains all the IO pairs contiguously */
    void *ioPairs;
    void *ioPairsU32;

    /* The current state for tall the programs in the batch */
    Program *progs;

    float *rewards;

    Impl(uint32_t num_worlds,
         std::pair<float *, uint32_t *> io_pairs);
};

SimManager::Impl::Impl(uint32_t num_worlds,
                       std::pair<float *, uint32_t *> io_pairs)
    : globalTimeStep(0),
      numWorlds(num_worlds),
      ioPairs(io_pairs.first),
      ioPairsU32(io_pairs.second),
      progs(nullptr),
      rewards(nullptr)
{
}

static std::pair<float *, uint32_t *> loadIOPairs(uint32_t num_worlds)
{
    namespace fs = std::filesystem;

    uint32_t num_floats_per_set = (kMaxInputs + kMaxOutputs) * kNumIOPairs;
    uint32_t bytes_per_io_set = num_floats_per_set * sizeof(float);

    void *io_pairs = malloc(num_worlds * bytes_per_io_set);
    uint32_t *io_pairs_u32 = (uint32_t *)malloc(num_worlds * bytes_per_io_set);

    const fs::path kDatasetDir = fs::path(PROJECT_DIR) / "dataset";
    const fs::path kTrainDir = kDatasetDir / "train";

    /* Load all the IO pairs now */
    for (int io_set = 0; io_set < num_worlds; ++io_set) {
        uint8_t *current_set_ptr = (uint8_t *)io_pairs + 
                                   bytes_per_io_set * io_set;

        uint32_t *current_set_u32_ptr = io_pairs_u32 +
                                        num_floats_per_set * io_set;

        std::string file_name = "io-pair-" + std::to_string(io_set);
        fs::path set_path = kTrainDir / file_name;

        std::ifstream file_stream(set_path);

        /* Just read all the floats */
        file_stream.read((char *)current_set_ptr, bytes_per_io_set);

        for (int p = 0; p < num_floats_per_set; ++p) {
            current_set_u32_ptr[p] = *((float *)current_set_ptr + p);
        }
    }

    return std::make_pair((float *)io_pairs, io_pairs_u32);
}

static void resetProgram(Program *prog)
{
    for (int i = 0; i < kProgramNumInstructions; ++i) {
        prog->tokens[i][0] = operationToByte(Operation::Mov);
        prog->tokens[i][1] = leftOperandToByte(Operand::Input, i % kMaxInputs);
        prog->tokens[i][2] = rightOperandToByte(Operand::Literal, 0);
    }
}

static void applyAction(uint32_t num_worlds,
                        Program *programs,
                        uint32_t current_token_id,
                        ActionTensor &actions)
{
    auto view = actions.view();

    float action_pmf[kActionSize] = {};

    uint32_t instr_to_mod = current_token_id / 3;
    uint32_t token_to_mod = current_token_id % 3;

    assert(num_worlds == view.shape(0));

    for (int i = 0; i < num_worlds; ++i) {
        /* Copy the actions over */
        for (int j = 0; j < kActionSize; ++j) {
            action_pmf[j] = view(i, j);
        }

        /* The action to take corresponds to the token. */
        uint32_t token_to_write = argmaxIdx(action_pmf, kActionSize);

        switch (token_to_mod) {
        case 0: {
            token_to_write %= (uint32_t)Operation::None;
        } break;

        case 1: {
            token_to_write %= (kMaxInputs + kNumRegisters);
        } break;

        case 2: {
            token_to_write %= (kMaxInputs + kNumRegisters + kLiteralRange);
        } break;

        default: {
            assert(false);
        } break;
        }

        programs[i].tokens[instr_to_mod][token_to_mod] = (uint8_t)token_to_write;
    }
}

static ExecutionStatus executeProgram(Program *program,
                                      uint32_t *io_inputs,
                                      uint32_t *io_outputs)
{
    int inputs[kMaxInputs] = {};
    int registers[kNumRegisters] {};
    int current_literal = 0;

    for (int i = 0; i < kMaxInputs; ++i) {
        inputs[i] = io_inputs[i];
    }

    uint32_t instr_idx = 0;

    for (int i = 0; i < kProgramNumInstructions; ++i) {
        uint8_t b0 = program->tokens[i][0];
        uint8_t b1 = program->tokens[i][1];
        uint8_t b2 = program->tokens[i][2];

        Operation op_type = (Operation)b0;
        OperandData left_operand = byteToLeftOperand(b1);
        OperandData right_operand = byteToRightOperand(b2);

        int *left_container;
        if (left_operand.op == Operand::Input) {
            left_container = &inputs[left_operand.value];
        }
        else if (left_operand.op == Operand::Register) {
            left_container = &registers[left_operand.value];
        }
        else {
            assert(false);
        }

        int *right_container;
        if (right_operand.op == Operand::Input) {
            right_container = &inputs[right_operand.value];
        }
        else if (right_operand.op == Operand::Register) {
            right_container = &registers[right_operand.value];
        }
        else if (right_operand.op == Operand::Literal) {
            current_literal = right_operand.value;
            right_container = &current_literal;
        }

        switch (op_type) {
        case Operation::Add: {
            *left_container += *right_container;
        } break;

        case Operation::Sub: {
            *left_container -= *right_container;
        } break;

        case Operation::Mul: {
            *left_container *= *right_container;
        } break;

        case Operation::Div: {
            if (*right_container == 0) {
                break;
            }
            *left_container /= *right_container;
        } break;

        case Operation::Mov: {
            *left_container = *right_container;
        } break;

        default: {} break;
        }
    }

    int *gen_outputs = inputs;

    ExecutionStatus status = {};
    status.diffs[0] = (uint32_t)abs((int)gen_outputs[0] - (int)io_outputs[0]);
    status.diffs[1] = (uint32_t)abs((int)gen_outputs[1] - (int)io_outputs[1]);
    status.diffs[2] = (uint32_t)abs((int)gen_outputs[2] - (int)io_outputs[2]);

    status.ioMatch = (status.diffs[0] == 0 && 
                      status.diffs[1] == 0 && 
                      status.diffs[2] == 0);

    return status;
}

static void checkPrograms(uint32_t num_worlds,
                          Program *programs,
                          uint32_t *io_pairs,
                          float *rewards_out)
{
    uint32_t num_ints_per_set = (kMaxInputs + kMaxOutputs) * kNumIOPairs;

    /* Run the program on all the IO pairs and count the amount of matching. */
    for (int prog_idx = 0; prog_idx < num_worlds; ++prog_idx) {
        Program *current_prog = programs + prog_idx;

        uint32_t *current_io_pairs = io_pairs + prog_idx * num_ints_per_set;

        float rewards_for_prog = 0.0f;
        
        for (int io_pair_idx = 0; io_pair_idx < kNumIOPairs; ++io_pair_idx) {
            uint32_t *current_io_ptr = current_io_pairs + 
                                       io_pair_idx * (kMaxInputs + kMaxOutputs);

            uint32_t *current_inputs = current_io_ptr;
            uint32_t *current_outputs = current_io_ptr + kMaxInputs;

            auto status = executeProgram(current_prog, current_inputs, current_outputs);

            if (status.ioMatch) {
                rewards_for_prog += 1.0f;
            }
            else {
                uint32_t matches = 0;
                for (int d = 0; d < kMaxOutputs; ++d)
                    if (status.diffs[d] == 0)
                        matches++;

                if (matches == 2) {
                    rewards_for_prog += 0.1f;
                }
            }
        }

        rewards_out[prog_idx] = rewards_for_prog;
    }
}

SimManager::SimManager() = default;
SimManager::~SimManager() = default;

SimManager::SimManager(uint32_t num_worlds)
    : impl(new Impl(num_worlds, loadIOPairs(num_worlds)))
{
}

void SimManager::reset()
{
    impl->globalTimeStep = 0;

    if (!impl->progs) {
        impl->progs = (Program *)malloc(sizeof(Program) * impl->numWorlds);
    }

    if (!impl->rewards) {
        impl->rewards = (float *)malloc(sizeof(float) * impl->numWorlds);
    }

    for (int i = 0; i < impl->numWorlds; ++i) {
        resetProgram(&impl->progs[i]);
        impl->rewards[i] = 0;
    }
}

void SimManager::step(ActionTensor action_tensor)
{
    auto action_tensor_view = action_tensor.view();

    uint32_t batch_size = action_tensor_view.shape(0);
    uint32_t action_size = action_tensor_view.shape(1);

    /* Apply the modification to the program. */
    applyAction(impl->numWorlds, impl->progs, 
                impl->globalTimeStep % Program::kNumTokensPerProg,
                action_tensor);

    checkPrograms(impl->numWorlds, impl->progs,
                  (uint32_t *)impl->ioPairsU32,
                  impl->rewards);

    ++impl->globalTimeStep;
}

ProgObservationTensor SimManager::getProgObservations()
{
    uint32_t tensor_floats = impl->numWorlds * kProgObservationSize;
    float *tensor_values = new float[tensor_floats];

    uint32_t total_tokens = kProgramNumInstructions * 3;
    uint32_t token_cursor = impl->globalTimeStep % total_tokens;

    /* Encode the programs into the tensor values. */
    for (int i = 0; i < impl->numWorlds; ++i) {
        Program *prog = impl->progs + i;

        float *current_ptr = tensor_values + i * kProgObservationSize;
        *(current_ptr++) = (float)token_cursor;

        for (int instr_idx = 0; instr_idx < kProgramNumInstructions; ++instr_idx) {
            *(current_ptr++) = (float)prog->tokens[instr_idx][0];
            *(current_ptr++) = (float)prog->tokens[instr_idx][1];
            *(current_ptr++) = (float)prog->tokens[instr_idx][2];
        }

        assert(current_ptr - tensor_values == (i+1) * kProgObservationSize);
    }

    nb::capsule owner(tensor_values, [](void *p) noexcept {
        delete[] (float *) p;
    });

    return ProgObservationTensor(tensor_values, 
        { impl->numWorlds, kProgObservationSize }, owner);   
}

IOPairObservationTensor SimManager::getIOPairObservations()
{
    uint32_t tensor_floats = impl->numWorlds * kIOPairObservationSize;
    float *tensor_values = new float[tensor_floats];

    memcpy(tensor_values, impl->ioPairs, tensor_floats * sizeof(float));

    nb::capsule owner(tensor_values, [](void *p) noexcept {
        delete[] (float *) p;
    });

    return IOPairObservationTensor(tensor_values, 
        { impl->numWorlds, kIOPairObservationSize }, owner);   
}

RewardTensor SimManager::getRewards()
{
    /* Allocate the reward tensor and fill in */
    uint32_t tensor_floats = impl->numWorlds;
    float *tensor_values = new float[tensor_floats];

    memcpy(tensor_values, impl->rewards, tensor_floats * sizeof(float));

    nb::capsule owner(tensor_values, [](void *p) noexcept {
        delete[] (float *) p;
    });

    return RewardTensor(tensor_values, 
        { impl->numWorlds }, owner);   
}
