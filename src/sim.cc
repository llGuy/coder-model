#include "sim.h"
#include "prog.h"

#include <string>
#include <fstream>
#include <filesystem>

namespace nb = nanobind;

/* This encapsulates the "instructions" of the output program */
struct Program {
    /* Each instruction has 3 entries (op leftOperand rightOperand) */
    uint8_t entries[kProgramNumInstructions][3];
};

struct SimManager::Impl {
    /* Gets incremented after each call to step() */
    uint32_t globalTimeStep;

    /* Number of worlds / agents trying to generate programs */
    uint32_t numWorlds;

    /* This buffer contains all the IO pairs contiguously */
    void *ioPairs;

    /* The current state for tall the programs in the batch */
    Program *progs;

    Impl(uint32_t num_worlds,
         void *io_pairs);
};

SimManager::Impl::Impl(uint32_t num_worlds,
                       void *io_pairs)
    : globalTimeStep(0),
      numWorlds(num_worlds),
      ioPairs(io_pairs),
      progs(nullptr)
{
}

static void *loadIOPairs(uint32_t num_worlds)
{
    namespace fs = std::filesystem;

    uint32_t num_io_pairs_per_set = (kMaxInputs + kMaxInputs) * kNumIOPairs;
    uint32_t bytes_per_io_set = num_io_pairs_per_set * sizeof(float);

    void *io_pairs = malloc(num_worlds * sizeof(float) * num_io_pairs_per_set);

    const fs::path kDatasetDir = fs::path(PROJECT_DIR) / "dataset";
    const fs::path kTrainDir = kDatasetDir / "train";

    /* Load all the IO pairs now */
    for (int io_set = 0; io_set < num_worlds; ++io_set) {
        uint8_t *current_set_ptr = (uint8_t *)io_pairs + 
                                   bytes_per_io_set * io_set;

        std::string file_name = "io-pair-" + std::to_string(io_set);
        fs::path set_path = kTrainDir / file_name;

        std::ifstream file_stream(set_path);

        /* Just read all the floats */
        file_stream.read((char *)current_set_ptr, bytes_per_io_set);
    }

    return io_pairs;
}

static void resetProgram(Program *prog)
{
    for (int i = 0; i < kProgramNumInstructions; ++i) {
        prog->entries[i][0] = operationToByte(Operation::Mov);
        prog->entries[i][1] = leftOperandToByte(Operand::Input, i % kMaxInputs);
        prog->entries[i][2] = rightOperandToByte(Operand::Literal, 0);
    }
}

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

    for (int i = 0; i < impl->numWorlds; ++i) {
        resetProgram(&impl->progs[i]);
    }
}

void SimManager::step(ActionTensor action_tensor)
{
    auto action_tensor_view = action_tensor.view();

    uint32_t batch_size = action_tensor_view.shape(0);
    uint32_t action_size = action_tensor_view.shape(1);

    /* Access an element in the tensor with `action_tensor_view(x, y)` */
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
            *(current_ptr++) = (float)prog->entries[instr_idx][0];
            *(current_ptr++) = (float)prog->entries[instr_idx][1];
            *(current_ptr++) = (float)prog->entries[instr_idx][2];
        }

        assert(current_ptr - tensor_values == kProgObservationSize);
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
}
