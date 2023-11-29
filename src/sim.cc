#include "sim.h"
#include "prog.h"

#include <string>
#include <fstream>
#include <filesystem>

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

void SimManager::step(ActionTensor action)
{
    
}

ObservationTensor SimManager::getObservations()
{
    
}

RewardTensor SimManager::getRewards()
{
    
}
