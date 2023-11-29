#include "sim.h"
#include "prog.h"

/* This encapsulates the "instructions" of the output program */
struct Program {
    /* Each instruction has 3 entries (op leftOperand rightOperand) */
    uint8_t entries[kProgramNumInstructions][3];
};

struct SimManager::Impl {
    /* Gets incremented after each call to step() */
    uint32_t globalTimeStep;

    /* This buffer contains all the IO pairs contiguously */
    void *ioPairs;

    /* The current state for tall the programs in the batch */
    Program *progs;
};

SimManager::SimManager(const char *path_to_data)
{
}
