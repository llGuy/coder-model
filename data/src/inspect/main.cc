#include <filesystem>
#include <fstream>
#include <assert.h>

#include "datalib.h"

/* Inspect a given input io pairs file, or inspect a label program.
*/
void parse_label_program(int idx, std::string &root)
{
    namespace fs = std::filesystem;
    // std::fstream stream(fs::path(dirPath) / programFileName, std::ios::binary | std::ios::out);
    std::string label_file_name = "src-" + std::to_string(idx);

    std::fstream stream(fs::path(root) / label_file_name, std::ios::binary | std::ios::in);
    assert(stream.is_open());

    // Read entire program into memory.
    char *prog_buf[PROGRAM_SIZE_BYTES];
    float *it = (float *) prog_buf;
    stream.read(reinterpret_cast<char *>(prog_buf), PROGRAM_SIZE_BYTES);

    char print_buf[100] = {0};
    for (int i = 0; i < PROGRAM_NUM_INSTR; ++i)
    {
        OperationData instr;
        it = deserializeInstruction(it, instr);
        getReadableInstruction(instr, print_buf);
        printf("%s", print_buf);
    }
}

int main(int argc, char *argv[])
{
    std::string default_dir = "dataset/train";
    int idx = std::stoi(argv[1]);

    parse_label_program(idx, default_dir);
    return 0;
}
