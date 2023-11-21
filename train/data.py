import torch
import numpy as np

# Dimensions of the input to the model (1000 IO pairs)
INPUT_DIM = 1000 * 6

INSTRUCTIONS_PER_PROGRAM = 30

# Baseline model dimensions
BRANCH1_OUTPUTS = INSTRUCTIONS_PER_PROGRAM * 5
BRANCH2_OUTPUTS = INSTRUCTIONS_PER_PROGRAM * 6
BRANCH3_OUTPUTS = INSTRUCTIONS_PER_PROGRAM * 26

class Dataset:
    def __init__(
        self,
        data_folder_path: str,
        num_examples: int
    ):
        self.folder_path = data_folder_path
        self.num_examples = num_examples

    # Load IO pair.
    def load_io_pair(self, idx: int):
        io_pair_path = self.folder_path + "/io-pair-" + str(idx) + ".txt"

    # Load label program.
    def load_label(self, idx: int):
        label_path = self.folder_path + "/src-" + str(idx) + ".asm"

        dims = [5, 6, 26]

        # Parse data into lists, which can be formatted into tensors.
        data = [[] for _ in dims]

        with open(prog_binary, 'rb') as f:

            for i in range(INSTRUCTIONS_PER_PROGRAM):

                instr_probs = np.fromfile(f, dtype=np.float32, count=dims[0])
                lval_probs = np.fromfile(f, dtype=np.float32, count=dims[1])
                rval_probs = np.fromfile(f, dtype=np.float32, count=dims[2])
                
                if len(instr_probs) == 0 or len(lval_probs) == 0 or len(rval_probs) == 0:
                    print("Improperly formatted input file")
                    return 0

                data[0].append(instr_probs)
                data[1].append(lval_probs)
                data[2].append(rval_probs)

        data = [torch.from_numpy(np.array(data[i])) for i in range(len(dims))]
    
if __name__ == "__main__":
    obj = Dataset()
    obj.load_label('../data/dataset/src-0.asm')




class BatchedDataLoader:

    # Define the splits by a dictionary with entries for 'train',
    # 'validation' and 'test'. Need to associate these with their
    # sizes.
    def __init__(
        self,
        data_set: Dataset,
        batch_size: int,
        shuffle: bool
    ):
        self.data_set = data_set
        self.batch_size = batch_size

        self.current = 0
        self.indices = None

        if shuffle:
          # TODO: implement shuffle
          pass
        else:
          self.indices = torch.arange(0, self.data_set.num_examples, 1)

    def __len__(self):
        return self.data_set.num_examples

    def __getitem__(self, idx):
        io_pairs = []
        labels = []
        for i in range(self.batch_size):
            example_idx = self.indices[i + idx * self.batch_size]
            current_io_pair = self.data_set.load_io_pair(example_idx)
            current_label = self.data_set.load_label(example_idx)
            io_pairs.append(current_io_pair)
            labels.append(current_label)

        io_pairs_full = torch.cat(io_pairs, dim=0)
        labels_full = torch.cat(labels, dim=0)

        return io_pairs_full, labels_full
