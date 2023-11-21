import torch

# Dimensions of the input to the model (1000 IO pairs)
INPUT_DIM = 1000 * 6

MAX_INSTRUCTIONS_PER_PROGRAM = 10

# Baseline model dimensions
BRANCH1_OUTPUTS = MAX_INSTRUCTIONS_PER_PROGRAM * 5
BRANCH2_OUTPUTS = MAX_INSTRUCTIONS_PER_PROGRAM * 6
BRANCH3_OUTPUTS = MAX_INSTRUCTIONS_PER_PROGRAM * 26

class Dataset:
  def __init_(
    self,
    data_folder_path: str,
    num_examples: int
  ):
    self.folder_path = data_folder_path
    self.num_examples = num_examples

  # Load IO pair
  def load_io_pair(idx: int):
    io_pair_path = self.folder_path + "/io-pair-" + str(idx) + ".txt"
    pass

  def load_label(idx: int):
    src_path = self.folder_path + "/src-" + str(idx) + ".asm"
    # Load label
    pass

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
