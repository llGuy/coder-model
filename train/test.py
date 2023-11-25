import torch
from torch.utils.data import DataLoader

import model
import data

import argparse

def translate_probabilities(out1, out2, out3):
    operations = [ 'add', 'sub', 'mul', 'div', 'mov' ]
    inputs = [ 'x0', 'x1', 'x2' ]
    registers = [ 'r0', 'r1', 'r2' ]

    output = ''

    for i in range(data.INSTRUCTIONS_PER_PROGRAM):
        instruction_probs = out1[i]
        lvalue_probs = out2[i]
        rvalue_probs = out3[i]

        op_idx = int(torch.argmax(instruction_probs))
        lvalue_idx = int(torch.argmax(lvalue_probs))
        rvalue_idx = int(torch.argmax(rvalue_probs))

        instruction_str = operations[op_idx]
        lvalue_str = None
        rvalue_str = None

        if lvalue_idx < 3:
            lvalue_str = inputs[lvalue_idx]
        else:
            lvalue_str = registers[lvalue_idx - 3]

        if rvalue_idx < 3:
            rvalue_str = inputs[rvalue_idx];
        elif rvalue_idx < 6:
            rvalue_str = registers[rvalue_idx - 3]
        else:
            rvalue_str = str(rvalue_idx - 6)

        output += instruction_str + ' ' + lvalue_str + ' ' + rvalue_str + '\n'

    return output



def test_main(model_path, example_idx):
    dataset = data.ProgDataset('../data/dataset/validation', data.NUM_VAL_EXAMPLES)

    io_pairs = dataset.load_io_pairs(example_idx)
    label1, label2, label3 = dataset.load_label(example_idx)

    # Test output of the model
    model = torch.load(model_path)

    # Turn off gradient computation
    model.eval()
    with torch.no_grad():
        out1, out2, out3 = model(io_pairs)
        print("Predicted:")
        print(translate_probabilities(out1, out2, out3))

        print("Original:")
        print(translate_probabilities(label1, label2, label3))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--example', type=int, required=True)
    args = parser.parse_args()

    test_main(args.model, args.example)
