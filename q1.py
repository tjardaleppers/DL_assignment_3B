import torch


# def create_batches(input, batch_size):
#     """
#     This function creates batches from a list of input sequences.
#     """

#     batches = []

#     for i in range(0, len(input), batch_size):
#         batches.append(input[i:i+batch_size])
    
#     return batches


def pad_convert(input_batch):
    """
    This function takes a batch of input sequences and pads them to the same length (maximal length in the batch).
    It returns the padded batch as a tensor.
    """

    (input, labels) = zip(*input_batch)

    pad_token = 0
    max_len = len(max(input, key=len))
    
    for seq in input:
        if len(seq) < max_len:
            seq.extend([pad_token] * (max_len - len(seq)))
    padded_input = torch.tensor(input, dtype = torch.long)

    labels = torch.tensor(labels, dtype = torch.long)

    return padded_input, labels
