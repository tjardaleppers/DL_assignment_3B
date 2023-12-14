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

    (input, labels, _) = zip(*input_batch)

    pad_token = 0
    max_len = len(max(input, key=len))

    padded_input = []
    masks = []
    
    for seq in input:
        # Create mask for the sequence, 1 for real tokens, 0 for padding
        mask = [1] * len(seq) + [0] * (max_len - len(seq))
        masks.append(mask)

        padded_seq = seq + [pad_token] * (max_len - len(seq))
        padded_input.append(padded_seq)

    padded_input = torch.tensor(padded_input, dtype = torch.long)
    masks = torch.tensor(masks, dtype = torch.long)
    labels = torch.tensor(labels, dtype = torch.long)

    return padded_input, labels, masks
