import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as dist
from torch.nn.utils.rnn import pad_sequence
from data_rnn import load_ndfa, load_brackets


# Define the LSTM model
class NDFA_LSTM(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, number_layers = 3):
        super(NDFA_LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.lstm = nn.LSTM(input_size = emb_size, 
                            hidden_size = hidden_size, 
                            num_layers = number_layers, 
                            batch_first = True,
                            dropout = 0.5) # Regularization between layers
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        output = self.linear(lstm_out)
        return output



######### LOAD DATA #########


# Load NFDA dataset
x_train, (i2w, w2i) = load_brackets(n=150_000)

# Add start and end tokens to each sequence
x_train_preprocessed = [[w2i['.start']] + seq + [w2i['.end']] for seq in x_train]

# Set a maximum number of tokens per batch
max_tokens_per_batch = 50000  # Adjust this value as needed

# Split data into batches
batches = []
current_batch = []
current_batch_tokens = 0

for seq in x_train_preprocessed:
    # Check if adding the sequence exceeds the maximum tokens per batch
    if current_batch_tokens + len(seq) <= max_tokens_per_batch:
        current_batch.append(seq)
        current_batch_tokens += len(seq)
    else:
        batches.append(current_batch)
        current_batch = [seq]
        current_batch_tokens = len(seq)

# Add the last batch
if current_batch:
    batches.append(current_batch)

# Now, 'batches' contains lists of sequences with start and end tokens, 
# respecting the maximum tokens per batch

# Pad sequences within each batch to the same length
padded_batches = [pad_sequence([torch.tensor(s) for s in batch],
                    batch_first=True, padding_value=w2i['.pad']) for batch in batches]

# Convert lists of batches to PyTorch tensors
tensor_batches = [torch.tensor(batch, dtype=torch.long) for batch in padded_batches]

# Remove the first column (start token) from each sequence in tensor_batches 
# to get target_batches
target_batches = [batch[:, 1:] for batch in tensor_batches]

# Append a column of zeros to each sequence in target_batches
target_batches = [torch.cat((batch, torch.zeros(batch.size(0), 1, dtype=torch.long)), 
                    dim=1) for batch in target_batches]




def take_sample(lnprobs, temperature=1.0):
    """
    Sample an element from a categorical distribution
    :param lnprobs: Outcome logits
    :param temperature: Sampling temperature. 1.0 follows the given distribution, 0.0 returns the maximum probability element.
    :return: The index of the sampled element.
    """
    if temperature == 0.0:
        return lnprobs.argmax()
    
    p = F.softmax(lnprobs / temperature, dim=0)
    cd = dist.Categorical(p)

    return cd.sample()


def check_viability(sequence, open_token, close_token):
    # A simple check for viability based on balanced parentheses
    balance = 0
    for token in sequence:
        if token == open_token:
            balance += 1
        elif token == close_token:
            balance -= 1
        if balance < 0:  # A closing bracket came before an opening bracket
            return 'unviable'
    return 'viable' if balance == 0 else 'unviable'


def generate_sequence(model, seed_sequence, max_length, temperature, num_samples):
    model.eval()
    generated_sequences = []  # Store sequences and their viability

    with torch.no_grad():
        for _ in range(num_samples):
            generated_sequence = seed_sequence.copy()
            sequence_tensor = torch.tensor([generated_sequence], dtype=torch.long)

            for _ in range(max_length - len(seed_sequence)):
                output = model(sequence_tensor)
                last_logits = output[0, -1, :]
                next_token = take_sample(last_logits, temperature).item()
                generated_sequence.append(next_token)

                if next_token == w2i['.end']:
                    break

                sequence_tensor = torch.tensor([generated_sequence], dtype=torch.long)

            generated_tokens = [i2w[token_id] for token_id in generated_sequence]
            viability = check_viability(generated_tokens, i2w[w2i['(']], i2w[w2i[')']])
            generated_sequences.append({'sequence': ' '.join(generated_tokens), 'viability': viability})

    return generated_sequences




######### TRAIN MODEL #########



# Initialize the model
vocab_size = len(i2w)
emb_size = 100
hidden_size = 100

model = NDFA_LSTM(vocab_size, emb_size, hidden_size)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=w2i['.pad']) # Ignore padding tokens
optimizer = optim.Adam(model.parameters(), lr=0.005)

num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    for input_batch, target_batch in zip(tensor_batches, target_batches):
        optimizer.zero_grad()

        # Forward pass
        output = model(input_batch)

        # Reshape output and target to (batch_size * sequence_length, vocab_size)
        output = output.view(-1, vocab_size)
        target = target_batch.view(-1)

        # Compute the loss
        loss = criterion(output, target)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}\n')

    # Generate sequences after the epoch
    seed_sequence = [w2i['.start'], w2i['('], w2i['('], w2i[')']]  # Example seed sequence
    samples = generate_sequence(model, seed_sequence, max_length = 75, temperature = 1.0, num_samples = 10)
    
    total_viability = sum(1 for sample in samples if sample['viability'] == 'viable')

    for i, sample in enumerate(samples, 1):
        print(f"Sample {i}: {sample['sequence']}, {sample['viability']}")

    print(f'Total viable samples for epoch {epoch + 1}: {total_viability}/{len(samples)}\n')

