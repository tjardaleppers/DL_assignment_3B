import data_rnn as rn
import random
import torch
import torch.nn as nn
import torch.optim as optim

# Load data
(x_train, y_train), (x_val, y_val), (i2w, w2i), numcls = rn.load_imdb(final=False)
split_index = len(x_val) // 2
x_val = x_val[:split_index]
y_val = y_val[:split_index]
x_test = x_val[split_index:]
y_test = y_val[split_index:]

class RNN(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.rnn = nn.RNN(emb_size, hidden_size, nonlinearity='relu',batch_first=True )        
        self.linear = nn.Linear(hidden_size, 2)

    def forward(self, x):
        x = self.embedding(x)
        # Only keep first output, since second is hidden layer value, which this implementation tracks on its own
        output_hidden, _ = self.rnn(x)      
        output_maxpool,_ = torch.max(output_hidden, dim=1)
        output = self.linear(output_maxpool)
        
        return output

def pad(inputs):
    # Pad the input sequences to the max lenght in batch
    max_seq_length = len(max(inputs, key=len))
    idx_pad = w2i['.pad']
    batch_copy = inputs.copy()
    for seq in batch_copy:
        if len(seq) < max_seq_length:       
            seq += [idx_pad] * (max_seq_length - len(seq))

    return batch_copy

def getBatches(x, y, batch_size, random_slices):   
    batches_inp = []
    targets_out = []
    # Set batch size to closest number that splits training set equally
    while len(x)%batch_size != 0:
        batch_size += 1
    number_of_batches = int(len(x)/batch_size)        
    
    # If random slices take random incides
    for i in range(number_of_batches):
        if random_slices:
            slice = random.sample(range(0,len(x)), batch_size)
            batch = [x[i] for i in slice]
            targets = [y[i] for i in slice]
        else:
            batch = x[i:i+batch_size]
            targets = y[i:i+batch_size]
        # Pad sequences and append to batches
        batch_padded = pad(batch) 
        batches_inp.append(torch.tensor(batch_padded, dtype=torch.long))
        targets_out.append(torch.tensor(targets, dtype=torch.long))    
    
    return batches_inp, targets_out

def evaluate_Classifier(model, val_set):
    # Evaluation mode
    model.eval()
    total = 0
    correct = 0
    # No gradient calculation
    with torch.no_grad():
        for input, labels in val_set:
            output = model(input)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy 

def trainModel(**params):
    torch.set_default_device('cuda')
    emb_size = int(params['emb_size'])
    hidden_size = int(params['hidden_size'])
    lr = params['lr']   
    
    # Get batches with random slices if True
    batch_size = 200
    padded_tensor_batches, targets = getBatches(x_train, y_train, batch_size, random_slices=True) 
    val_x, val_y = getBatches(x_val, y_val, batch_size,random_slices=False)

    # Initialize the model
    vocab_size = len(i2w)
    model = RNN(vocab_size, emb_size, hidden_size)
    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(model.parameters(), lr)

    num_epochs = 1
    for epoch in range(num_epochs):
        model.train()
        total = 0
        correct = 0
        total_loss = 0 
        # Loop of all batches where for each doing forward and backward
        for input_batch, target in zip(padded_tensor_batches, targets):
            optimizer.zero_grad()
            output = model(input_batch)
            loss = criterion(output, target)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()               

            # Get statistics        
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

        epoch_loss = total_loss / len(targets)
        epoch_accuracy = 100 * correct / total
        validation_accuracy = evaluate_Classifier(model, zip(val_x, val_y))
        print(f'Epoch [{epoch+1}/{num_epochs}], Epoch Loss: {epoch_loss:.4f}, Epoch Accuracy: {epoch_accuracy:.2f}%, Validation Accuracy: {validation_accuracy:.2f}%')
    
    return validation_accuracy
    

