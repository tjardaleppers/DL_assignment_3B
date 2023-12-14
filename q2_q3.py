from data_rnn import load_imdb
from q1 import pad_convert
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class IMDBDataset(Dataset):
    def __init__(self, input, labels):
        self.input = input
        self.labels = labels
    
    def __len__(self):
        return len(self.input)
    
    def __getitem__(self, idx):
        input_sequence = self.input[idx]
        label = self.labels[idx]
        mask = (input_sequence != 0)
        
        return input_sequence, label, mask


class Classifier(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size):
        super(Classifier, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.linear1 = nn.Linear(embedding_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, input, mask):
        input = self.embedding(input)
        batch_size, seq_len, embedding_size = input.size()
        input = input.reshape(batch_size * seq_len, embedding_size) # Reshape to apply linear layer
        input = self.linear1(input)
        input = input.reshape(batch_size, seq_len, embedding_size) # Shape back to original
        input = F.relu(input)
        input = input * mask.unsqueeze(-1) # Apply mask to zero out padding
        global_max_pool, _ = torch.max(input, dim = 1) # Global max pooling along the time dimension (dim = 1)
        output = self.linear2(global_max_pool)
    
        return output


def training_Classifier(model, training_loader, validation_loader, optimizer, num_epochs):
    for epoch in range(num_epochs):
        total_loss = 0
        total = 0
        correct = 0

        for input, label, mask in training_loader:
            optimizer.zero_grad()
            output = model(input, mask)
            loss = F.cross_entropy(output, label)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

            batch_accuracy = 100 * correct / total
        
        epoch_loss = total_loss / len(training_loader)
        epoch_accuracy = 100 * correct / total
        
        validation_accuracy = evaluate_Classifier(model, validation_loader)

        print(f'Epoch [{epoch+1}/{num_epochs}], Epoch Loss: {epoch_loss:.4f}, Epoch Accuracy: {epoch_accuracy:.2f}%, Validation Accuracy: {validation_accuracy:.2f}%')
        

def evaluate_Classifier(model, validation_loader):
    model.eval()  # Evaluation mode
    total = 0
    correct = 0

    with torch.no_grad():  # No gradient calculation
        for input, label, mask in validation_loader:
            output = model(input, mask)
            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

    accuracy = 100 * correct / total
    
    return accuracy


# Load data
(x_train, y_train), (x_val, y_val), (i2w, w2i), numcls = load_imdb(final = False)
training_dataset = IMDBDataset(x_train, y_train)
validation_dataset = IMDBDataset(x_val, y_val)
training_loader = DataLoader(training_dataset, batch_size = 200, shuffle = False, collate_fn = pad_convert)
validation_loader = DataLoader(validation_dataset, batch_size = 200, shuffle = False, collate_fn = pad_convert)


Classifier_model = Classifier(input_size = len(i2w), embedding_size = 300, hidden_size = 300, output_size = 2)
optimizer = optim.Adam(Classifier_model.parameters(), lr=0.001)
training_Classifier(Classifier_model, training_loader, validation_loader, optimizer, num_epochs = 5)
