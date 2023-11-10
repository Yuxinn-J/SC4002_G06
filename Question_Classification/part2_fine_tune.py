import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import gensim.downloader
import time
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader


def sentence_to_indices(sentence, vocab):
    return [vocab.get(word, vocab.get('<UNK>')) for word in sentence]


class TRECDataset(Dataset):
    def __init__(self, sentences, labels, vocab):
        
        self.sentences = [torch.tensor(sentence_to_indices(sentence, vocab)) for sentence in sentences]
        self.labels = [torch.tensor(label) for label in labels]
        
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        return self.sentences[idx], self.labels[idx]


def collate_fn(batch):
    sentences, labels = zip(*batch)
    sentences_padded = pad_sequence(sentences, batch_first=True, padding_value=word2idx['<PAD>'])
    return sentences_padded, labels


def train_model(model, train_loader, dev_loader, num_epochs, loss_function, optimizer, patience):
    train_losses = []
    dev_losses = []
    train_accuracies = []
    dev_accuracies = []
    y_true = []
    y_pred = []
    best_accuracy = 0.0
    no_improvement_count = 0

    start_time = time.time()

    print(100*"-")

    print("Model Train Starts: " + '\n')
    # open a txt file for logging
    print(str(model) + '\n')

    print(100*"-")
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for sentences, label_tuple in train_loader:
            model.zero_grad()
            # labels = torch.stack(label_tuple)
            sentences = sentences.to(DEVICE)
            labels = torch.stack(label_tuple).to(DEVICE)

            predictions = model(sentences)
            
            loss = loss_function(predictions, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            _, predicted = torch.max(predictions, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_accuracies.append(100 * correct / total)
        train_losses.append(total_loss/len(train_loader))
        
        # Validation loop
        model.eval()
        dev_loss = 0
        correct_dev = 0
        total_dev = 0
        with torch.no_grad():
            for sentences, label_tuple in dev_loader:
                sentences = sentences.to(DEVICE)
                labels = torch.stack(label_tuple).to(DEVICE)
                predictions = model(sentences)
                
                loss = loss_function(predictions, labels)
                dev_loss += loss.item()
                
                _, predicted = torch.max(predictions, 1)
                total_dev += labels.size(0)
                correct_dev += (predicted == labels).sum().item()
                
                y_true.extend(labels)
                y_pred.extend(predicted)
        
        dev_accuracy = 100 * correct_dev / total_dev
        dev_accuracies.append(dev_accuracy)
        dev_losses.append(dev_loss/len(dev_loader))
        
        # Early stopping
        if dev_accuracy > best_accuracy:
            best_accuracy = dev_accuracy
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if no_improvement_count >= patience:
            print(f'Early stopping after {epoch+1} epochs with no improvement.')

            end_time = time.time()
            total_train_time = end_time - start_time
            print(f'Total training time: {total_train_time:.2f} seconds.')

            print(100*"-")

            break

    end_time = time.time()
    total_train_time = end_time - start_time
    print(f'Total training time: {total_train_time:.2f} seconds.')
    print("Final Results:")
    print(f"Train Loss: {train_losses[-1]:.4f}")
    print(f"Dev Loss: {dev_losses[-1]:.4f}")
    print(f"Train Accuracy: {train_accuracies[-1]:.2f}%")
    print(f"Dev Accuracy: {dev_accuracies[-1]:.2f}%")
    print(100*"-")

    return train_losses, dev_losses, train_accuracies, dev_accuracies


def evaluate_model(model, test_loader, loss_function):
    model.eval()
    test_loss = 0
    correct_test = 0
    total_test = 0
    y_true_test = []
    y_pred_test = []

    print(100*"-")

    print("Model Test Starts: " + '\n')

    print(100*"-")
    
    with torch.no_grad():
        for sentences, tag_tuple in test_loader:
            sentences = sentences.to(DEVICE)
            labels = torch.stack(tag_tuple).to(DEVICE)
            predictions = model(sentences)
            
            loss = loss_function(predictions, labels)
            test_loss += loss.item()
            
            _, predicted = torch.max(predictions, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()
            
            y_true_test.extend(labels)
            y_pred_test.extend(predicted)

    test_accuracy = 100 * correct_test / total_test
    test_loss /= len(test_loader)

    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

    print(100*"-")

    return test_loss, test_accuracy


# Define model 6 architecture
class QuestionClassifierModel6(nn.Module):
    def __init__(self, embedding_dim, output_dim, num_layers, num_heads, dropout_rate):
        super(QuestionClassifierModel6, self).__init__()
 
        
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, padding_idx=word2idx['<PAD>'], freeze=True)
        
        # Hidden Layer
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, dropout=dropout_rate)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output Layer
        self.output = nn.Linear(embedding_dim, output_dim)
    
    def forward(self, x):
        # x shape: batch_size x seq_length
        x = self.embedding(x)  # Now, shape: batch_size x seq_length x embedding_dim
        
        # Aggregation Layer
        x = self.transformer_encoder(x)
        x = x.mean(dim=1) # Global average pooling

        # Output Layer
        x = self.output(x)  # Now, shape: batch_size x output_dim
        
        return x


# Load training and development datasets
df_train=pd.read_csv("/p/scratch/ccstdl/xu17/jz/SC4002_G06/datasets/TREC/train.csv")
df_train["text"] = df_train["text"].str.lower()

# Load test dataset
df_test=pd.read_csv("/p/scratch/ccstdl/xu17/jz/SC4002_G06/datasets/TREC/test.csv")
df_test["text"] = df_test["text"].str.lower()

# Split the training dataset to create a development set of 500 examples
train_data, dev_data = train_test_split(df_train, test_size=500, random_state=42)

# Rename the test dataset to synchronize the namings
test_data = df_test

# Download the "glove-twitter-25" embeddings
w2v = gensim.downloader.load('word2vec-google-news-300')

word2idx = w2v.key_to_index

# Add '<UNK>' and '<PAD>' tokens to the vocabulary index
word2idx['<UNK>'] = len(word2idx)
word2idx['<PAD>'] = len(word2idx)

print(f"word2idx['<UNK>']: {word2idx['<UNK>']}")
print(f"word2idx['<PAD>']: {word2idx['<PAD>']}")

# Get unique coarse labels
unique_labels = train_data['label-coarse'].unique()

# Randomly select 4 classes
np.random.seed(19260817)
selected_labels = np.random.choice(unique_labels, size=4, replace=False)

train_data['new_label'] = train_data['label-coarse'].apply(lambda x: x if x in selected_labels else "OTHERS")
dev_data['new_label'] = dev_data['label-coarse'].apply(lambda x: x if x in selected_labels else "OTHERS")
test_data['new_label'] = test_data['label-coarse'].apply(lambda x: x if x in selected_labels else "OTHERS")

# Encode labels for easier reference in the following part
label_encoder = LabelEncoder()

test_data['new_label'] = test_data['new_label'].astype(str)
test_data["label_transformed"] = label_encoder.fit_transform(test_data['new_label'])

train_data['new_label'] = train_data['new_label'].astype(str)
train_data["label_transformed"] = label_encoder.fit_transform(train_data['new_label'])

dev_data['new_label'] = dev_data['new_label'].astype(str)
dev_data["label_transformed"] = label_encoder.fit_transform(dev_data['new_label'])

# Create PyTorch datasets and data loaders
train_dataset = TRECDataset(train_data['text'], train_data['label_transformed'], word2idx)
dev_dataset = TRECDataset(dev_data['text'], dev_data['label_transformed'], word2idx)
test_dataset = TRECDataset(test_data['text'], test_data['label_transformed'], word2idx)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
dev_loader = DataLoader(dev_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)


# Load embedding layer
weights = torch.FloatTensor(w2v.vectors)

# Build nn.Embedding() layer
embedding = nn.Embedding.from_pretrained(weights)
# embedding = nn.Embedding.from_pretrained(embedding_matrix, padding_idx=vocab.get('<PAD>', None), freeze=True)
embedding.requires_grad = False
padding_idx=word2idx['<PAD>']
embedding_matrix = torch.FloatTensor(w2v.vectors)
if padding_idx >= embedding_matrix.size(0):
    # Add a zero vector for the padding token
    padding_vector = torch.zeros(2, embedding_matrix.size(1))
    embedding_matrix = torch.cat([embedding_matrix, padding_vector], dim=0)

# Define hyperparameters
EMBEDDING_DIM = 300
VOCAB_SIZE = len(word2idx)
LABELSET_SIZE = 5
DROPOUT_RATE = 0.1

# Possible values for experiments
NUM_HEADS_OPTIONS = [6, 10, 15]
HIDDEN_LAYERS_OPTIONS = [3, 4, 5]
LR_LAYERS_OPTIONS = [0.0001, 0.001, 0.01]

# Define the rest of the parameters and functions needed for training
loss_function = nn.CrossEntropyLoss()
optimizer = None  # This will be defined inside the loop for each model
num_epochs = 200
patience = 10
DEVICE = torch.device("cuda")

best_dev_loss = float('inf')

for num_heads in NUM_HEADS_OPTIONS:
    for hidden_layers in HIDDEN_LAYERS_OPTIONS:
        for lr in LR_LAYERS_OPTIONS:
            print(100*"-")

            print(f"Training with NUM_HEADS: {num_heads}, HIDDEN_LAYERS: {hidden_layers}, LR: {lr}")

            print(100*"-")
            
            # Initialize model with current set of hyperparameters
            model = QuestionClassifierModel6(EMBEDDING_DIM, LABELSET_SIZE, hidden_layers, num_heads, DROPOUT_RATE)
            
            # Define optimizer for the current model
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            
            # Use nn.DataParallel for multi-GPU training
            model = nn.DataParallel(model, device_ids=[0, 1, 2, 3]).cuda()
            
            # Train model with the current set of hyperparameters
            train_losses, dev_losses, train_accuracies, dev_accuracies = train_model(
                model, train_loader, dev_loader, num_epochs, loss_function, optimizer, patience
            )

            # If the current model is the best so far, save its weights

            if min(dev_losses) < best_dev_loss:
                best_dev_loss = min(dev_losses)
                torch.save(model.state_dict(), f'/p/scratch/ccstdl/xu17/jz/SC4002_G06/Question_Classification/model_weights_transformers/best_model_{num_heads}_{hidden_layers}_{lr}.pt')

            # Test model
            test_loss, test_accuracy = evaluate_model(model, test_loader, loss_function)
