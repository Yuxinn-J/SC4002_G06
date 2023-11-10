import argparse
import time
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD, Adagrad, Adam, RMSprop
from datasets import Dataset
import gensim.downloader
import torch.nn.init as init
import numpy as np

from seqeval.metrics import f1_score
from seqeval.metrics import classification_report
from seqeval.scheme import IOB1

# Define the argument parser for hyperparameters
parser = argparse.ArgumentParser(description='Model Hyperparameters')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate for training')
parser.add_argument('--batch_size', type=int, default=16, help='batch size for training')
parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
parser.add_argument('--hidden_dim', type=int, default=512 , help='hidden dimension size')
parser.add_argument('--num_layers', type=int, default=2, help='number of layers')


def read_conll_file(file_path):
    with open(file_path, "r") as f:
        content = f.read().strip()
        sentences = content.split("\n\n")
        data = []
        for sentence in sentences:
            tokens = sentence.split("\n")
            token_data = []
            for token in tokens:
                token_data.append(token.split())
            data.append(token_data)
    return data

# prepare data
def convert_to_dataset(data, label_map):
    formatted_data = {"tokens": [], "ner_tags": []}
    for sentence in data:
        tokens = [token_data[0] for token_data in sentence]
        ner_tags = [label_map[token_data[3]] for token_data in sentence]
        formatted_data["tokens"].append(tokens)
        formatted_data["ner_tags"].append(ner_tags)
    return Dataset.from_dict(formatted_data)


def prepare_w2v():
    w2v = gensim.models.KeyedVectors.load('word2vec-google-news-300.model', mmap='r')
    word2idx = w2v.key_to_index

    # Add '<UNK>' and '<PAD>' tokens to the vocabulary index
    word2idx['<UNK>'] = len(word2idx)
    word2idx['<PAD>'] = len(word2idx)

    # add the '<UNK>' word to the vocabulary of the Word2Vec model 
    # initialize it with the average of all word vectors int he pretrained embeddings.
    unk_vector = np.mean(w2v.vectors, axis=0)
    w2v.vectors = np.vstack([w2v.vectors, unk_vector])

    # add the '<PAD>' word to the vocabulary of the Word2Vec model 
    # initialize it with a row of zeros in the vectors matrix.
    w2v.vectors = np.vstack([w2v.vectors, np.zeros(w2v.vectors[0].shape)])
    return word2idx, w2v

word2idx, w2v = prepare_w2v()

# Map words to Indices
def sentence_to_indices(sentence, vocab):
    return [vocab.get(word, vocab.get('<UNK>')) for word in sentence]

tag2idx = {
    'B-LOC': 0,
    'B-MISC': 1,
    'B-ORG': 2,
    'I-LOC': 3,
    'I-MISC': 4,
    'I-ORG': 5,
    'I-PER': 6,
    'O': 7,
    'PAD': 8
}
idx2tag = {v: k for k, v in tag2idx.items()}
def idx_to_tags(indices):
    return [idx2tag[idx] for idx in indices]


class NERDataset(Dataset):
    def __init__(self, sentences, tags, vocab):
        self.sentences = [torch.tensor(sentence_to_indices(sentence, vocab)) for sentence in sentences]
        self.tags = [torch.tensor(tag) for tag in tags]

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx], self.tags[idx]

def collate_fn(batch):
    sentences, tags = zip(*batch)
    sentences_padded = pad_sequence(sentences, batch_first=True, padding_value=word2idx['<PAD>'])
    tags_padded = pad_sequence(tags, batch_first=True, padding_value=tag2idx['PAD'])
    return sentences_padded, tags_padded


class BiLSTMNERModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_of_layers, output_dim):
        super(BiLSTMNERModel, self).__init__()
        embedding_matrix = torch.FloatTensor(w2v.vectors)
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, padding_idx=word2idx['<PAD>'], freeze=True)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_of_layers, batch_first=True, bidirectional=True)
        # Initialize LSTM weights and biases
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                
        # Initialize fully connected layer
        self.fc = nn.Linear(hidden_dim*2, output_dim)
        init.xavier_uniform_(self.fc.weight)
        self.fc.bias.data.fill_(0)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        tag_space = self.fc(lstm_out)
        tag_scores = torch.log_softmax(tag_space, dim=-1)
        return tag_scores


class EarlyStopper:
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.max_f1 = 0

    def early_stop(self, f1):
        if f1 > self.max_f1:
            self.max_f1 = f1
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def evaluate(model, validation_loader, device):
    print("evaluate.....")
    # Evaluate on the validation dataset
    # Placeholder to store true and predicted tags
    y_true = [] # true tags
    y_pred = [] # predicted tags
        
    # Evaluate the model on the validation dataset
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        for sentences, tags in validation_loader:
            # Move the data to the GPU
            sentences, tags = sentences.to(device), tags.to(device)
            tag_scores = model(sentences)
            # print(tag_scores.device)
            predictions = tag_scores.argmax(dim=-1).tolist()
            # print(predictions)
                
            # Convert index to tags
            # Note: filtering out padding tokens
            for sentence, true_seq, pred_seq in zip(sentences, tags.tolist(), predictions):
                valid_length = (sentence != word2idx['<PAD>']).sum().item()
                true_tags = [idx2tag[idx] for idx in true_seq[:valid_length]]
                pred_tags = [idx2tag[idx] for idx in pred_seq[:valid_length]]
                y_true.append(true_tags)
                y_pred.append(pred_tags)
        
    # Compute F1 score
    f1 = f1_score(y_true, y_pred)
    return f1


def test(model, test_loader, device):
    # Placeholder to store true and predicted tags for the test set
    y_true_test = []
    y_pred_test = []

    # Evaluate the model on the test dataset
    model.eval()  # Set the model to evaluation mode

    # Start the clock for timing the test evaluation
    start_time = time.time()

    with torch.no_grad():
        for sentences, tags in test_loader:
            sentences = sentences.to(device)
            tag_scores = model(sentences)
            predictions = tag_scores.argmax(dim=-1).tolist()
            
            # Convert index to tags
            # Note: filtering out padding tokens
            for sentence, true_seq, pred_seq in zip(sentences, tags.tolist(), predictions):
                valid_length = (sentence != word2idx['<PAD>']).sum().item()
                true_tags = [idx2tag[idx] for idx in true_seq[:valid_length]]
                pred_tags = [idx2tag[idx] for idx in pred_seq[:valid_length]]
                y_true_test.append(true_tags)
                y_pred_test.append(pred_tags)

    # Stop the clock after the test evaluation
    end_time = time.time()
    test_time = end_time - start_time

    # Compute F1 score for the test set
    f1_test = f1_score(y_true_test, y_pred_test)#
    report_test = classification_report(y_true_test, y_pred_test)

    print("F1 Score on Test Set:", f1_test)
    # print("Classification Report on Test Set:\n", report_test)
    # print(f"Test evaluation time: {test_time} seconds")
    return f1_test

train_data = read_conll_file("/mnt/lustre/yuxin/SC4002_G06/datasets/CoNLL2003/eng.train")
validation_data = read_conll_file("/mnt/lustre/yuxin/SC4002_G06/datasets/CoNLL2003/eng.testa")
test_data = read_conll_file("/mnt/lustre/yuxin/SC4002_G06/datasets/CoNLL2003/eng.testb")
    
label_list = sorted(list(set([token_data[3] for sentence in train_data for token_data in sentence])))
label_map = {label: i for i, label in enumerate(label_list)}

train_dataset = convert_to_dataset(train_data, label_map)
validation_dataset = convert_to_dataset(validation_data, label_map)
test_dataset = convert_to_dataset(test_data, label_map)

# Create PyTorch datasets and data loaders
train_dataset = NERDataset(train_dataset['tokens'], train_dataset['ner_tags'], word2idx)
validation_dataset = NERDataset(validation_dataset['tokens'], validation_dataset['ner_tags'], word2idx)
test_dataset = NERDataset(test_dataset['tokens'], test_dataset['ner_tags'], word2idx)


# Parse arguments
args = parser.parse_args()

EMBEDDING_DIM = w2v[0].shape[0]
VOCAB_SIZE = len(word2idx)
TAGSET_SIZE = len(tag2idx)
MAX_EPOCHS = 50
learning_rate = args.learning_rate
batch_size = args.batch_size
optimizer_choice = args.optimizer # TODO: str -> function call
hidden_dim = args.hidden_dim
num_layers = args.num_layers

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

# set up model
model = BiLSTMNERModel(EMBEDDING_DIM, hidden_dim, num_layers , TAGSET_SIZE).to(device)
loss_function = nn.NLLLoss(ignore_index=tag2idx['PAD']).to(device)
optimizer = RMSprop(model.parameters(), lr=learning_rate)

# Initialize dataloaders
train_loader = DataLoader(train_dataset, batch_size, shuffle=True, collate_fn=collate_fn)
validation_loader = DataLoader(validation_dataset, batch_size, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size, shuffle=False, collate_fn=collate_fn)

# Initialize early stopper
early_stopper = EarlyStopper()

# Metrics and checkpoint initialization
best_f1_score = 0
metrics = {'f1': [], 'loss': [], 'epoch_time': []}
total_start_time = time.time()

# Training loop
for epoch in range(MAX_EPOCHS):
    epoch_start_time = time.time()
    
    total_loss = 0
    model.train()  # Make sure the model is in training mode
    for sentences, tags in train_loader:
        sentences, tags = sentences.to(device), tags.to(device)  # Move data to GPU
        model.zero_grad()
        tag_scores = model(sentences)
        loss = loss_function(tag_scores.view(-1, TAGSET_SIZE), tags.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    epoch_loss = total_loss / len(train_loader)
    metrics['loss'].append(epoch_loss)

    f1 = evaluate(model, validation_loader, device)
    metrics['f1'].append(f1)
    
    print(f"Epoch {epoch+1}, loss: {epoch_loss}, f1_score: {f1}")
    
    # Early stopping check
    if early_stopper.early_stop(f1):
        print(f"Stopping early at epoch {epoch+1}")
        break
        
    epoch_time = time.time() - epoch_start_time
    metrics['epoch_time'].append(epoch_time)

    f1_test = test(model, test_loader, device)
    print(f"Epoch {epoch+1}, loss: {epoch_loss}, f1_score: {f1}, f1_test: {f1_test}")
    # Save the best model
    if f1_test > best_f1_score:
        best_f1_score = f1_test
        best_model_state = model.state_dict()
        torch.save(best_model_state, 'best_model-tr.pth')

total_train_time = time.time() - total_start_time
print(f"Total training time: {total_train_time}s")

# Example of saving the model (assuming your model variable is named 'model')
torch.save(model.state_dict(), 'model.pth')
print('Model saved successfully!')

# Make sure to replace 'model' with the actual variable name of your model in the script
