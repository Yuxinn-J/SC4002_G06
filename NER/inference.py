import torch
import torch.nn as nn
import argparse
import gensim.downloader
import numpy as np

# Define the argument parser and add the interactive flag
parser = argparse.ArgumentParser(description="Run inference on a test sentence using a saved model checkpoint.")
parser.add_argument('test_sentence', type=str, nargs='?', default='', help="The sentence you want to test, wrapped in quotes.")
parser.add_argument('--interactive', action='store_true', help="Run the script in interactive mode.")
args = parser.parse_args()


# Download the embeddings
print("Load word2vec embedding...")
w2v = gensim.models.KeyedVectors.load('word2vec-google-news-300.model', mmap='r')

word2idx = w2v.key_to_index
word2idx['<UNK>'] = len(word2idx)
word2idx['<PAD>'] = len(word2idx)
unk_vector = np.mean(w2v.vectors, axis=0)
w2v.vectors = np.vstack([w2v.vectors, unk_vector])
w2v.vectors = np.vstack([w2v.vectors, np.zeros(w2v.vectors[0].shape)])
embedding_matrix = torch.FloatTensor(w2v.vectors)


class BiLSTMNERModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_of_layers, output_dim):
        super(BiLSTMNERModel, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, padding_idx=word2idx['<PAD>'], freeze=True)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_of_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim*2, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        tag_space = self.fc(lstm_out)
        tag_scores = torch.log_softmax(tag_space, dim=-1)
        return tag_scores
        
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

# Help functions
def idx_to_tags(indices):
    return [idx2tag[idx] for idx in indices]

def sentence_to_indices(sentence, vocab):
    return [vocab.get(word, vocab.get('<UNK>')) for word in sentence]


# Load the model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Load pre-trained checkpoint...")
model_checkpoint_path = '/mnt/lustre/yuxin/SC4002_G06/NER/best_model_8522.pth'  
model = BiLSTMNERModel(embedding_dim=w2v[0].shape[0], hidden_dim=256, num_of_layers=3, output_dim=len(tag2idx))
model.load_state_dict(torch.load(model_checkpoint_path, map_location=device))
model.to(device)
model.eval()


def inference(sentence, model, device):
    # Tokenize the sentence
    tokens = sentence.split()

    # Convert tokens to indices
    token_indices = torch.tensor([sentence_to_indices(tokens, word2idx)]).to(device)

    # Get predictions from the model
    with torch.no_grad():
        tag_scores = model(token_indices)
        predictions = tag_scores.argmax(dim=-1).tolist()[0]

    # Convert index to tags
    predicted_tags = idx_to_tags(predictions)

    # Prepare aligned output
    aligned_output = "\n".join([f"{token}: {tag}" for token, tag in zip(tokens, predicted_tags)])
    return aligned_output

# Interactive loop for user input
if args.interactive:
    # Interactive mode
    while True:
        test_sentence = input("Enter a sentence for NER classification (or type 'exit' to quit): ")
        if test_sentence.lower() == 'exit':
            print("Exiting...")
            break
        aligned_output = inference(test_sentence, model, device)
        print("Predicted Tags:")
        print(aligned_output)
        print("\n")
else:
    # One-shot inference mode
    if args.test_sentence:
        aligned_output = inference(args.test_sentence, model, device)
        print("Predicted Tags:")
        print(aligned_output)
    else:
        print("No sentence provided for one-shot inference.")



