import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report as skl_classification_report


sentences = [
    "The_DET cat_NOUN sleeps_VERB",
    "A_DET dog_NOUN barks_VERB",
    "The_DET dog_NOUN sleeps_VERB",
    "My_DET dog_NOUN runs_VERB fast_ADV",
    "A_DET cat_NOUN meows_VERB loudly_ADV",
    "Your_DET cat_NOUN runs_VERB",
    "The_DET bird_NOUN sings_VERB sweetly_ADV",
    "A_DET bird_NOUN chirps_VERB"
]

word2idx = {"<PAD>": 0}
tag2idx = {}
training_data = []

for sentence in sentences:
    tokens = sentence.split()
    words, tags = [], []
    for token in tokens:
        word, tag = token.rsplit("_", 1)
        if word not in word2idx:
            word2idx[word] = len(word2idx)
        if tag not in tag2idx:
            tag2idx[tag] = len(tag2idx)
        words.append(word2idx[word])
        tags.append(tag2idx[tag])
    training_data.append((words, tags))

idx2tag = {v: k for k, v in tag2idx.items()}
max_len = max(len(x[0]) for x in training_data)

def pad(seq, max_len, pad_val=0):
    return seq + [pad_val] * (max_len - len(seq))

# Split
train_data, test_data = train_test_split(training_data, test_size=0.25, random_state=42)
X_train = [pad(words, max_len) for words, _ in train_data]
Y_train = [pad(tags, max_len) for _, tags in train_data]
X_test = [pad(words, max_len) for words, _ in test_data]
Y_test = [pad(tags, max_len) for _, tags in test_data]


class PosDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.Y = torch.tensor(Y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

train_loader = DataLoader(PosDataset(X_train, Y_train), batch_size=2, shuffle=True)
test_loader = DataLoader(PosDataset(X_test, Y_test), batch_size=1)

class LSTMTagger(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim=64, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, tagset_size)

    def forward(self, x):
        emb = self.embedding(x)
        lstm_out, _ = self.lstm(emb)
        tag_space = self.fc(lstm_out)
        return tag_space

vocab_size = len(word2idx)
tagset_size = len(tag2idx)

model = LSTMTagger(vocab_size, tagset_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(20):
    model.train()
    total_loss = 0
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs.view(-1, tagset_size), y_batch.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")


def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            outputs = model(x_batch)
            predictions = torch.argmax(outputs, dim=-1)
            mask = y_batch != 0  # Ignore padding
            correct += (predictions[mask] == y_batch[mask]).sum().item()
            total += mask.sum().item()
    return correct / total if total > 0 else 0.0

accuracy = evaluate(model, test_loader)
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

true_tags = []
pred_tags = []

model.eval()
with torch.no_grad():
    for x_batch, y_batch in test_loader:
        outputs = model(x_batch)
        preds = torch.argmax(outputs, dim=-1).squeeze(0)
        labels = y_batch.squeeze(0)
        seq_len = (labels != 0).sum().item()

        pred_seq = [idx2tag[i.item()] for i in preds[:seq_len]]
        true_seq = [idx2tag[i.item()] for i in labels[:seq_len]]

        pred_tags.append(pred_seq)
        true_tags.append(true_seq)

flat_true = [tag for seq in true_tags for tag in seq]
flat_pred = [tag for seq in pred_tags for tag in seq]

print("\nDetailed Tagging Performance:")
print(skl_classification_report(flat_true, flat_pred, digits=3, zero_division=0))
