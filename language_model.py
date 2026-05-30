import numpy as np
import torch
import torch.nn as nn

sentences = [
    "the cat sat on the mat",
    "the dog ran on the grass",
    "the cat ran fast",
    "the dog sat still"
]

vocab = {}
idx = 0
for sentence in sentences:
    for word in sentence.split():
        if word not in vocab:
            vocab[word] = idx
            idx += 1

vocab['[UNK]'] = len(vocab)

print("vocabulary:", vocab)
print("vocab size:", len(vocab))

class TinyLanguageModel(nn.Module):

    def __init__(self, vocab_size, embedding_size, hidden_size):
        super().__init__()
        self.embedding  = nn.Embedding(vocab_size, embedding_size)
        self.attention  = nn.MultiheadAttention(embedding_size, num_heads=2, batch_first=True)
        self.feedforward = nn.Sequential(
            nn.Linear(embedding_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, embedding_size)
        )
        self.output = nn.Linear(embedding_size, vocab_size)

    def forward(self, x):
        embedded        = self.embedding(x)                              
        attended, _     = self.attention(embedded, embedded, embedded)   
        refined         = self.feedforward(attended)                     
        scores          = self.output(refined)                           
        return scores

model = TinyLanguageModel(
    vocab_size     = len(vocab),
    embedding_size = 32,
    hidden_size    = 64
)

print(model)

def prepare_training_data(sentences, vocab):
    inputs  = []
    targets = []
    for sentence in sentences:
        words     = sentence.split()
        token_ids = [vocab.get(w, vocab['[UNK]']) for w in words]
        inputs.append(token_ids[:-1])
        targets.append(token_ids[1:])
    return inputs, targets

inputs, targets = prepare_training_data(sentences, vocab)

print("example input: ", inputs[0])
print("example target:", targets[0])
print("meaning: given", [list(vocab.keys())[i] for i in inputs[0]])
print("predict:       ", [list(vocab.keys())[i] for i in targets[0]])

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 300

for epoch in range(epochs):
    total_loss = 0
    for inp, tgt in zip(inputs, targets):
        inp_tensor = torch.tensor([inp], dtype=torch.long)   
        tgt_tensor = torch.tensor(tgt,  dtype=torch.long)

        optimizer.zero_grad()
        output     = model(inp_tensor)
        output     = output.squeeze(0)
        loss       = criterion(output, tgt_tensor)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if epoch % 50 == 0:
        print(f"Epoch {epoch} | Loss: {total_loss:.4f}")

def generate(model, vocab,start_word, max_words=6, temperature=0.3):
    model.eval()

    idx_to_word = {v: k for k, v in vocab.items()}

    current_ids = [vocab.get(start_word, vocab['[UNK]'])]
    generated = [start_word]

    with torch.no_grad():
        for _ in range(max_words):
            inp = torch.tensor([current_ids], dtype=torch.long)
            output = model(inp)

            last_scores = output[0, -1, :]


            last_scores = last_scores / temperature
            probs = torch.softmax(last_scores, dim=0)


            next_id = torch.multinomial(probs, 1).item()
            next_word = idx_to_word[next_id]

            generated.append(next_word)
            current_ids.append(next_id)

        
    return " ".join(generated)

print("\n--- generation ---")
print(generate(model, vocab, "the"))
print(generate(model, vocab, "cat"))
print(generate(model, vocab, "dog"))



    