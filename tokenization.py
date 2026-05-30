import numpy as np

sentences = [
    "the cat sat",
    "the dog ran",
    "the cat ran"
]

vocab = {}
idx = 0
for sentence in sentences:
    for word in sentence.split():
        if word not in vocab:
            vocab[word] = idx
            idx += 1

vocab['[UNK]'] = len(vocab)

embedding_size = 4
embeddings = np.random.randn(len(vocab), embedding_size)

def tokenize(sentence, vocab):
    return [vocab.get(word, vocab['[UNK]']) for word in sentence.split()]

def embed_sentence(sentence, vocab, embeddings):
    token_ids = tokenize(sentence, vocab)
    return np.array([embeddings[idx] for idx in token_ids])

def add_positional_encoding(embedded):
    sequence_length = embedded.shape[0]
    embedding_size = embedded.shape[1]
    positional_encoding = np.zeros_like(embedded)
    for pos in range(sequence_length):
        for i in range(embedding_size):
         positional_encoding[pos][i] = pos / (10000 ** (2 * i / embedding_size))
    return embedded + positional_encoding

def attention_with_scores(query, key, value):
    scores = np.dot(query, key.T)
    scores = scores / np.sqrt(query.shape[-1])
    scores = np.exp(scores) / np.exp(scores).sum(axis=-1, keepdims=True)

    print("attention scores:")
    print("         the      cat      sat")
    for i, word in enumerate(["the", "cat", "sat"]):
        print(f"{word:8}", [f"{s:.2f}" for s in scores[i]])

    return np.dot(scores, value)

sentence = "the cat sat"
embedded = embed_sentence(sentence, vocab, embeddings)
positioned = add_positional_encoding(embedded)
output = attention_with_scores(positioned, positioned, positioned)