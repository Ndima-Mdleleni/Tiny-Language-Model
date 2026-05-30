import numpy as np

embeddings = {
    "king": np.array([0.9, 0.5]),
    "queen": np.array([0.9, 0.1]),
    "man": np.array([0.2, 0.5]),
     "woman": np.array([0.2, 0.1])
}

result = embeddings["king"] - embeddings["man"] + embeddings["woman"]

print("king - man + woman =", result)
print("queen embedding    =", embeddings["queen"])