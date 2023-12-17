"""
Idea:

# Pairs:
query: str = "where is paris"
docs: list[str] = ["Paris is in france","lets have chocolate", "berlin is a capital city"]

pairs = [(query, doc) for doc in docs]


scores = model.predict(pairs, activation_fct=lambda x: x)

if normalize:
    scores = np.sigmoid(scores)

"""