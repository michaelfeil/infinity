from infinity_emb.transformer.utils import get_lengths_with_tokenize


def test_get_lengths_with_tokenize():
    assert get_lengths_with_tokenize(["hi", "you"]) == ([2, 3], 5)
