"""small code to generate a sequence classifier from a causal language model
Copyright Michael Feil, 2025, MIT License
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
BASE_NAME = "michaelfeil"

@torch.no_grad()
def convert_to_sequence_classifier(
    model_name: str, slice_single_token_list: None | set[int] = None
) -> torch.nn.Module:
    """Convert a causal language model to a sequence classifier. build for mixedbread-ai/mxbai-rerank-base-v2

    Example usage:
        The model classifies a static prompt (prefill) and uses the next token distribution of no and yes to classify the prompt.
        https://github.com/mixedbread-ai/mxbai-rerank/blob/ca0c55d03770d9bb183ca759850bf7cdfbcc9f50/mxbai_rerank/mxbai_rerank_v2.py#L34
        a good example is:
            "You are a search relevance expert who evaluates how well documents match search queries. For each query-document pair, carefully analyze the semantic relationship between them, then provide your binary relevance judgment (0 for not relevant, 1 for relevant).\nRelevance:"
        thus, we need to get the next token distribution of no and yes to classify the prompt.
        the token ids of no and yes are [15, 16] == [tokenizer("0").input_ids, tokenizer("1").input_ids]
    Args:
        model_name (str): model name for the causal language model / AutoModelForCausalLM
        slice_single_token_list (None | set[int], optional): slice the lm_head to a subset of tokens. Defaults to None, which will give vocab_size outputs.

    Returns:
        AutoModelForSequenceClassification: model classifier
    """
    model_lm = AutoModelForCausalLM.from_pretrained(model_name)
    model_lm.model = None  # free up memory
    assert model_lm.lm_head.bias is None
    model_classifier = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def get_input_ids(x):
        return tokenizer(x, return_tensors=None, add_special_tokens=False)["input_ids"]

    # tokenizer select tokens
    num_total_tokens = model_lm.lm_head.out_features
    if slice_single_token_list is not None:
        slice_single_token_list = list(sorted(set(slice_single_token_list)))
        assert max(slice_single_token_list) <= num_total_tokens
        assert min(slice_single_token_list) >= 0

    tokens = tokenizer.convert_ids_to_tokens(range(num_total_tokens))
    if slice_single_token_list is not None:
        tokens_selected = [tokens[i] for i in slice_single_token_list]
        # slice the score head and build a linear on the fly
        new_score = model_lm.lm_head.weight[slice_single_token_list]
    else:
        tokens_selected = tokens
        new_score = model_lm.lm_head.weight
    num_tokens = len(tokens_selected)
    # add classifier head from lm head
    model_classifier.config.num_labels = num_tokens
    linear = torch.nn.Linear(model_lm.lm_head.in_features, num_tokens, bias=False)
    linear.weight.data = new_score
    model_classifier.score = linear
    # add id2label and label2id
    model_classifier.config.id2label = {
        num: label for num, label in enumerate(tokens_selected)
    }
    model_classifier.config.label2id = {
        label: num for num, label in enumerate(tokens_selected)
    }
    return model_classifier


def as_no_id_yes_id(
    model_name: str = "mixedbread-ai/mxbai-rerank-base-v2",
    no: str = "0",
    yes: str = "1",
) -> tuple[int, int]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    no_id_yes_id = [tokenizer(no, add_special_tokens=False).input_ids, tokenizer(yes, add_special_tokens=False).input_ids]
    assert len(no_id_yes_id[0]) == 1
    assert len(no_id_yes_id[1]) == 1
    return no_id_yes_id[0][0], no_id_yes_id[1][0]

def only_yes_id(
    model_name: str = "mixedbread-ai/mxbai-rerank-base-v2",
    yes: str = "1",
) -> tuple[int]:
    """Get the id of the yes token."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    yes_id = tokenizer(yes, add_special_tokens=False).input_ids
    assert len(yes_id) == 1
    return (yes_id[0],)


def upload_and_convert(
    model_name: str = "mixedbread-ai/mxbai-rerank-base-v2",
    no: str = "0",
    yes: str = "1",
    uses_no_and_yes: bool = True,
):
    """Upload the converted sequence classifier to the hub."""
    if not uses_no_and_yes:
        no_id_yes_id = only_yes_id(model_name, yes)
    else:
        no_id_yes_id = as_no_id_yes_id(model_name, no, yes)
    split_name = model_name.split("/")[1]
    model_cls = convert_to_sequence_classifier(f"{model_name}", no_id_yes_id)
    model_cls = model_cls.to(torch.float16)

    from huggingface_hub import HfApi, snapshot_download

    snapshot_download(f"{model_name}", local_dir=f"./{split_name}")
    model_cls.save_pretrained(f"./{split_name}")

    api = HfApi()
    api.create_repo(repo_id=f"{BASE_NAME}/{split_name}-seq", exist_ok=True)
    api.upload_folder(
        repo_id=f"{BASE_NAME}/{split_name}-seq",
        folder_path=f"./{split_name}",
    )


def test_convert_to_seq_classifier(
    name="mixedbread-ai/mxbai-rerank-base-v2", no="0", yes="1"
):
    no_id_yes_id = as_no_id_yes_id(name, no, yes)

    with torch.no_grad():
        model_cls = convert_to_sequence_classifier(name, no_id_yes_id)
        model_lm = AutoModelForCausalLM.from_pretrained(name)

        tokenizer = AutoTokenizer.from_pretrained(name)

        example = {
            "instruction": "You are a search relevance expert who evaluates how well documents match search queries. For each query-document pair, carefully analyze the semantic relationship between them, then provide your binary relevance judgment (0 for not relevant, 1 for relevant).",
            "query": ["What is the capital of France?"],
            "document": [
                "The capital of France is Paris.",
                "Who is the president of France?",
            ],
        }
        examples_formatted = [
            f"{example['instruction']}\n{example['query'][0]}\n{example['document'][0]}",
            f"{example['instruction']}\n{example['query'][0]}\n{example['document'][1]}",
        ]
        # create pytorch tensors
        for example_formatted in examples_formatted:
            tokenized = tokenizer(
                example_formatted,
                return_tensors="pt",
                truncation=True,
            )
            # forward pass
            output = model_cls(**tokenized).logits
            output_lm = model_lm(**tokenized).logits[0, -1, no_id_yes_id]
            print(output_lm)
            print(output)
            assert torch.allclose(output_lm, output)
            print("done")


if __name__ == "__main__":
    test_convert_to_seq_classifier(
        name="mixedbread-ai/mxbai-rerank-base-v2", no="0", yes="1"
    )
    test_convert_to_seq_classifier(name="Qwen/Qwen3-Reranker-0.6B", no="no", yes="yes")
    # upload_and_convert(model_name="mixedbread-ai/mxbai-rerank-base-v2", no="0", yes="1")
    upload_and_convert(model_name="Qwen/Qwen3-Reranker-0.6B", no="no", yes="yes")
    upload_and_convert(model_name="Qwen/Qwen3-Reranker-4B", no="no", yes="yes")
    upload_and_convert(model_name="Qwen/Qwen3-Reranker-8B", no="no", yes="yes")
