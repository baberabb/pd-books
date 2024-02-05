import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast
import datasets


def process_(
    data: datasets.Dataset, tokenizer: PreTrainedTokenizerFast = None, model=None
) -> datasets.Dataset:
    inputs = tokenizer(data["full_text"], return_tensors="pt", padding=True)
    outputs = model.generate(
        inputs["input_ids"],
        max_new_tokens=20,
        do_sample=True,
        temperature=1.0,
        top_p=0.3,
        top_k=0,
    )
    for x, y in zip(data, outputs):
        x["author"] = tokenizer.decode(y.tolist(), skip_special_tokens=True)
    return data


model = AutoModelForCausalLM.from_pretrained(
    "RWKV/HF_v5-Eagle-7B", trust_remote_code=True
).to(torch.float32)
tokenizer = AutoTokenizer.from_pretrained("RWKV/HF_v5-Eagle-7B", trust_remote_code=True)


if __name__ == "__main__":
    data = datasets.load_dataset("baber/cce-renewals", "unmatched")["train"].filter(
        lambda x: x["author"] is None
    )
    data = data.map(
        process_,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer, "model": model},
        batch_size=64,
    )
