import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerFast,
    StoppingCriteria,
    StoppingCriteriaList,
)
import datasets

pythia = "EleutherAI/pythia-160m"
rwkv = "RWKV/HF_v5-Eagle-7B"
DEVICE = "cuda" if torch.cuda.is_available() else "mps"


class Stop(StoppingCriteria):
    def __init__(
        self, start_length: int, stops: list, tokenizer: PreTrainedTokenizerFast
    ):
        self.start_length = start_length
        self.eof_strings = stops
        self.tokenizer = tokenizer

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        """Returns true if all generated sequences contain any of the end-of-function strings."""
        decoded_generations = self.tokenizer.batch_decode(
            input_ids[:, self.start_length :]
        )
        done = []
        for decoded_generation in decoded_generations:
            done.append(
                any(
                    [
                        stop_string in decoded_generation
                        for stop_string in self.eof_strings
                    ]
                )
            )
        return all(done)


def process_(
    data: datasets.Dataset,
    tokenizer: PreTrainedTokenizerFast = None,
    model=None,
    until: list = None,
) -> datasets.Dataset:
    prompt = (
        lambda x: f"Extract the Author name from the message. If there is not one return None:\nExample 1:\nLEAVES OF GRASS (1850-1881) by Walt Whitman, with an introd. by Stuart P. Sherman. (The Modern student's library, American division) © on introd.; 27Jan22, A654456. R57007, 9Jan50, Ruth Sherman (W)\nResponce: Walt Whitman\nExample 2:\nMACHAON, by E. F. Benson. pub. abroad in Hutchinson's magazine, Jan. 1923; illus. by Blam,\nResponce:\nE. F. Benson\nExample 3:\nINTERNATIONAL CORRESPONDENCE SCHOOLS. Commercial signs. Instruction paper. Serial 2086. 1st ed.\nResponce:\nNone\nExample 4:\n{x}\nResponce:\n"
    )
    inputs = [prompt(x) for x in data["full_text"]]
    ctx_lens = [len(x) for x in inputs]
    inputs = tokenizer(inputs, return_tensors="pt", padding=True).to(DEVICE)
    start_length = inputs.input_ids.size(1)
    outputs = model.generate(
        **inputs,
        max_new_tokens=20,
        do_sample=True,
        temperature=1.0,
        top_p=0.3,
        top_k=0,
        stopping_criteria=StoppingCriteriaList(
            [
                Stop(
                    stops=until,
                    tokenizer=tokenizer,
                    start_length=start_length,
                )
            ]
        ),
    )

    processed = [
        tokenizer.decode(y, skip_special_tokens=True)[ctx_len:]
        for y, ctx_len in zip(outputs.tolist(), ctx_lens)
    ]
    output = []
    for term in processed:
        for stop in until:
            term = term.split(stop)[0]
        output.append(term)
    data["author"] = processed
    return data


model = AutoModelForCausalLM.from_pretrained(
    pythia, trust_remote_code=True, torch_dtype=torch.float16
).to(DEVICE)


if __name__ == "__main__":
    data = (
        datasets.load_dataset("baber/cce-renewals", "unmatched")["train"]
        .filter(lambda x: x["author"] is None)
        .select(list(range(30)))
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "EleutherAI/pythia-160m", trust_remote_code=True
    )
    try:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"
    except:
        pass
    data = data.map(
        process_,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer, "model": model, "until": ["\n", "\n\n"]},
        batch_size=8,
    )
    data.to_parquet("authors")
