import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast
import datasets


def process_(
    data: datasets.Dataset, tokenizer: PreTrainedTokenizerFast = None, model=None
) -> datasets.Dataset:
    prompt = f"Extract the Author name from the message. If there is not one return None:\nExample 1:\nLEAVES OF GRASS (1850-1881) by Walt Whitman, with an introd. by Stuart P. Sherman. (The Modern student's library, American division) © on introd.; 27Jan22, A654456. R57007, 9Jan50, Ruth Sherman (W)\nResponce: Walt Whitman\nExample 2:\nMACHAON, by E. F. Benson. pub. abroad in Hutchinson's magazine, Jan. 1923; illus. by Blam,\nResponce:\nE. F. Benson\nExample 3:\nINTERNATIONAL CORRESPONDENCE SCHOOLS. Commercial signs. Instruction paper. Serial 2086. 1st ed.\nResponce:\nNone\nExample 4:\nRAND MCNALLY AND COMPANY. Rand McNally indexed pocket map; tourists' and shippers' guide. © Rand McNally & Co. (PCB) Manitoba. © 9Jun23, A710667. R75878, 19Mar51.\nResponce:\n"
    inputs = []
    for x in data["full_text"]:
        inputs.append(prompt + x)
    inputs = tokenizer(inputs, return_tensors="pt", padding=True).input_ids.to(0)
    outputs = model.generate(
        inputs,
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
    "RWKV/HF_v5-Eagle-7B", trust_remote_code=True, torch_dtype=torch.float16
).to(0)
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
