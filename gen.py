from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import datasets
import argparse
from tqdm import tqdm
# import logging
# logging.disable(logging.WARNING)

def build_description_prompt(vccs_msg, diff):
    prompt_des = f"Give description of this code:\n{{vccs}}\n{{diff}}\Des:"
    return prompt_des.format(vccs=vccs_msg, diff=diff)

def split_batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def write_string_to_file(absolute_filename, string):
    with open(absolute_filename, 'a') as fout:
        fout.write(string)

def run(args):
    dataset_id = args.dataset_id
    model_id = args.model_id

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, )
    # print(tokenizer)
    if args.load_in_8bit:
        model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto', load_in_8bit=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto').cuda()
    
    model = PeftModel.from_pretrained(model, args.model_peft)
    print('Loaded peft model from ', args.model_peft)

    model.eval()
    
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    dataset = datasets.load_dataset(dataset_id, split=args.data_split)
    sources = [
        build_description_prompt(vccs_msg, diff)
        for vccs_msg, diff in zip(dataset['vccs_msg'], dataset['diff'])
    ]

    batch_list = split_batch(sources, args.batch_size)
    len_batch = len(sources) // args.batch_size
    with tqdm(total=len_batch, desc="gen") as pbar:
        for batch in batch_list:
            model_inputs = tokenizer(batch, return_tensors="pt", padding=True, max_length=args.max_length, truncation=True).to("cuda")
            generated_ids = model.generate(**model_inputs, max_new_tokens=args.max_new_tokens, pad_token_id=tokenizer.eos_token_id)

            truncated_ids = [ids[len(model_inputs[idx]):] for idx, ids in enumerate(generated_ids)]

            output = tokenizer.batch_decode(truncated_ids, skip_special_tokens=True)

            for idx, source in enumerate(batch):
                try:
                    out = output[idx].replace('\n', ' ')
                    write_string_to_file(args.output_file, out + '\n')
                except Exception as e:
                    print(e)
                    write_string_to_file(args.output_file, '\n')
            pbar.update(1)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=2, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--load_in_8bit", action='store_true',
                        help="Load model 8 bit.")
    parser.add_argument("--model_peft", type=str, default='')
    parser.add_argument("--model_id", default='codellama/CodeLlama-7b-hf', type=str,
                        help="Path to pre-trained model: e.g. codellama/CodeLlama-7b-hf")
    parser.add_argument("--dataset_id", default='zhaospei/cmg_allinone', type=str,
                        help="Path to dataset for generation")
    parser.add_argument("--output_file", type=str, default="gen.output")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--data_split", type=str, default='test')
    args = parser.parse_args()

    run(args)


if __name__ == '__main__':
    main()