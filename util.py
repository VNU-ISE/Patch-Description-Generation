import copy
import datasets

def get_preprocessed_data(dataset_id, tokenizer, split):
    dataset = datasets.load_dataset(dataset_id, split=split)

    prompt_type = f"Give type of this code:\n{{vccs}}\n{{diff}}\nType:"
    prompt_des = f"Give description of this code:\n{{vccs}}\n{{diff}}\Des:"
    def apply_prompt_template(sample):
        return {
            "type_input": prompt_type.format(vccs=sample['vccs_msg'], diff=sample['diff']),
            "des_input": prompt_des.format(vccs=sample['vccs_msg'], diff=sample['diff']),
            "type_label": f"type_{{type}}".format(type=sample['type']),
            "des_label": sample['msg']
        }

    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))

    def tokenize_function(sample):
        model_inputs = tokenizer.encode(tokenizer.bos_token + sample['des_input'], add_special_tokens=False, max_length=991, truncation=True)
        type_model_inputs = tokenizer.encode(tokenizer.bos_token + sample['type_input'], add_special_tokens=False, max_length=991, truncation=True)
        label_output_encodings = tokenizer.encode(sample['des_label'] + tokenizer.eos_token, add_special_tokens=False, max_length=32, truncation=True)
        type_output_encodings = tokenizer.encode(sample['type_label'] + tokenizer.eos_token, add_special_tokens=False, max_length=32, truncation=True)
        max_length = 1024 - len(model_inputs) - len(label_output_encodings)
        pad_des = tokenizer.encode(tokenizer.eos_token, add_special_tokens=False, max_length=max_length, padding='max_length', truncation=True)
        max_length = 1024 - len(type_model_inputs) - len(type_output_encodings)
        pad_type = tokenizer.encode(tokenizer.eos_token, add_special_tokens=False, max_length=max_length, padding='max_length', truncation=True)

        sample = {
            "input_ids": model_inputs + label_output_encodings + pad_des,
            "attention_mask" : [1] * (len(model_inputs) + len(label_output_encodings) + len(pad_des)),
            "labels": [-100] * len(model_inputs) + label_output_encodings + [-100] * len(pad_des),
            "type_input_ids": type_model_inputs + type_output_encodings + pad_type,
            "type_attention_mask": [1] * (len(type_model_inputs) + len(type_output_encodings) + len(pad_type)),
            "type_labels": [-100] * len(type_model_inputs) + type_output_encodings + [-100] * len(pad_type),
        }

        return sample


    tokenized_datasets = dataset.map(
        tokenize_function,
        remove_columns=list(dataset.features),
    )

    return tokenized_datasets