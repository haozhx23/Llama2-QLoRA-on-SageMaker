# Import libraries
import argparse
import bitsandbytes as bnb
from datasets import load_dataset
from functools import partial
import os
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, AutoPeftModelForCausalLM
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, Trainer, TrainingArguments, BitsAndBytesConfig, \
    DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import load_dataset

import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import transformers
from torch.utils.data import Dataset
from transformers import Trainer

############################
from datetime import datetime
import os

import json, socket
import argparse
import io




##################################################


# def _make_r_io_base(f, mode: str):
#     if not isinstance(f, io.IOBase):
#         f = open(f, mode=mode)
#     return f


# def _jload(f, mode="r"):
#     """Load a .json file into a dictionary."""
#     f = _make_r_io_base(f, mode)
#     jdict = json.load(f)
#     f.close()
#     return jdict


def create_prompt_formats(sample):
    """
    Format various fields of the sample ('instruction', 'context', 'response')
    Then concatenate them using two newline characters
    :param sample: Sample dictionnary
    """
    INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    INSTRUCTION_KEY = "### Instruction:"
    INPUT_KEY = "Input:"
    RESPONSE_KEY = "### Response:"
    END_KEY = "### End"

    blurb = f"{INTRO_BLURB}"
    instruction = f"{INSTRUCTION_KEY}\n{sample['instruction']}"
    input_context = f"{INPUT_KEY}\n{sample['input']}" if sample["input"] else None
    response = f"{RESPONSE_KEY}\n{sample['output']}"
    end = f"{END_KEY}"

    parts = [part for part in [blurb, instruction, input_context, response, end] if part]

    formatted_prompt = "\n\n".join(parts)

    sample["text"] = formatted_prompt

    return sample


# SOURCE https://github.com/databrickslabs/dolly/blob/master/training/trainer.py
def get_max_length(model):
    conf = model.config
    max_length = None
    for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
        max_length = getattr(model.config, length_setting, None)
        if max_length:
            print(f"Found max lenth: {max_length}")
            break
    if not max_length:
        max_length = 1024
        print(f"Using default max length: {max_length}")
    return max_length


def preprocess_batch(batch, tokenizer, max_length):
    """
    Tokenizing a batch
    """
    return tokenizer(
        batch["text"],
        max_length=max_length,
        truncation=True,
    )


# SOURCE https://github.com/databrickslabs/dolly/blob/master/training/trainer.py
def preprocess_dataset(tokenizer: AutoTokenizer, max_length: int, seed, dataset: str):
    """Format & tokenize it so it is ready for training
    :param tokenizer (AutoTokenizer): Model Tokenizer
    :param max_length (int): Maximum number of tokens to emit from tokenizer
    """

    # Add prompt to each sample
    print("Preprocessing dataset...")
    dataset = dataset.map(create_prompt_formats)  # , batched=True)

    # Apply preprocessing to each batch of the dataset & and remove 'instruction', 'context', 'response', 'category' fields
    _preprocessing_function = partial(preprocess_batch, max_length=max_length, tokenizer=tokenizer)
    dataset = dataset.map(
        _preprocessing_function,
        batched=True,
        remove_columns=["instruction", "input", "output", "text"],
    )

    # Filter out samples that have input_ids exceeding max_length
    dataset = dataset.filter(lambda sample: len(sample["input_ids"]) < max_length)

    # Shuffle dataset
    dataset = dataset.shuffle(seed=seed)

    return dataset


def create_peft_config(modules):
    """
    Create Parameter-Efficient Fine-Tuning config for your model
    :param modules: Names of the modules to apply Lora to
    """
    config = LoraConfig(
        r=16,  # dimension of the updated matrices
        lora_alpha=64,  # parameter for scaling
        target_modules=modules,
        lora_dropout=0.1,  # dropout probability for layers
        bias="none",
        task_type="CAUSAL_LM",
    )

    return config


# SOURCE https://github.com/artidoro/qlora/blob/main/qlora.py
def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit  # if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def print_trainable_parameters(model, use_4bit=False):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    if use_4bit:
        trainable_params /= 2
    print(
        f"all params: {all_param:,d} || trainable params: {trainable_params:,d} || trainable%: {100 * trainable_params / all_param}"
    )


def train(model, tokenizer, dataset, output_dir):
    # Apply preprocessing to the model to prepare it by
    # 1 - Enabling gradient checkpointing to reduce memory usage during fine-tuning
    model.gradient_checkpointing_enable()

    # 2 - Using the prepare_model_for_kbit_training method from PEFT
    model = prepare_model_for_kbit_training(model)

    # Get lora module names
    modules = find_all_linear_names(model)

    # Create PEFT config for these modules and wrap the model to PEFT
    peft_config = create_peft_config(modules)
    model = get_peft_model(model, peft_config)

    # Print information about the percentage of trainable parameters
    print_trainable_parameters(model)

    # Training parameters
    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        args=TrainingArguments(
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=4,
            warmup_steps=2,
            max_steps=args.max_steps,
            learning_rate=2e-4,
            # fp16=True,
            bf16=True,
            logging_steps=1,
            output_dir=output_dir,
            optim="paged_adamw_8bit",
        ),
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
        # data_collator=data_collator
    )

    model.config.use_cache = False  # re-enable for inference to speed up predictions for similar inputs

    ### SOURCE https://github.com/artidoro/qlora/blob/main/qlora.py
    # Verifying the datatypes before training

    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes: dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items(): total += v
    for k, v in dtypes.items():
        print(k, v, v / total)

    do_train = True

    # Launch training
    print("Training...")

    if do_train:
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        print(metrics)

        ###

    # Saving model
    print("Saving last checkpoint of the model...")
    os.makedirs(output_dir, exist_ok=True)
    trainer.model.save_pretrained(output_dir)

    # Free memory for merging weights
    del model
    del trainer
    torch.cuda.empty_cache()


def parse_arge():
    """Parse the arguments."""
    parser = argparse.ArgumentParser()

    # add model id and dataset path argument
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="/tmp/fm/",
        help="Model id to use for training.",
    )
    parser.add_argument(
        "--data_dir", type=str, default="/opt/ml/input/data/train123/", help="Path to dataset."
    )
    parser.add_argument(
        "--output_dir", type=str, default="/tmp/fm_output/", help="Path to dataset."
    )
    # parser.add_argument(
    #     "--hf_token", type=str, default=HfFolder.get_token(), help="Path to dataset."
    # )
    # add training hyperparameters for epochs, batch size, learning rate, and seed
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of epochs to train for."
    )
    parser.add_argument(
        "--max_steps", type=int, default=20, help="Number of steps to train for."
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Batch size to use for training.",
    )
    parser.add_argument(
        "--lr", type=float, default=5e-5, help="Learning rate to use for training."
    )
    parser.add_argument(
        "--seed", type=int, default=1, help="Seed to use for training."
    )
    parser.add_argument(
        "--gradient_checkpointing",
        type=bool,
        default=True,
        help="Path to deepspeed config file.",
    )
    parser.add_argument(
        "--bf16",
        type=bool,
        default=True if torch.cuda.get_device_capability()[0] == 8 else False,
        help="Whether to use bf16.",
    )
    parser.add_argument(
        "--merge_weights",
        type=bool,
        default=True,
        help="Whether to merge LoRA weights with base model.",
    )
    args, _ = parser.parse_known_args()

    # if args.hf_token:
    #     print(f"Logging into the Hugging Face Hub with token {args.hf_token[:10]}...")
    #     login(token=args.hf_token)

    return args


if __name__ == "__main__":
    os.system("chmod +x ./s5cmd")

    args = parse_arge()
    
    # Reproducibility
    set_seed(args.seed)

    # copy model from s3 to training docker
    os.system("./s5cmd sync {0} {1}".format(os.environ['MODEL_S3_PATH'], args.model_name_or_path))
    print(f'------Model finished cp from s3 to docker-------')

    # args = {
    #     'model_name_or_path': 'llama2/models--TheBloke--Llama-2-7B-fp16/snapshots/ec92360670debf11267ccece99b8e65a8c723802/',
    #     'data_dir': 'alpaca_data.json'
    # }

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto",  # dispatch efficiently the model on the available ressources
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # Needed for LLaMA tokenizer
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset('json', data_files=args.data_dir, split='train')
    max_length = get_max_length(model)
    dataset = preprocess_dataset(tokenizer, max_length, args.seed, dataset)  # seed

    output_dir = args.output_dir + "final_checkpoint"
    train(model, tokenizer, dataset, output_dir)
    tokenizer.save_pretrained(output_dir)

    model = AutoPeftModelForCausalLM.from_pretrained(output_dir, device_map="auto", torch_dtype=torch.bfloat16)
    model = model.merge_and_unload()

    output_merged_dir = args.output_dir + "final_merged_checkpoint"
    os.makedirs(output_merged_dir, exist_ok=True)
    model.save_pretrained(output_merged_dir)#, safe_serialization=True)
    print(f'------merge dir saved-------')

    with open(output_merged_dir+'/config.json','r') as f:
        rconf = json.load(f)

    rconf.pop('_name_or_path')

    with open(output_merged_dir+'/config.json','w') as f:
        json.dump(rconf,f)
        
    print(f'------tmp path removed-------')

    # save tokenizer for easy inference
    # tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.save_pretrained(output_merged_dir)

    persistant_path = os.environ['OUTPUT_MODEL_S3_PATH'] # + str(datetime.now().strftime("%Y-%m-%d-%H-%M-%S")) + '/'
    # os.system("./s5cmd sync {0} {1}".format(args.output_dir, persistant_path))
    os.system("./s5cmd sync {0} {1}".format(output_merged_dir+'/', persistant_path))
    print(f'------Model output finished cp from docker to s3-------')
    # print(f'------Adapter Only Model Path: {persistant_path}final_checkpoint/-------')
    print(f'######Merged PEFT Model Path: {persistant_path}######')

