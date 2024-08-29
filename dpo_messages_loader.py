
import json
from loguru import logger
from torch.utils.data import Dataset
from typing import List, Dict
from component.template import template_dict
from transformers import PreTrainedTokenizer,AutoTokenizer
from glob import glob
from common.common_utils import read_json

def auto_detect_template(model_name):
    """
    support:
        llama3,gemma,yi,internlm2,mistral
    model_name in HF:
        meta-llama/Meta-Llama-3.1-8B-Instruct; google/gemma-2-2b-it; 01-ai/Yi-1.5-34B-Chat
        internlm/internlm2_5-7b-chat; mistralai/Mistral-7B-Instruct-v0.3
    """
    if "meta-llama/Meta-Llama-3" in model_name:
        name = "llama3"
    elif "google/gemma" in model_name:
        name = "gemma"
    elif "01-ai/Yi" in model_name:
        name = "yi"
    elif "internlm/internlm2" in model_name:
        name = "internlm2"
    elif "mistralai/Mistral" in model_name:
        name = "mistral"
    else:
        name = "default"
    logger.info(f"Auto detect template: {name}")
    return template_dict[name]
    

class DPODatasetMessages(Dataset):
    def __init__(self, data:List[Dict], tokenizer:PreTrainedTokenizer, max_seq_length:int=4096, max_prompt_length:int=4096):
        """
        Required format of data:
            [{
                "id": "01",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an AI assistant."
                    },
                    {
                        "role": "user",
                        "content": "Hello, how are you?"
                    }
                ],
                "chosen": "I am fine, thank you.",
                "rejected": "What are you doing?"
            }, ...]
        """
        self.data = data
        self.tokenizer = tokenizer

        template = auto_detect_template(tokenizer.name_or_path)
        self.template_name = template.template_name
        self.system_format = template.system_format
        self.user_format = template.user_format
        self.assistant_format = template.assistant_format
        self.system = template.system

        self.max_seq_length = max_seq_length
        self.max_prompt_length = max_prompt_length

        logger.info(f"Dataset size: {len(data)}")

    @staticmethod
    def biuld_from_dir(dir_path, model_name, max_seq_length=4096, max_prompt_length=4096):
        """
        Load dataset from a directory.
        """
        paths = glob(f"{dir_path}/*.json")
        assert len(paths) > 0, f"No json file found in {dir_path}"
        logger.info(f"Loading dataset from {dir_path}.")
        logger.info(f"Find {len(paths)} files.")
        data = [read_json(path) for path in paths]

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        return DPODatasetMessages(data, tokenizer, max_seq_length, max_prompt_length)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        
        """
        d = self.data[index]

        
        chosen = d['chosen'] # dict{"role": "assistant|user", "content": "xxx"}
        rejected = d['rejected']
        assert len(chosen) == len(rejected)

        # 冗余？？？
        assert chosen[0]["content"] == rejected[0]["content"]

        # 判断第0个是否为system
        if chosen[0]['role'] == 'system':
            system = chosen[0]['content'].strip()
            history = chosen[1:-1]  # 对话上文
            chosen, rejected = chosen[-1], rejected[-1]
        else:
            system = None
            history = chosen[:-1]  # 对话上文
            chosen, rejected = chosen[-1], rejected[-1]

        # build prompt
        prompt_input_ids = self.build_prompt_input_ids(system, history)

        # build response
        chosen = self.assistant_format.format(content=chosen['content'], stop_token=self.tokenizer.eos_token)
        rejected = self.assistant_format.format(content=rejected['content'], stop_token=self.tokenizer.eos_token)

        chosen_input_ids = self.tokenizer.encode(chosen, add_special_tokens=False)
        rejected_input_ids = self.tokenizer.encode(rejected, add_special_tokens=False)

        # truncate by max_seq_length
        longer_response_length = max(len(chosen_input_ids), len(rejected_input_ids))
        # if combined sequence is too long, truncate the prompt
        if len(prompt_input_ids) + longer_response_length > self.max_seq_length:
            max_prompt_length = max(self.max_prompt_length, self.max_seq_length - longer_response_length)
            prompt_input_ids = prompt_input_ids[-max_prompt_length:]
        # if that's still too long, truncate the response
        if len(prompt_input_ids) + longer_response_length > self.max_seq_length:
            chosen_input_ids = chosen_input_ids[: self.max_seq_length - len(prompt_input_ids)]
            rejected_input_ids = rejected_input_ids[: self.max_seq_length - len(prompt_input_ids)]

        chosen_labels = [-100] * len(prompt_input_ids) + chosen_input_ids
        chosen_input_ids = prompt_input_ids + chosen_input_ids
        rejected_labels = [-100] * len(prompt_input_ids) + rejected_input_ids
        rejected_input_ids = prompt_input_ids + rejected_input_ids
        assert len(chosen_labels) == len(chosen_input_ids)
        assert len(rejected_labels) == len(rejected_input_ids)

        inputs = dict(
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=[1]*len(prompt_input_ids),
            chosen_input_ids=chosen_input_ids,
            chosen_attention_mask=[1]*len(chosen_input_ids),
            chosen_labels=chosen_labels,
            rejected_input_ids=rejected_input_ids,
            rejected_attention_mask=[1]*len(rejected_input_ids),
            rejected_labels=rejected_labels,
        )
        return inputs

    # for adapting to DPOTrainer
    def map(self, func, **kwargs):
        return self
    
if __name__ == '__main__':
    # python dpo_messages_loader.py

    train_dataset = DPODatasetMessages.biuld_from_dir("test_json", "meta-llama/Meta-Llama-3.1-8B-Instruct")

    # print batch data with data loader
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    for batch in train_loader:
        print(batch)
        break
