from loguru import logger
import os
from component.template import template_dict
from transformers import (
    AutoTokenizer,
    AutoModel,AutoModelForCausalLM
)
from trl import DPOConfig, DPOTrainer
from dataset_dpo import UnifiedDPODataset

model_name = "EleutherAI/pythia-70m"

tokenizer = AutoTokenizer.from_pretrained(model_name,clean_up_tokenization_spaces=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


model = AutoModelForCausalLM.from_pretrained(model_name)
ref_model = AutoModelForCausalLM.from_pretrained(model_name)

train_dataset = UnifiedDPODataset("data/dummy_dpo.jsonl", tokenizer, 4096, 512, template_dict['gemma'])
data_collator = None

training_args = DPOConfig(
    output_dir="output",
    do_train=True,
    do_eval=False,
    report_to="none",
    use_cpu=True,
    per_device_train_batch_size=4,
)
trainer = DPOTrainer(
    model = model,
    ref_model = ref_model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

logger.info("*** starting training ***")
train_result = trainer.train()

# 保存最好的checkpoint
final_save_path = os.path.join(training_args.output_dir)
trainer.save_model(final_save_path)

# 保存训练指标
metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()