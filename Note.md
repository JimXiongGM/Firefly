# note

全量参数指令微调，将{num_gpus}替换为显卡数量
```bash
deepspeed --num_gpus=2 train.py --train_args_file train_args/sft/full/bloom-1b1-sft-full.json
```

多卡QLoRA指令微调：
```bash
torchrun --nproc_per_node={num_gpus} train.py --train_args_file train_args/sft/qlora/yi-6b-sft-qlora.json
```


DPO

```bash
deepspeed train_my.py --train_args_file train_args/dpo/full/llama3.1-8B-dpo-full.json
deepspeed --include localhost:3,5 train.py --train_args_file train_args/dpo/full/llama3.1-8B-dpo-full.json
```

dummy_dpo_messages.jsonl

```json
{
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
}
```


demo code
```json
{
	"prompt": "how can i develop a habit of drawing daily",
	"prompt_id": "086b3e24f29b8956a01059f79c56db35d118a06fb6b844b095737d042795cd43",
	"chosen": [{
		"content": "how can i develop a habit of drawing daily",
		"role": "user"
	}, {
		"content": "xxxxx.",
		"role": "assistant"
	}],
	"rejected": [{
		"content": "how can i develop a habit of drawing daily",
		"role": "user"
	}, {
		"content": "yyyyy.",
		"role": "assistant"
	}],
	"messages": [{
		"content": "how can i develop a habit of drawing daily",
		"role": "user"
	}, {
		"content": "zzzzz.",
		"role": "assistant"
	}],
	"score_chosen": 8.5,
	"score_rejected": 8.5
}
```