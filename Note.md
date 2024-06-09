# note

全量参数指令微调，将{num_gpus}替换为显卡数量
```bash
deepspeed --num_gpus=2 train.py --train_args_file train_args/sft/full/bloom-1b1-sft-full.json
```

多卡QLoRA指令微调：
```bash
torchrun --nproc_per_node={num_gpus} train.py --train_args_file train_args/sft/qlora/yi-6b-sft-qlora.json
```

