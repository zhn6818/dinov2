#!/usr/bin/env python3
"""
从 DINOv2 训练 checkpoint 中提取 teacher 权重，保存为 eval 脚本可用的格式。
训练 checkpoint 格式: {"model": SSLMetaArch_state_dict, "optimizer": ..., "iteration": ...}
eval 所需格式: {"teacher": teacher_state_dict}

用法:
    python scripts/extract_teacher_from_checkpoint.py \
        --input output/vitb14_finetune2/model_0000749.rank_0.pth \
        --output output/vitb14_finetune2/teacher_checkpoint_0749.pth
"""
import argparse
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="训练 checkpoint 路径")
    parser.add_argument("--output", type=str, required=True, help="输出的 teacher checkpoint 路径")
    args = parser.parse_args()

    ckpt = torch.load(args.input, map_location="cpu")
    model_sd = ckpt["model"]

    # 提取 teacher 部分 (teacher.backbone.xxx -> backbone.xxx, teacher.dino_head.xxx -> dino_head.xxx)
    teacher_sd = {k.replace("teacher.", ""): v for k, v in model_sd.items() if k.startswith("teacher.")}
    if not teacher_sd:
        raise ValueError("Checkpoint 中未找到 teacher 权重")

    out_ckpt = {"teacher": teacher_sd}
    torch.save(out_ckpt, args.output)
    print(f"已提取 {len(teacher_sd)} 个 teacher 参数，保存到 {args.output}")


if __name__ == "__main__":
    main()
