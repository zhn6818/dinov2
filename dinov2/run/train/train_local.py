# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
单机直接训练入口，绕过 submitit/SLURM。
适用于 1-N GPU 单机训练，训练在前台运行并实时输出日志。
"""

import os
import sys

# 将项目根目录加入 sys.path，无需安装 dinov2
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import logging

from dinov2.logging import setup_logging
from dinov2.train import get_args_parser as get_train_args_parser, main as train_main


logger = logging.getLogger("dinov2")


def main():
    description = "DINOv2 单机直接训练（不使用 submitit）"
    args_parser = get_train_args_parser(add_help=True)
    args_parser.description = description
    args = args_parser.parse_args()

    setup_logging()

    assert os.path.exists(args.config_file), f"配置文件不存在: {args.config_file}"
    assert args.output_dir, "请指定 --output-dir"

    logger.info("使用单机直接训练模式（非 submitit）")
    logger.info(f"输出目录: {os.path.abspath(args.output_dir)}")

    train_main(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
