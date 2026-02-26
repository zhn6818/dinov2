# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import os
from typing import Optional, Callable

from .extended import ExtendedVisionDataset


class FlatFolderDataset(ExtendedVisionDataset):
    """
    支持简单文件夹结构的数据集，无需类别子文件夹。
    适用于自监督学习场景。

    目录结构示例:
        root/
            ├── image1.jpg
            ├── image2.png
            └── ...
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        extensions: tuple = ('.jpg', '.jpeg', '.png', '.JPEG', '.PNG', '.bmp', '.BMP', '.webp', '.WEBP'),
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.root = root
        self.extensions = extensions

        # 收集所有图像文件
        self.samples = self._load_images()
        if len(self.samples) == 0:
            raise RuntimeError(f"在目录 {root} 中未找到任何图像文件 (支持的扩展名: {extensions})")

    def _load_images(self) -> list:
        """扫描文件夹中的所有图像"""
        samples = []
        root = os.path.expanduser(self.root)

        if not os.path.exists(root):
            raise FileNotFoundError(f"目录不存在: {root}")

        if not os.path.isdir(root):
            raise NotADirectoryError(f"路径不是目录: {root}")

        for fname in sorted(os.listdir(root)):
            if fname.lower().endswith(self.extensions):
                samples.append(os.path.join(root, fname))

        return samples

    def get_image_data(self, index: int) -> bytes:
        """返回图像字节数据"""
        path = self.samples[index]
        with open(path, 'rb') as f:
            return f.read()

    def get_target(self, index: int):
        """自监督学习不需要标签"""
        return None

    def __len__(self) -> int:
        return len(self.samples)

    def __repr__(self) -> str:
        return f"FlatFolderDataset(root={self.root}, num_samples={len(self.samples)})"
