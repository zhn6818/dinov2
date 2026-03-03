import logging
import os
from typing import Any, Callable, List, Optional, Tuple

import numpy as np

from .extended import ExtendedVisionDataset


logger = logging.getLogger("dinov2")


class JinXiang(ExtendedVisionDataset):
    """
    简单的无标签金相图像数据集。

    目录假设:
        root/
            class_or_group_1/
                img1.jpg
                img2.png
                ...
            class_or_group_2/
                ...

    所有子目录中的图片都会被视为一个无标签集合，用于自监督预训练。
    """

    def __init__(
        self,
        *,
        root: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        **kwargs: Any,
    ) -> None:
        # ExtendedVisionDataset 负责解码与 transforms
        super().__init__(
            root,
            transforms,
            transform,
            target_transform,
            **kwargs,
        )
        self.root = root

        self._image_relpaths: List[str] = []
        self._labels: List[int] = []

        self._collect_images()

    def _is_image_file(self, filename: str) -> bool:
        filename_lower = filename.lower()
        return filename_lower.endswith(
            (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
        )

    def _collect_images(self) -> None:
        if not os.path.isdir(self.root):
            raise RuntimeError(f'JinXiang dataset root "{self.root}" is not a directory')

        for dirpath, _, filenames in os.walk(self.root):
            for fname in filenames:
                if not self._is_image_file(fname):
                    continue
                full_path = os.path.join(dirpath, fname)
                relpath = os.path.relpath(full_path, self.root)
                self._image_relpaths.append(relpath)
                # 自监督训练不需要真正的标签，这里使用占位符 0
                self._labels.append(0)

        if not self._image_relpaths:
            raise RuntimeError(f'No image files found under "{self.root}" for JinXiang dataset')

        logger.info(f"JinXiang dataset loaded from {self.root}, #images={len(self._image_relpaths):,d}")

    def get_image_relpath(self, index: int) -> str:
        return self._image_relpaths[index]

    def get_image_data(self, index: int) -> bytes:
        image_relpath = self.get_image_relpath(index)
        image_full_path = os.path.join(self.root, image_relpath)
        with open(image_full_path, mode="rb") as f:
            image_data = f.read()
        return image_data

    def get_target(self, index: int) -> Any:
        # 自监督任务下，target 仅作为占位
        return self._labels[index]

    def get_targets(self) -> np.ndarray:
        return np.array(self._labels)

    def __len__(self) -> int:
        return len(self._image_relpaths)

