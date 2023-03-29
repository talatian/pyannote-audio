# MIT License
#
# Copyright (c) 2020- CNRS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import Dict, List, Optional, Sequence, Text, Tuple, Union

import numpy as np
import torch
from pyannote.core import Segment, SlidingWindow, SlidingWindowFeature
from pyannote.database import Protocol
from torch_audiomentations.core.transforms_interface import BaseWaveformTransform
from torchmetrics import Metric

from pyannote.audio.core.task import Problem, Resolution, Specifications, Task
from pyannote.audio.tasks.segmentation.mixins import SegmentationTaskMixin


class MultiLabelSegmentation(SegmentationTaskMixin, Task):
    """Generic multi-label segmentation

    Multi-label segmentation is the process of detecting temporal intervals
    when a specific audio class is active.

    Example use cases include speaker tracking, gender (male/female)
    classification, or audio event detection.

    Parameters
    ----------
    protocol : Protocol
        pyannote.database protocol
    classes : List[str], optional
        List of classes. Defaults to the list of classes available in the training set.
    duration : float, optional
        Chunks duration. Defaults to 2s.
    warm_up : float or (float, float), optional
        Use that many seconds on the left- and rightmost parts of each chunk
        to warm up the model. While the model does process those left- and right-most
        parts, only the remaining central part of each chunk is used for computing the
        loss during training, and for aggregating scores during inference.
        Defaults to 0. (i.e. no warm-up).
    balance: str, optional
        When provided, training samples are sampled uniformly with respect to that key.
        For instance, setting `balance` to "uri" will make sure that each file will be
        equally represented in the training samples.
    weight: str, optional
        When provided, use this key to as frame-wise weight in loss function.
    batch_size : int, optional
        Number of training samples per batch. Defaults to 32.
    num_workers : int, optional
        Number of workers used for generating training samples.
        Defaults to multiprocessing.cpu_count() // 2.
    pin_memory : bool, optional
        If True, data loaders will copy tensors into CUDA pinned
        memory before returning them. See pytorch documentation
        for more details. Defaults to False.
    augmentation : BaseWaveformTransform, optional
        torch_audiomentations waveform transform, used by dataloader
        during training.
    metric : optional
        Validation metric(s). Can be anything supported by torchmetrics.MetricCollection.
        Defaults to AUROC (area under the ROC curve).
    """

    def __init__(
        self,
        protocol: Protocol,
        classes: Optional[List[str]] = None,
        duration: float = 2.0,
        warm_up: Union[float, Tuple[float, float]] = 0.0,
        balance: Text = None,
        weight: Text = None,
        batch_size: int = 32,
        num_workers: int = None,
        pin_memory: bool = False,
        augmentation: BaseWaveformTransform = None,
        metric: Union[Metric, Sequence[Metric], Dict[str, Metric]] = None,
    ):
        super().__init__(
            protocol,
            duration=duration,
            warm_up=warm_up,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            augmentation=augmentation,
            metric=metric,
        )

        self.balance = balance
        self.weight = weight
        self.classes = classes

        # task specification depends on the data: we do not know in advance which
        # classes should be detected. therefore, we postpone the definition of
        # specifications to setup()

    def setup(self, stage: Optional[str] = None):

        super().setup(stage=stage)

        self.specifications = Specifications(
            classes=self.classes,
            problem=Problem.MULTI_LABEL_CLASSIFICATION,
            resolution=Resolution.FRAME,
            duration=self.duration,
            warm_up=self.warm_up,
        )

    def prepare_chunk(self, file_id: int, start_time: float, duration: float):
        """Prepare chunk for multi-label segmentation

        Parameters
        ----------
        file_id : int
            File index
        start_time : float
            Chunk start time
        duration : float
            Chunk duration.

        Returns
        -------
        sample : dict
            Dictionary containing the chunk data with the following keys:
            - `X`: waveform
            - `y`: target
            - `meta`:
                - `database`: database index
                - `file`: file index

        Notes
        -----
        y is a trinary matrix with shape (num_classes, num_frames):
            -  0: class is inactive
            -  1: class is active
            - -1: we have no idea

        """

        # TODO: handle un-annotated classes self.annotated_classes
        label_idx = 0  # for linting to pass...

        file = self.get_file(file_id)

        chunk = Segment(start_time, start_time + duration)

        sample = dict()
        sample["X"], _ = self.model.audio.crop(file, chunk, duration=duration)

        # use model introspection to predict how many frames it will output
        # TODO: this should be cached
        num_samples = sample["X"].shape[1]
        num_frames, _ = self.model.introspection(num_samples)
        resolution = duration / num_frames
        frames = SlidingWindow(start=0.0, duration=resolution, step=resolution)

        # gather all annotations of current file
        annotations = self.annotations[self.annotations["file_id"] == file_id]

        # gather all annotations with non-empty intersection with current chunk
        chunk_annotations = annotations[
            (annotations["start"] < chunk.end) & (annotations["end"] > chunk.start)
        ]

        # discretize chunk annotations at model output resolution
        start = np.maximum(chunk_annotations["start"], chunk.start) - chunk.start
        start_idx = np.floor(start / resolution).astype(np.int)
        end = np.minimum(chunk_annotations["end"], chunk.end) - chunk.start
        end_idx = np.ceil(end / resolution).astype(np.int)

        # get list and number of labels for current scope
        labels = np.unique(chunk_annotations[label_idx])
        num_labels = len(labels)

        # initial frame-level targets
        y = np.zeros((num_frames, num_labels), dtype=np.uint8)

        # map labels to indices
        mapping = {label: idx for idx, label in enumerate(labels)}

        for c, chunk_annotation in enumerate(chunk_annotations):
            start, end = start_idx[c], end_idx[c]
            label = mapping[chunk_annotation[label_idx]]
            y[start:end, label] = 1

        sample["y"] = SlidingWindowFeature(y, frames, labels=labels)

        metadata = self.metadata[file_id]
        sample["meta"] = {key: metadata[key] for key in metadata.dtype.names}
        sample["meta"]["file"] = file_id

        return sample

    def collate_y(self, batch) -> torch.Tensor:

        labels = self.specifications.classes

        batch_size, num_frames, num_labels = (
            len(batch),
            len(batch[0]["y"]),
            len(labels),
        )
        Y = np.zeros((batch_size, num_frames, num_labels), dtype=np.int64)

        # TODO: mark unannotated classes as -1 instead of 0

        for i, b in enumerate(batch):
            for local_idx, label in enumerate(b["y"].labels):
                global_idx = labels.index(label)
                Y[i, :, global_idx] = b["y"].data[:, local_idx]

        return torch.from_numpy(Y)

    # TODO: add option to give more weights to smaller classes
    # TODO: add option to balance training samples between classes
