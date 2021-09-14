# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import time

import torch
from torch import Tensor
from pytorch_lightning import Callback
from pytorch_lightning.utilities import rank_zero_info
from pl_bolts.callbacks import SparseMLCallback
from typing import List, Union, Dict, Any


class LightningBoltsSparseMLCallback(SparseMLCallback, Callback):
    def __init__(self, output_dir, recipe_path):
        self.output_dir = output_dir
        super().__init__(recipe_path=recipe_path)
        self.sample_batch = None
    
    def training_epoch_end(self, training_step_outputs: List[Union[Tensor, Dict[str, Any]]]) -> None:
        # get sample batch data at the end of the training epoch
        if isinstance(training_step_outputs, list) and len(training_step_outputs) > 0:
            self.sample_batch = training_step_outputs[0]
        else:
            raise ValueError("Training batch output is empty.",
                             "Please check data to make sure there is no null instances")

    def on_save_checkpoint(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", callback_state: Dict[str, Any]
    ) -> None:
        self.export_to_sparse_onnx(output_dir=self.output_dir, sample_batch=self.sample_batch)


class CUDACallback(Callback):

    def on_train_epoch_start(self, trainer, pl_module):
        # Reset the memory use counter
        torch.cuda.reset_peak_memory_stats(trainer.root_gpu)
        torch.cuda.synchronize(trainer.root_gpu)
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        torch.cuda.synchronize(trainer.root_gpu)
        max_memory = torch.cuda.max_memory_allocated(trainer.root_gpu) / 2**20
        epoch_time = time.time() - self.start_time

        max_memory = trainer.training_type_plugin.reduce(max_memory)
        epoch_time = trainer.training_type_plugin.reduce(epoch_time)

        rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
        rank_zero_info(f"Average Peak memory {max_memory:.2f}MiB")
