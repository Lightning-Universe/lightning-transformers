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

import collections
import inspect
import os
from typing import Optional

import numpy
import pytorch_lightning as pl
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch import Tensor

from lightning_transformers.utilities.imports import _BOLTS_AVAILABLE

if _BOLTS_AVAILABLE:
    from pl_bolts.callbacks import SparseMLCallback

    class TransformerSparseMLCallback(SparseMLCallback):
        def __init__(self, output_dir, recipe_path):
            self.output_dir = output_dir
            super().__init__(recipe_path=recipe_path)

        def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
            optimizer = trainer.optimizers

            if len(optimizer) > 1:
                raise MisconfigurationException("SparseML only supports training with one optimizer.")
            optimizer = optimizer[0]

            loggers = trainer.logger

            if not isinstance(loggers, list):
                loggers = [loggers]

            self.manager.initialize(pl_module, epoch=0.0, logger=loggers)
            self.manager.initialize_loggers(loggers)

            optimizer = self.manager.modify(
                pl_module, optimizer, steps_per_epoch=self._num_training_steps_per_epoch(trainer), epoch=0
            )

            trainer.optimizers = [optimizer]

        @staticmethod
        def export_to_sparse_onnx(
            model: "pl.LightningModule", output_dir: str, sample_batch: Optional[Tensor] = None, **kwargs
        ) -> None:
            """Exports the model to ONNX format."""
            import onnxruntime
            from sparseml.pytorch.utils import ModuleExporter

            with model._prevent_trainer_and_dataloaders_deepcopy():
                exporter = ModuleExporter(model.model, output_dir=output_dir)
                sample_batch = sample_batch if sample_batch is not None else model.example_input_array
                if sample_batch is None:
                    raise MisconfigurationException(
                        "To export the model, a sample batch must be passed via "
                        "``SparseMLCallback.export_to_sparse_onnx(model, output_dir, sample_batch=sample_batch)`` "
                        "or an ``example_input_array`` property within the LightningModule"
                    )

                # the following is adapted from @natuan and @spacemanidol
                sample_inputs = os.path.join(output_dir, "sample-inputs")
                sample_outputs = os.path.join(output_dir, "sample-outputs")
                os.makedirs(sample_inputs, exist_ok=True)
                os.makedirs(sample_outputs, exist_ok=True)

                forward_args_spec = inspect.getfullargspec(exporter._module.__class__.forward)
                try:
                    # assume sample_batch is a callable dictionary-type object
                    one_sample_input = collections.OrderedDict(
                        [
                            (f, sample_batch[f][0].long().reshape(1, -1))
                            for f in forward_args_spec.args
                            if f in sample_batch
                        ]
                    )
                except RuntimeError:
                    # assume sample_batch is a tensor
                    one_sample_input = sample_batch

                try:
                    exporter.export_onnx(sample_batch=one_sample_input, convert_qat=True, **kwargs)
                    exporter.export_onnx(
                        sample_batch=one_sample_input,
                        name="small_model.onnx",
                        export_params=False,
                        **kwargs,
                    )
                    onnx_file = os.path.join(output_dir, "model.onnx")

                except RuntimeError:
                    raise RuntimeError("Error exporting ONNX models and/or inputs/outputs")

                sess = onnxruntime.InferenceSession(onnx_file)

                num_samples = 0
                # add additional files for testing since this feature is very new
                if isinstance(one_sample_input, (collections.OrderedDict, dict)):
                    input_names = list(sample_batch.keys())
                    output_names = [o.name for o in sess.get_outputs()]
                    for input_vals in zip(*sample_batch.values()):
                        input_feed = {k: v.long().numpy() for k, v in zip(input_names, input_vals)}
                        output_vals = sess.run(output_names, {k: input_feed[k].reshape(1, -1) for k in input_feed})
                        output_dict = {name: numpy.squeeze(val) for name, val in zip(output_names, output_vals)}
                        file_idx = f"{num_samples}".zfill(4)
                        numpy.savez(f"{sample_inputs}/inp-{file_idx}.npz", **input_feed)
                        numpy.savez(f"{sample_outputs}/out-{file_idx}.npz", **output_dict)
                        num_samples += 1
                else:
                    output_names = [o.name for o in sess.get_outputs()]
                    input_feed = {"input": sample_batch.numpy()}
                    output_vals = sess.run(output_names, input_feed)
                    output_dict = {name: numpy.squeeze(val) for name, val in zip(output_names, output_vals)}
                    file_idx = f"{num_samples}".zfill(4)
                    numpy.savez(f"{sample_inputs}/inp-{file_idx}.npz", **input_feed)
                    numpy.savez(f"{sample_outputs}/out-{file_idx}.npz", **output_dict)

        def teardown(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: Optional[str] = None) -> None:
            sample_batch = next(iter(trainer.train_dataloader))
            # if asked for output names, bert's ModelOutput gives two names
            # but when run, this the model only gives one output
            # workaround is just to force onnx to realize there is only one output
            output_names = ["logits"]
            self.export_to_sparse_onnx(
                output_dir=self.output_dir, model=pl_module, sample_batch=sample_batch, output_names=output_names
            )
