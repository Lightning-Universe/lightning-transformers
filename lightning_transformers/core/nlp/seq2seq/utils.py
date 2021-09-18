# Copyright 2020 The PyTorch Lightning team and The HuggingFace Team. All rights reserved.

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

import torch


def _pad_tensors_to_max_len(model_cfg, tensor, max_length):
    pad_token_id = model_cfg.pad_token_id if model_cfg.pad_token_id is not None else model_cfg.eos_token_id
    if pad_token_id is None:
        raise ValueError(
            f"Make sure that either `config.pad_token_id` or `config.eos_token_id` "
            f"is defined if tensor has to be padded to `max_length`={max_length}"
        )

    padded_tensor = pad_token_id * torch.ones((tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device)
    padded_tensor[:, : tensor.shape[-1]] = tensor
    return padded_tensor
