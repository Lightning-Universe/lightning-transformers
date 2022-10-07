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
import operator

from lightning_utilities.core.imports import compare_version, module_available

_BOLTS_AVAILABLE = module_available("pl_bolts") and compare_version("pl_bolts", operator.ge, "0.4.0")
_BOLTS_GREATER_EQUAL_0_5_0 = module_available("pl_bolts") and compare_version("pl_bolts", operator.ge, "0.5.0")
_WANDB_AVAILABLE = module_available("wandb")
_ACCELERATE_AVAILABLE = module_available("accelerate")
