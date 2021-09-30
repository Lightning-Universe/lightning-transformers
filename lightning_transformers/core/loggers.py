# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import time
from typing import Dict, Optional, Union

from pytorch_lightning.loggers import WandbLogger


class WABLogger(WandbLogger):
    """Modifier logger that handles outputting values to Weights and Biases.

    :param init_kwargs: the args to call into wandb.init with;
        ex: wandb.init(**init_kwargs). If not supplied, then init will not be called
    :param name: name given to the logger, used for identification;
        defaults to wandb
    :param enabled: True to log, False otherwise
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.enabled = True

    def _lambda_func(
        self,
        tag: Optional[str],
        value: Optional[float],
        values: Optional[Dict[str, float]],
        step: Optional[int],
        wall_time: Optional[float],
    ) -> bool:
        params = {}

        if value is not None:
            params[tag] = value

        if values:
            if tag:
                values = {f"{tag}/{key}": val for key, val in values.items()}
            params.update(values)

        try:
            self.log_metrics(params, step=step)
        except Exception as e:
            print(params, e)

        return True

    def log_scalar(
        self,
        tag: str,
        value: float,
        step: Union[None, int] = None,
        wall_time: Union[None, float] = None,
    ):
        """
        :param tag: identifying tag to log the value with
        :param value: value to save
        :param step: global step for when the value was taken
        :param wall_time: global wall time for when the value was taken,
            defaults to time.time()
        :return: True if logged, False otherwise.
        """
        if not self.enabled:
            return False

        if not wall_time:
            wall_time = time.time()

        return self._lambda_func(tag, value, None, step, wall_time)

    def log_scalars(
        self,
        tag: str,
        values: Dict[str, float],
        step: Union[None, int] = None,
        wall_time: Union[None, float] = None,
    ):
        """
        :param tag: identifying tag to log the values with
        :param values: values to save
        :param step: global step for when the values were taken
        :param wall_time: global wall time for when the values were taken,
            defaults to time.time()
        :return: True if logged, False otherwise.
        """
        if not self.enabled:
            return False

        if not wall_time:
            wall_time = time.time()

        return self._lambda_func(tag, None, values, step, wall_time)
