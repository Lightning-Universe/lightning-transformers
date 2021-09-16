from sparseml.pytorch.utils.logger import LambdaLogger
from typing import Optional, Dict

try:
    import wandb
    wandb_err = None
except Exception as err:
    wandb = None
    wandb_err = err


class WANDBLogger(LambdaLogger):
    """
    Modifier logger that handles outputting values to Weights and Biases.

    :param init_kwargs: the args to call into wandb.init with;
        ex: wandb.init(**init_kwargs). If not supplied, then init will not be called
    :param name: name given to the logger, used for identification;
        defaults to wandb
    :param enabled: True to log, False otherwise
    """

    @staticmethod
    def available() -> bool:
        """
        :return: True if wandb is available and installed, False, otherwise
        """
        return not wandb_err

    def __init__(
        self,
        init_kwargs: Optional[Dict] = None,
        name: str = "wandb",
        enabled: bool = True,
    ):
        super().__init__(lambda_func=self._log_lambda, name=name, enabled=enabled)

        if wandb_err:
            raise wandb_err

        if init_kwargs:
            wandb.init(**init_kwargs)