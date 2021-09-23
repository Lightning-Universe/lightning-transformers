import os
import shutil
import unittest
from importlib.util import find_spec

from lightning_transformers.core.loggers import WABLogger

if importlib.util.find_spec("wandb") != None:
    wandb_does_not_exist = False
else:
    wandb_does_not_exist = True

@unittest.skipIf(wandb_does_not_exist, "wandb does not exist")
class TestModifierLogger(unittest.TestCase):
    def setUp(self):
        # run logger offline
        self.logger = WABLogger(offline=True)

    def tearDown(self):
        # delete wandb folder
        cwd = os.getcwd()
        if os.path.exists(os.path.join(cwd, "wandb")):
            shutil.rmtree(os.path.join(cwd, "wandb"))

    def test_enabled(self):
        self.assertEqual(self.logger.enabled, True)

    def test_log_hyperparams(self):
        model_parameters = {"param1": 0.0, "param2": 1.0}
        self.logger.log_hyperparams(model_parameters)
        assert self.logger.experiment.config["param1"] == model_parameters["param1"]
        assert self.logger.experiment.config["param2"] == model_parameters["param2"]

    def test_log_scalar(self):
        self.logger.log_scalar("test-scalar-tag", 0.1)
        assert self.logger.experiment.summary["test-scalar-tag"] == 0.1

        self.logger.log_scalar("test-scalar-tag", 0.3, 5)
        assert self.logger.experiment.summary["test-scalar-tag"] == 0.3

        time = 1632368396.120609
        self.logger.log_scalar("test-scalar-tag", 0.9, 4, time - 1)
        assert self.logger.experiment.summary["test-scalar-tag"] == 0.9

    def test_log_scalars(self):
        self.logger.log_scalars("test-scalars-tag", {"scalar1": 0.2, "scalar2": 7.0})
        assert self.logger.experiment.summary["test-scalars-tag/scalar1"] == 0.2
        assert self.logger.experiment.summary["test-scalars-tag/scalar2"] == 7.0

        self.logger.log_scalars("test-scalars-tag", {"scalar1": 1.0, "scalar2": 5.6}, 1)
        assert self.logger.experiment.summary["test-scalars-tag/scalar1"] == 1.0
        assert self.logger.experiment.summary["test-scalars-tag/scalar2"] == 5.6

        time = 69082160104.686754
        self.logger.log_scalars("test-scalars-tag", {"scalar1": 6.0, "scalar2": 3.0}, 2, time - 1)
        assert self.logger.experiment.summary["test-scalars-tag/scalar1"] == 6.0
        assert self.logger.experiment.summary["test-scalars-tag/scalar2"] == 3.0
