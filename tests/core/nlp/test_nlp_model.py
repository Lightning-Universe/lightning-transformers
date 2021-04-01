from unittest.mock import MagicMock, patch

from lightning_transformers.core.nlp import HFTransformer


def test_pipeline_kwargs():

    class TestModel(HFTransformer):

        @property
        def hf_pipeline_task(self):
            return "task_name"

    downstream_model_type = "model_type"
    cls_mock = MagicMock()
    backbone_config = MagicMock()

    with patch("lightning_transformers.core.nlp.model.get_class", return_value=cls_mock) as get_class_mock:
        model = TestModel(downstream_model_type, backbone_config, pipeline_kwargs=dict(device=0), foo="bar")
    get_class_mock.assert_called_once_with(downstream_model_type)
    cls_mock.from_pretrained.assert_called_once_with(backbone_config.pretrained_model_name_or_path, foo="bar")

    with patch("lightning_transformers.core.nlp.model.hf_transformers_pipeline") as pipeline_mock:
        model.hf_pipeline  # noqa
        pipeline_mock.assert_called_once_with(
            task="task_name", model=cls_mock.from_pretrained.return_value, tokenizer=None, device=0
        )
