"""
Example adapting Avalanche to use Huggingface models and datasets.
To run this example you need huggingface datasets and transformers libraries.

You can install them by running:
pip install datasets transformers
"""

from avalanche.benchmarks.utils import DataAttribute, ConstantSequence
from avalanche.training.plugins import ReplayPlugin

from dataclasses import dataclass

from transformers import PreTrainedTokenizerBase
from typing import Optional, Union, Any

from transformers.utils import PaddingStrategy
import torch

import avalanche
import torch.nn

from avalanche.benchmarks import CLScenario, CLStream, CLExperience
from avalanche.evaluation.metrics import accuracy_metrics
import avalanche.training.templates.base
from avalanche.benchmarks.utils import AvalancheDataset
from transformers import AutoTokenizer
from transformers import T5ForConditionalGeneration
from datasets import load_dataset
import numpy as np


@dataclass
class CustomDataCollatorSeq2SeqBeta:
    """The collator is a standard huggingface collate.
    No need to change anything here.
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):

        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = (
            [feature["labels"] for feature in features]
            if "labels" in features[0].keys()
            else None
        )
        # We have to pad the labels before calling `tokenizer.pad` as this
        # method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(lab) for lab in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (
                    max_label_length - len(feature["labels"])
                )
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder
                        if padding_side == "right"
                        else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate(
                        [feature["labels"], remainder]
                    ).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate(
                        [remainder, feature["labels"]]
                    ).astype(np.int64)

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # prepare decoder_input_ids
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = (
                self.model.prepare_decoder_input_ids_from_labels(
                    labels=features["labels"]
                )
            )
            features["decoder_input_ids"] = decoder_input_ids

        return features


class HGNaive(avalanche.training.Naive):
    """There are only a couple of modifications needed to
    use huggingface:
    - we add a bunch of attributes corresponding to the batch items,
        redefining mb_x and mb_y too
    - _unpack_minibatch sends the dictionary values to the GPU device
    - forward and criterion are adapted for machine translation tasks.
    """

    @property
    def mb_attention_mask(self):
        return self.mbatch["attention_mask"]

    @property
    def mb_x(self):
        """Current mini-batch input."""
        return self.mbatch["input_ids"]

    @property
    def mb_y(self):
        """Current mini-batch target."""
        return self.mbatch["labels"]

    @property
    def mb_decoder_in_ids(self):
        """Current mini-batch target."""
        return self.mbatch["decoder_input_ids"]

    @property
    def mb_token_type_ids(self):
        return self.mbatch[3]

    def _unpack_minibatch(self):
        """HuggingFace minibatches are dictionaries of tensors.
        Move tensors to the current device."""
        for k in self.mbatch.keys():
            self.mbatch[k] = self.mbatch[k].to(self.device)

    def forward(self):
        out = self.model(
            input_ids=self.mb_x,
            attention_mask=self.mb_attention_mask,
            labels=self.mb_y,
        )
        return out.logits

    def criterion(self):
        mb_output = self.mb_output.view(-1, self.mb_output.size(-1))
        ll = self._criterion(mb_output, self.mb_y.view(-1))
        return ll


def main():
    tokenizer = AutoTokenizer.from_pretrained("t5-small", padding=True)
    tokenizer.save_pretrained(
        "./MLDATA/NLP/hf_tokenizers"
    )  # CHANGE DIRECTORY

    prefix = "<2en>"
    source_lang = "de"
    target_lang = "en"
    remote_data = load_dataset("news_commentary", "de-en")

    def preprocess_function(examples):
        inputs = [
            prefix + example[source_lang] for example in examples["translation"]
        ]
        targets = [example[target_lang] for example in examples["translation"]]
        model_inputs = tokenizer(inputs, max_length=128, truncation=True)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=128, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    remote_data = remote_data.map(preprocess_function, batched=True)
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    remote_data = remote_data.remove_columns(["id", "translation"])
    remote_data.set_format(type="torch")
    data_collator = CustomDataCollatorSeq2SeqBeta(
        tokenizer=tokenizer, model=model
    )

    train_exps = []
    for i in range(0, 2):
        # We use very small experiences only to showcase the library.
        # Adapt this to your own benchmark
        exp_data = remote_data["train"].select(range(30 * i, 30 * (i + 1)))
        tl = DataAttribute(
            ConstantSequence(i, len(exp_data)), "targets_task_labels"
        )

        exp = CLExperience()
        exp.dataset = AvalancheDataset(
            [exp_data], data_attributes=[tl], collate_fn=data_collator
        )
        train_exps.append(exp)

    benchmark = CLScenario(
        [
            CLStream("train", train_exps),
            # add more stream here (validation, test, out-of-domain, ...)
        ]
    )
    eval_plugin = avalanche.training.plugins.EvaluationPlugin(
        avalanche.evaluation.metrics.loss_metrics(
            epoch=True, experience=True, stream=True
        ),
        loggers=[avalanche.logging.InteractiveLogger()],
        strict_checks=False,
    )
    plugins = [ReplayPlugin(mem_size=200)]
    optimizer = torch.optim.Adam(model.parameters(), lr=2)
    strategy = HGNaive(
        model,
        optimizer,
        torch.nn.CrossEntropyLoss(ignore_index=-100),
        evaluator=eval_plugin,
        train_epochs=1,
        train_mb_size=10,
        plugins=plugins,
    )
    for experience in benchmark.train_stream:
        strategy.train(experience, collate_fn=data_collator)
        strategy.eval(benchmark.train_stream)


if __name__ == "__main__":
    main()
