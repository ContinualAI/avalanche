"""
An example to show how to use huggingface datasets by using a wrapper to 
convert them to AvalancheDataset and using them for CL experiments.

This example requires datasets and transformers libraries

You can install them by running:
pip install datasets transformers
"""

import datasets as ds
import numpy as np
import torch
import torch.nn
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    T5ForConditionalGeneration,
)

import avalanche
import avalanche.training.templates.base
from avalanche.benchmarks import CLExperience, CLScenario, CLStream
from avalanche.benchmarks.utils import AvalancheDataset, ConstantSequence, DataAttribute
from avalanche.benchmarks.utils.data import AvalancheDataset
from avalanche.benchmarks.utils.data_attribute import DataAttribute
from avalanche.benchmarks.utils.flat_data import ConstantSequence
from avalanche.training.plugins import ReplayPlugin


class HFTextDataWrapper:
    """
    A simple wrapper class to use HugginFace Datasets and convert them
    to AvalancheDatasets
    """

    def __init__(self, dataset_name, split) -> None:
        self.dataset_name = dataset_name
        self.split = split
        self.dataset = []
        self.collate_fn = None

    def download_data(self, **kwargs):
        self.dataset = ds.load_dataset(self.dataset_name, split=self.split, **kwargs)

    def add_collate_function(self, collate_fn):
        self.collate_fn = collate_fn

    def map_preprocess_func(self, preproc_func, batched, columns_to_keep=[]):
        """
        Applies a preprocessing function to the wrapped Hugging Face Datasets.

        Args:
        - preproc_func: A preprocessing function that will be applied to the
            dataset. See H.F. library documentation for details.
        - batched: A boolean indicating whether the preprocessing function
            should be applied to the dataset in batches.
        - columns_to_keep: A list of column names to keep in the dataset
            after the preprocessing function has been applied. If set to an
            empty list (default), ONLY columns added by the preproc_func will
            be kept.

        Returns:
        None
        """

        if len(columns_to_keep) == 0:
            old_f = self._all_features()
            self.dataset = self.dataset.map(preproc_func, batched=batched)
            new_f = self._all_features() - old_f
            print(f"The following columns are removed from dataset: {old_f}")
            self.dataset = self.dataset.remove_columns(list(old_f))
            print(f"Kept columns: {new_f - old_f}")
            print(
                "If the resulting dataset has 0 columns left. Please ensure"
                "that the preprocessing phase saves the modified features in"
                "new columns or pass a list of column names"
            )
        else:
            old_f = self._all_features()
            self.dataset = self.dataset.map(preproc_func, batched=batched)
            to_remove = old_f - columns_to_keep
            self.dataset = self.dataset.remove_columns(list(to_remove))
            print(
                f"The following columns have been removed" "from dataset: {to_remove}"
            )
        print("Dataset features: ", self.dataset.features.keys())

    def to_avalanche_dataset(self, dataset_index):
        tl = DataAttribute(
            ConstantSequence(dataset_index, len(self.dataset)), "targets_task_labels"
        )
        return AvalancheDataset(
            [self.dataset], data_attributes=[tl], collate_fn=self.collate_fn
        )

    def _all_features(self):
        return self.dataset.features.keys()

    def _get_hf_dataset(self):
        return self.dataset


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
    # Load SQuAD datasets from HuggingFace
    data_wrap = HFTextDataWrapper(dataset_name="squad", split="train")

    encoder_max_len = 40
    decoder_max_len = 60

    """
    Define a preprocessing function (code from HuggingFace) to:
    1. Convert squad dataset to be used in a text 2 text setting
    2. Encode the question and context with the tokenizer of T5 model
    """

    def t2t_converter(example):
        example["input_text"] = f"question: {example['question']}"
        +f"context: {example['context']} </s>"
        example["target_text"] = f"{example['answers']['text'][0]} </s>"
        return example

    def preprocess_function(
        examples,
        encoder_max_len=encoder_max_len,
        decoder_max_len=decoder_max_len,
        tokenizer=AutoTokenizer.from_pretrained("t5-small"),
    ):
        encoder_inputs = tokenizer(
            examples["input_text"],
            truncation=True,
            return_tensors="pt",
            max_length=encoder_max_len,
            pad_to_max_length=True,
        )

        decoder_inputs = tokenizer(
            examples["target_text"],
            truncation=True,
            return_tensors="pt",
            max_length=decoder_max_len,
            pad_to_max_length=True,
        )

        input_ids = encoder_inputs["input_ids"]
        input_attention = encoder_inputs["attention_mask"]
        target_ids = decoder_inputs["input_ids"]
        target_attention = decoder_inputs["attention_mask"]

        outputs = {
            "input_ids": input_ids,
            "attention_mask": input_attention,
            "labels": target_ids,
            "decoder_attention_mask": target_attention,
        }
        return outputs

    # define the data collator to pass to the resulting avalanche dataset
    data_collator = DataCollatorForSeq2Seq(AutoTokenizer.from_pretrained("t5-small"))
    data_wrap.add_collate_function(data_collator)

    # download the dataset
    data_wrap.download_data()

    # Optional: define the columns to keep after applying the preprocessing
    # function By default, only columns added to dataset by the preprocessing
    # function are kept
    columns_list = ["input_ids", "attention_masks", "decoder_attention_mask", "labels"]
    data_wrap.map_preprocess_func(
        preproc_func=t2t_converter, batched=False, columns_to_keep=columns_list
    )
    data_wrap.map_preprocess_func(
        preproc_func=preprocess_function, batched=True, columns_to_keep=columns_list
    )

    # Convert to an AvalancheDataset
    dataset = data_wrap.to_avalanche_dataset(1)

    # Print the type
    print(dataset, type(dataset))

    # Init a model
    model = T5ForConditionalGeneration.from_pretrained("t5-small")

    train_exps = []
    exp = CLExperience()
    exp.dataset = dataset
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
    # Define a Strategy
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
