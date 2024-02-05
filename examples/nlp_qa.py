"""
Simple example that show how to use Avalanche for Question Answering on 
Squad by using T5
"""

from avalanche.benchmarks.utils import DataAttribute, ConstantSequence
from avalanche.training.plugins import ReplayPlugin
from transformers import DataCollatorForSeq2Seq
import torch
import avalanche
import torch.nn
from avalanche.benchmarks import CLScenario, CLStream, CLExperience
import avalanche.training.templates.base
from avalanche.benchmarks.utils import AvalancheDataset
from transformers import AutoTokenizer
from transformers import T5ForConditionalGeneration
from datasets import load_dataset
import numpy as np


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
    squad_tr = load_dataset("squad", split="train")
    squad_val = load_dataset("squad", split="validation")

    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    encoder_max_len = tokenizer.model_max_length
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
        examples, encoder_max_len=encoder_max_len, decoder_max_len=decoder_max_len
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

    # Map the preprocessing function to the dataset so that it's applied to
    # all examples
    squad_tr = squad_tr.map(t2t_converter)
    squad_tr = squad_tr.map(preprocess_function, batched=True)
    squad_tr = squad_tr.remove_columns(
        ["id", "title", "context", "question", "answers", "input_text", "target_text"]
    )
    squad_val = squad_val.map(t2t_converter)
    squad_val = squad_val.map(preprocess_function, batched=True)
    # ,' input_text', 'target_text'])
    squad_val = squad_val.remove_columns(
        ["id", "title", "context", "question", "answers"]
    )

    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    # Use a standard data collator for QA
    data_collator = DataCollatorForSeq2Seq(tokenizer)

    train_exps = []
    for i in range(0, 2):
        # We use very small experiences only to showcase the library.
        # Adapt this to your own benchmark
        exp_data = squad_tr.select(range(30 * i, 30 * (i + 1)))
        tl = DataAttribute(ConstantSequence(i, len(exp_data)), "targets_task_labels")

        exp = CLExperience()
        exp.dataset = AvalancheDataset(
            [exp_data], data_attributes=[tl], collate_fn=data_collator
        )
        train_exps.append(exp)
    tl = DataAttribute(ConstantSequence(2, len(squad_val)), "targets_task_labels")
    val_exp = CLExperience()
    val_exp.dataset = AvalancheDataset(
        [squad_val], data_attributes=[tl], collate_fn=data_collator
    )
    val_exp = [val_exp]

    benchmark = CLScenario(
        [
            CLStream("train", train_exps),
            CLStream("valid", val_exp),
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

    # Test the model:
    model.eval()
    question = "Which libraries is Avalanche based upon?"
    context = """
    Avalanche is an End-to-End Continual Learning Library 
    based on PyTorch, born within ContinualAI with the goal of providing 
    a shared and collaborative open-source (MIT licensed) codebase for fast
    prototyping, training and reproducible evaluation of continual learning
    algorithms."
    """

    input_text = f"answer_me: {question} context: {context} </s>"
    encoded_query = tokenizer(
        input_text,
        return_tensors="pt",
        pad_to_max_length=True,
        truncation=True,
        max_length=250,
    )
    input_ids = encoded_query["input_ids"]
    attention_mask = encoded_query["attention_mask"]
    generated_answer = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=50,
        top_p=0.95,
        top_k=50,
        repetition_penalty=2.0,
    )

    decoded_answer = tokenizer.batch_decode(generated_answer, skip_special_tokens=True)
    print(f"Answer: {decoded_answer}")


if __name__ == "__main__":
    main()
