################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 15-06-2021                                                             #
# Author(s): Lorenzo Pellegrini                                                #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################

from collections import defaultdict
from typing import Sequence, Iterable, Dict, Optional, Iterator

from avalanche.benchmarks.utils import AvalancheDataset


class LazyDatasetSequence(Sequence[AvalancheDataset]):
    """
    A lazily initialized sequence of datasets.

    This class provides a way to lazily generate and store the datasets
    linked to each experience. This class uses a generator to get the sequence
    of datasets but it can also be used with a more classic statically
    initialized Sequence (like a list).

    This class will also keep track of the targets and task labels field of the
    generated datasets.
    """

    def __init__(
        self,
        experience_generator: Iterable[AvalancheDataset],
        stream_length: int,
    ):
        self._exp_source: Optional[
            Iterable[AvalancheDataset]
        ] = experience_generator
        """
        The source of the experiences stream, as an Iterable.
        
        Can be a simple Sequence or a Generator.
        
        This field is kept for reference and debugging. The actual generator
        is kept in the `_exp_generator` field, which stores the iterator.
        
        This field is None when if all the experiences have been loaded.
        """

        self._next_exp_id: int = 0
        """
        The ID of the next experience that will be generated.
        """

        self._loaded_experiences: Dict[int, AvalancheDataset] = dict()
        """
        The sequence of experiences obtained from the generator.
        """

        self._stream_length: int = stream_length
        """
        The length of the stream.
        """
        try:
            self._exp_generator: Optional[Iterator[AvalancheDataset]] = iter(
                self._exp_source
            )
        except TypeError as e:
            if callable(self._exp_source):
                # https://stackoverflow.com/a/17092033
                raise ValueError(
                    "The provided generator is not iterable. When using a "
                    'generator function based on "yield", remember to pass the'
                    " result of that function, not the "
                    "function itself!"
                ) from None
            raise e
        """
        The experience generator, as an Iterator.
        
        This field is None when if all the experiences have been loaded.
        """

        self.targets_field_sequence: Dict[
            int, Optional[Sequence]
        ] = defaultdict(lambda: None)
        """
        A dictionary mapping each experience to its `targets` field.
        
        This dictionary contains the targets field of datasets generated up to
        now, including the ones of dropped experiences.
        """

        self.task_labels_field_sequence: Dict[
            int, Optional[Sequence[int]]
        ] = defaultdict(lambda: None)
        """
        A dictionary mapping each experience to its `targets_task_labels` field.

        This dictionary contains the task labels of datasets generated up to
        now, including the ones of dropped experiences.
        """

    def __len__(self) -> int:
        """
        Gets the length of the stream (number of experiences).

        :return: The length of the stream.
        """
        return self._stream_length

    def __getitem__(self, exp_idx: int) -> AvalancheDataset:
        """
        Gets the dataset associated to an experience.

        :param exp_idx: The ID of the experience.
        :return: The dataset associated to the experience.
        """
        exp_idx = int(exp_idx)  # Handle single element tensors
        self.load_all_experiences(exp_idx)
        if exp_idx not in self._loaded_experiences:
            raise RuntimeError(f"Experience {exp_idx} has been dropped")

        return self._loaded_experiences[exp_idx]

    def get_experience_if_loaded(
        self, exp_idx: int
    ) -> Optional[AvalancheDataset]:
        """
        Gets the dataset associated to an experience.

        Differently from `__getitem__`, this will return None if the experience
        has not been (lazily) loaded yet.

        :param exp_idx: The ID of the experience.
        :return: The dataset associated to the experience or None if the
            experience has not been loaded yet or if it has been dropped.
        """
        exp_idx = int(exp_idx)  # Handle single element tensors
        if exp_idx >= len(self):
            raise IndexError(
                f"The stream doesn't contain {exp_idx+1}" f"experiences"
            )

        return self._loaded_experiences.get(exp_idx, None)

    def drop_previous_experiences(self, to_exp: int) -> None:
        """
        Drop the reference to experiences up to a certain experience ID
        (inclusive).

        This means that experiences with ID [0, from_exp] will be released.
        Beware that the associated object will be valid until all the references
        to it are dropped.

        :param to_exp: The ID of the last exp to drop (inclusive). If None,
            the whole stream will be loaded. Can be a negative number, in
            which case this method doesn't have any effect. Can be greater
            or equal to the stream length, in which case all currently loaded
            experiences will be dropped.
        :return: None
        """

        to_exp = int(to_exp)  # Handle single element tensors
        if to_exp < 0:
            return

        to_exp = min(to_exp, len(self) - 1)

        for exp_id in range(0, to_exp + 1):
            if exp_id in self._loaded_experiences:
                del self._loaded_experiences[exp_id]

    def load_all_experiences(self, to_exp: int = None) -> None:
        """
        Load all experiences up to a certain experience ID (inclusive).

        Beware that this won't re-load any already dropped experience.

        :param to_exp: The ID of the last exp to load (inclusive). If None,
            the whole stream will be loaded.
        :return: None
        """
        if to_exp is None:
            to_exp = len(self) - 1
        else:
            to_exp = int(to_exp)  # Handle single element tensors

        if to_exp >= len(self):
            raise IndexError(
                f"The stream doesn't contain {to_exp+1}" f"experiences"
            )

        if self._next_exp_id > to_exp:
            # Nothing to do
            return

        for exp_id in range(self._next_exp_id, to_exp + 1):
            try:
                generated_exp: AvalancheDataset = next(self._exp_generator)
            except StopIteration:
                raise RuntimeError(
                    f"Unexpected end of stream. The generator was supposed to "
                    f"generate {len(self)} experiences, but an error occurred "
                    f"while generating experience {exp_id}."
                )

            if not isinstance(generated_exp, AvalancheDataset):
                raise ValueError(
                    "All experience datasets must be subclasses of"
                    " AvalancheDataset"
                )

            self._loaded_experiences[exp_id] = generated_exp
            self.targets_field_sequence[exp_id] = generated_exp.targets
            self.task_labels_field_sequence[
                exp_id
            ] = generated_exp.targets_task_labels
            self._next_exp_id += 1

        if self._next_exp_id == len(self):
            # Release all references to the generator
            self._exp_generator = None
            self._exp_source = None


__all__ = ["LazyDatasetSequence"]
