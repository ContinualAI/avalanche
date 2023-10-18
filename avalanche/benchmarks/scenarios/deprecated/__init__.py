from abc import ABC
from typing import Generic, List, Optional, Sequence, TypeVar

from .. import CLExperience, CLStream, CLScenario

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from avalanche.benchmarks.utils import AvalancheDataset

TCLDataset = TypeVar("TCLDataset", bound="AvalancheDataset")
TDatasetExperience = TypeVar(
    "TDatasetExperience", bound="DatasetExperience"
)  # Implementation, defined here


class DatasetExperience(CLExperience, Generic[TCLDataset], ABC):
    """Base Experience.

    Experiences have an index which track the experience's position
    inside the stream for evaluation purposes.
    """

    def __init__(
        self: TDatasetExperience,
        current_experience: int,
        origin_stream: "CLStream[TDatasetExperience]",
        benchmark: "CLScenario",
        dataset: TCLDataset,
    ):
        super().__init__(
            current_experience=current_experience, origin_stream=origin_stream
        )

        self._benchmark: CLScenario = benchmark
        self._dataset: TCLDataset = dataset

    @property
    def benchmark(self) -> "CLScenario":
        bench = self._benchmark
        CLExperience._check_unset_attribute("benchmark", bench)
        return bench

    @benchmark.setter
    def benchmark(self, bench: "CLScenario"):
        self._benchmark = bench

    @property
    def dataset(self) -> TCLDataset:
        data = self._dataset
        CLExperience._check_unset_attribute("dataset", data)
        return data

    @dataset.setter
    def dataset(self, d: TCLDataset):
        self._dataset = d

    @property
    def task_label(self) -> int:
        """
        The task label. This value will never have value "None". However,
        for scenarios that don't produce task labels a placeholder value like 0
        is usually set. Beware that this field is meant as a shortcut to obtain
        a unique task label: it assumes that only patterns labeled with a
        single task label are present. If this experience contains patterns from
        multiple tasks, accessing this property will result in an exception.
        """
        if len(self.task_labels) != 1:
            raise ValueError(
                "The task_label property can only be accessed "
                "when the experience contains a single task label"
            )

        return self.task_labels[0]

    @property
    def task_labels(self) -> List[int]:
        task_labels = getattr(self.dataset, "targets_task_labels", None)

        assert task_labels is not None, (
            "In its default implementation, DatasetExperience will use the "
            "the dataset `targets_task_labels` field to compute the "
            "content of the `task_label(s)` field. The given does not "
            "contain such field."
        )

        return list(set(task_labels))


class AbstractClassTimelineExperience(DatasetExperience[TCLDataset], ABC):
    """
    Definition of a learning experience. A learning experience contains a set of
    patterns which has become available at a particular time instant. The
    content and size of an Experience is defined by the specific benchmark that
    creates the experience.

    For instance, an experience of a New Classes scenario will contain all
    patterns belonging to a subset of classes of the original training set. An
    experience of a New Instance scenario will contain patterns from previously
    seen classes.
    """

    def __init__(
        self: TDatasetExperience,
        origin_stream: "CLStream[TDatasetExperience]",
        dataset: TCLDataset,
        current_experience: int,
        classes_in_this_exp: Optional[Sequence[int]],
        previous_classes: Optional[Sequence[int]],
        classes_seen_so_far: Optional[Sequence[int]],
        future_classes: Optional[Sequence[int]],
    ):
        """
        Creates an instance of an experience given the benchmark
        stream, the current experience ID and data about the classes timeline.

        :param origin_stream: The stream from which this experience was
            obtained.
        :param current_experience: The current experience ID, as an integer.
        :param classes_in_this_exp: The list of classes in this experience.
        :param previous_classes: The list of classes in previous experiences.
        :param classes_seen_so_far: List of classes of current and previous
            experiences.
        :param future_classes: The list of classes of next experiences.
        """

        self.classes_in_this_experience: Optional[Sequence[int]] = classes_in_this_exp
        """ The list of classes in this experience """

        self.previous_classes: Optional[Sequence[int]] = previous_classes
        """ The list of classes in previous experiences """

        self.classes_seen_so_far: Optional[Sequence[int]] = classes_seen_so_far
        """ List of classes of current and previous experiences """

        self.future_classes: Optional[Sequence[int]] = future_classes
        """ The list of classes of next experiences """

        super().__init__(
            current_experience=current_experience,
            origin_stream=origin_stream,
            benchmark=origin_stream.benchmark,  # type: ignore
            dataset=dataset,
        )
