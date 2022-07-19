from unittest.mock import Mock

import time
from os.path import expanduser

from tqdm import tqdm

from avalanche.benchmarks import fixed_size_experience_split, SplitMNIST
from avalanche.training import ReservoirSamplingBuffer
from avalanche.training import ParametricBuffer

benchmark = SplitMNIST(
    n_experiences=5,
    dataset_root=expanduser("~") + "/.avalanche/data/mnist/",
)

experience = benchmark.train_stream[0]
print("len experience: ", len(experience.dataset))

# start = time.time()
# buffer = ReservoirSamplingBuffer(100)
# for exp in tqdm(fixed_size_experience_split(experience, 1)):
#     buffer.update_from_dataset(exp.dataset)
#
# end = time.time()
# duration = end - start
# print("ReservoirSampling Duration: ", duration)


start = time.time()
buffer = ParametricBuffer(100)
for exp in tqdm(fixed_size_experience_split(experience, 1)):
    buffer.update(Mock(experience=exp, dataset=exp.dataset))

end = time.time()
duration = end - start
print("ParametricBuffer (random sampling) Duration: ", duration)
