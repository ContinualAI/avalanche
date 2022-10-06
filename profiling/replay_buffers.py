from unittest.mock import Mock

import time
from os.path import expanduser

from tqdm import tqdm

from avalanche.benchmarks import fixed_size_experience_split, SplitMNIST
from avalanche.benchmarks.utils.flat_data import _flatdata_depth
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
    bgs = buffer.buffer_datasets

    depths = [_flatdata_depth(b) for b in bgs]
    lenidxs = [len(b._indices) for b in bgs]
    lendsets = [len(b._datasets) for b in bgs]
    lentots = sum([len(b) for b in bgs])
    print(f"DATA depth={depths}, idxs={lenidxs}, dsets={lendsets}, len={lentots}")

    atts = [bgs[0].targets.data, bgs[0].targets_task_labels.data]
    depths = [_flatdata_depth(b) for b in atts]
    lenidxs = [len(b._indices) for b in atts]
    lendsets = [len(b._datasets) for b in atts]
    lentots = sum([len(b) for b in bgs])
    print(f"ATTS depth={depths}, idxs={lenidxs}, dsets={lendsets}, len={lentots}")


end = time.time()
duration = end - start
print("ParametricBuffer (random sampling) Duration: ", duration)
