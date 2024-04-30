import numpy as np


def forgetting(accuracy_matrix, boundary_indices=None):
    """Forgetting.

    Given an experience `k` learned at time `boundary_indices[k]`,
    forgetting is the `accuracy_matrix[t, k] - accuracy_matrix[t, boundary_indices[k]]`.
    Forgetting is set to zero before learning on the experience.

    :param accuracy_matrix: 2D accuracy matrix with shape <time, experiences>
    :param boundary_indices: time index for each experience, corresponding to
        the time after the experience was learned. Optional if
        `accuracy_matrix` is a square matrix, which is the most common case.
        In this setting, `boundary_indices[k] = k`.
    :return:
    """
    accuracy_matrix = np.array(accuracy_matrix)
    forgetting_matrix = np.zeros_like(accuracy_matrix)

    if boundary_indices is None:
        assert accuracy_matrix.shape[0] == accuracy_matrix.shape[1]
        boundary_indices = list(range(accuracy_matrix.shape[0]))

    for k in range(accuracy_matrix.shape[1]):
        t_task = boundary_indices[k]
        acc_first = accuracy_matrix[t_task][k]

        # before learning exp k, forgetting is zero
        forgetting_matrix[: t_task + 1, k] = 0
        # then, it's acc_first - acc[t, k]
        forgetting_matrix[t_task + 1 :, k] = (
            acc_first - accuracy_matrix[t_task + 1 :, k]
        )

    return forgetting_matrix
