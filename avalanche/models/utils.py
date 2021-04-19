from avalanche.models.dynamic_modules import MultiTaskModule


def avalanche_forward(model, x, task_labels):
    if isinstance(model, MultiTaskModule):
        return model.forward(x, task_labels)
    else:  # no task labels
        return model.forward(x)
