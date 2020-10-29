from torch.utils.data import DataLoader

from avalanche.benchmarks.scenarios import IStepInfo, DatasetPart


class StrategyFlow:
    def __init__(self, train_epochs=1):
        self.train_epochs = train_epochs
        self.plugins = []

        self.current_train_dataset = None
        self.current_train_dataloader = None
        self.current_test_dataset = None
        self.current_test_dataloader = None

    def train(self, step_info: IStepInfo, **kwargs):
        # BeforeTraining
        self.before_training(**kwargs)
        self.make_train_dataset(step_info, **kwargs)
        self.adapt_train_dataset(**kwargs)
        self.make_train_dataloader(**kwargs)

        # ModelTraining
        self.epoch = 0
        for self.epoch in range(self.train_epochs):
            self.before_training_epoch(**kwargs)
            # TrainingLoop
            self.training_epoch(**kwargs)

            self.after_training_epoch(**kwargs)
        # TrainingModelAdaptation

        # AfterTraining
        self.after_training(**kwargs)

    def test(self, step_info: IStepInfo, test_part: DatasetPart, **kwargs):
        # BeforeTesting
        self.set_initial_test_step_id(step_info, test_part)
        self.before_testing(**kwargs)

        # MultiTestLoop: loop over steps
        while self.has_testing_steps_left(step_info):
            self.before_step_testing(**kwargs)
            # TestingModelAdaptation -- empty
            self.make_test_dataset(**kwargs)
            self.adapt_test_dataset(**kwargs)
            self.make_test_dataloader(**kwargs)

            # ModelTesting
            self.before_testing_epoch(**kwargs)
            # TestingEpoch
            self.testing_epoch(**kwargs)

            self.after_testing_epoch(**kwargs)
            self.after_step_testing(**kwargs)

            self.step_id += 1  # equivalent to self.next_testing_step

        # AfterTesting
        self.after_testing(**kwargs)

    def before_training(self, **kwargs):
        [p.before_training(**kwargs) for p in self.plugins]

    def make_train_dataset(self, step_info: IStepInfo, **kwargs):
        self.current_train_dataset = step_info.current_training_set()[0]

    def make_train_dataloader(self, num_workers=0,
                              train_mb_size=1, **kwargs):
        self.current_train_data_loader = DataLoader(self.current_train_dataset,
            num_workers=num_workers, batch_size=train_mb_size)

    def set_initial_test_step_id(self, step_info: IStepInfo,
                                 dataset_part: DatasetPart = None):
        # TODO: if we remove DatasetPart this may become unnecessary
        self.step_id = -1
        if dataset_part is None:
            dataset_part = DatasetPart.COMPLETE

        if dataset_part == DatasetPart.CURRENT:
            self.step_id = step_info.current_step
        if dataset_part in [DatasetPart.CUMULATIVE, DatasetPart.OLD,
                            DatasetPart.COMPLETE]:
            self.step_id = 0
        if dataset_part == DatasetPart.FUTURE:
            self.step_id = step_info.current_step + 1

        if self.step_id < 0:
            raise ValueError('Invalid dataset part')

    def has_testing_steps_left(self, step_info: IStepInfo,
                               test_part: DatasetPart = None):
        step_id = self.step_id
        if test_part is None:
            test_part = DatasetPart.COMPLETE

        if test_part == DatasetPart.CURRENT:
            return step_id == step_info.current_step
        if test_part == DatasetPart.CUMULATIVE:
            return step_id <= step_info.current_step
        if test_part == DatasetPart.OLD:
            return step_id < step_info.current_step
        if test_part == DatasetPart.FUTURE:
            return step_info.current_step < step_id < step_info.n_steps
        if test_part == DatasetPart.COMPLETE:
            return step_id < step_info.n_steps

        raise ValueError('Invalid dataset part')

    def make_test_dataset(self, step_info: IStepInfo, step_id: int):
        self.current_test_dataset = step_info.step_specific_test_set(step_id)[0]

    def make_test_dataloader(self, num_workers=0, test_mb_size=1):
        self.currrent_test_data_loader = DataLoader(self.current_test_dataset,
              num_workers=num_workers, batch_size=test_mb_size)

    def adapt_train_dataset(self, **kwargs):
        [p.adapt_train_dataset(**kwargs) for p in self.plugins]

    def before_training_epoch(self, **kwargs):
        [p.before_training_epoch(**kwargs) for p in self.plugins]

    def training_epoch(self, **kwargs):
        pass

    def after_training_epoch(self, **kwargs):
        [p.after_training_epoch(**kwargs) for p in self.plugins]

    def after_training(self, **kwargs):
        [p.after_training(**kwargs) for p in self.plugins]

    def before_testing(self, **kwargs):
        [p.before_testing(**kwargs) for p in self.plugins]

    def before_step_testing(self, **kwargs):
        [p.before_step_testing(**kwargs) for p in self.plugins]

    def adapt_test_dataset(self, **kwargs):
        [p.adapt_test_dataset(**kwargs) for p in self.plugins]

    def before_testing_epoch(self, **kwargs):
        [p.before_testing_epoch(**kwargs) for p in self.plugins]

    def testing_epoch(self, **kwargs):
        pass

    def after_testing_epoch(self, **kwargs):
        [p.after_testing_epoch(**kwargs) for p in self.plugins]

    def after_step_testing(self, **kwargs):
        [p.after_step_testing(**kwargs) for p in self.plugins]

    def after_testing(self, **kwargs):
        [p.after_testing(**kwargs) for p in self.plugins]
