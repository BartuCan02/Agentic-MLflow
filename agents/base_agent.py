class BaseAgent:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.augmentation = False

    def enable_augmentation(self):
        self.augmentation = True

    def run_all_models(self):
        raise NotImplementedError("Agent must implement run_all_models().")
