class ROCKET:
    @classmethod
    def from_config(cls):
        raise NotImplementedError

    def __init__(self):
        raise NotImplementedError

    def fit(self):
        raise NotImplementedError

    def __call__(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError
