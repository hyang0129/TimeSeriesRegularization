from collections import UserDict
from yaml import load, FullLoader
from schema import Schema, And, Use, Optional
import os
import inspect


class Config(UserDict):

    schema = Schema(
        {
            "hyperparameters": {
                "NFOLD": And(Use(int), int),
                "truncate_last_timesteps": And(Use(int), int),
                "num_class" : And(Use(int), int),
                "batch_size" : And(Use(int), int),
                "training_epochs" : And(Use(int), int),
                "training_steps_per_epoch" : And(Use(int), int),
            },
            "model" : {
                "architecture_name" : And(Use(str), str),
            },
            "environment" : {
                'type' : And(Use(str), str),
            },
            Optional(object): object,  # for allowing all keys, should be removed at some point probably
        }
    )

    configpypath = inspect.getfile(inspect.currentframe())

    def __init__(self, path_to_config="config.yaml"):
        """
        The config object is used as a dictionary with added functionality to pass along parameters
        for training and evaluation. These files are validated via the validate_yaml function.

        Args:
            path_to_config: path to a config file
        """

        self.yaml_contents = load(open(path_to_config, "r").read(), Loader=FullLoader)
        self.validate_yaml()
        super(Config, self).__init__(self.yaml_contents)
        self.as_attr_dict()

    def validate_yaml(self):
        """
        This method performs the assertions in the try block to validate the yaml file.

        Returns:
            Raises an error or returns nothing
        """

        self.yaml_contents = self.schema.validate(self.yaml_contents)

    def as_attr_dict(self):
        """
        Allows access of values as attributes, making code more readable and/or easier to type.

        Eg. config.task.num_classes vs config['task']['num_classes']

        Returns:
            None
        """

        attrdict = AttrDict.from_nested_dicts(self)

        for k in self.keys():
            assert not hasattr(self, k), (
                "A config top level section has the same name as an existing attribute of the config class. See %s" % k
            )
            setattr(self, k, attrdict[k])

    @classmethod
    def get_default_config(cls):
        """
        We expect the default config to be stored alongside the config.py


        Returns:
            a config object initialized from the default config file
        """

        path = os.path.join(os.path.dirname(cls.configpypath), "config.yaml")

        return cls(path_to_config=path)


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        """Dictionary subclass whose entries can be accessed by attributes (as well
        as normally).
        """
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    @classmethod
    def from_nested_dicts(cls, data):
        """ Construct nested AttrDicts from nested dictionaries. """
        if not isinstance(data, dict) and not isinstance(data, Config):
            return data
        else:
            return cls({key: cls.from_nested_dicts(data[key]) for key in data})


if __name__ == "__main__":
    c = Config()
    print(c)
