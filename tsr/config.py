from collections import UserDict
from yaml import load, FullLoader


class Config(UserDict):
    def __init__(self, path_to_config="config.yaml"):
        """
        The config object is used as a dictionary with added functionality to pass along parameters
        for training and evaluation. These files are validated via the validate_yaml function.

        Args:
            path_to_config: path to a config file
        """

        super(Config, self).__init__(load(open(path_to_config, "r").read(), Loader=FullLoader))
        self.validate_yaml()
        self.as_attr_dict()

    def validate_yaml(self):
        """
        This method performs the assertions in the try block to validate the yaml file.

        Returns:
            Raises an error or returns nothing
        """

        try:

            # TODO: Alternative YAML schema validation should be explored
            #   after a single full training run has been successfully executed using the yaml configs

            assert self["task"]["type"] in ["object detection"]
            assert isinstance(self["task"]["num_classes"], int)
            assert self["model"]["architecture"] in ["RetinaNet"]

        except AssertionError as e:
            msg = "It appears your YAML configuration is invalid." "The traceback will identify which assertion failed"
            if len(e.args) >= 1:
                e.args = (e.args[0] + msg,) + e.args[1:]
            else:
                e.args = [msg]
            raise e

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

    print(c.task.num_classes)
