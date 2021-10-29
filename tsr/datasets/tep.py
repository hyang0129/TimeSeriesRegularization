import tensorflow as tf

from tsr.config import Config
from tsr.datasets import DatasetManager

class TEP_DatasetManager(DatasetManager):
	def __init__(self, config: Config):
		pass

	def get_train_and_val_for_fold(self, fold: int) -> (tf.data.Dataset, tf.data.Dataset):
		pass
	

	url = "https://drive.google.com/uc?id=1m6Gkp2tNnnlAzaAVLaWnC2TtXNX2wJV8"

