from tsr.datasets.datasetmanager import DatasetManager
from tsr.config import Config
import gdown
import pandas as pd
import numpy as np
from sklearn import preprocessing
from tqdm.autonotebook import tqdm
import tensorflow as tf
from loguru import logger
import pyreadr as py

from tsr.utils import shell_exec


class TEP_DatasetManager(DatasetManager):
	url = "https://drive.google.com/uc?id=1m6Gkp2tNnnlAzaAVLaWnC2TtXNX2wJV8"

	def __init__(self, config: Config):
		dataframe = self.get_tep_data_as_dataframe()

	def get_train_and_val_for_fold(self, fold: int) -> (tf.data.Dataset, tf.data.Dataset):
		pass

	@classmethod
	def get_tep_data_as_dataframe(cls):
		output = "tep_dataset.zip"
		gdown.download(cls.url, output, quiet = False)

		shell_exec("unzip -q -n tep_dataset.zip")

		# reading train data
		a1 = py.read_r("TEP_FaultFree_Training.RData")
		a2 = py.read_r("TEP_Faulty_Training.RData")
		b1 = cls.fix_column_types(a1['fault_free_training'])
		b2 = cls.fix_column_types(a2['faulty_training'])

		# reading test data
		a3 = py.read_r("TEP_FaultFree_Testing.RData")
		a4 = py.read_r("TEP_Faulty_Testing.RData")
		b3 = cls.fix_column_types(a3['fault_free_testing'])
		b4 = cls.fix_column_types(a4['faulty_testing'])

		b1['split'] = 'train'
		b2['split'] = 'train'
		b3['split'] = 'test'
		b4['split'] = 'test'

		return pd.concat([b1, b2, b3, b4])

	@staticmethod
	def fix_column_types(b1: pd.DataFrame):
		for col in b1.columns:
			b1.loc[:, col] = b1.loc[:, col].astype('float32')
		return b1
