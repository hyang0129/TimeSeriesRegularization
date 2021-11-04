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
from sklearn.model_selection import train_test_split, KFold
from tsr.utils import shell_exec


class TEP_DatasetManager(DatasetManager):
    url = "https://drive.google.com/uc?id=1m6Gkp2tNnnlAzaAVLaWnC2TtXNX2wJV8"
    num_examples = 10500

    def __init__(self, config: Config):
        self.config = config
        self.dataframe, self.scaler = self.get_tep_data_as_dataframe()
        self.dataframe = self.apply_scaler(self.dataframe, self.scaler)

        self.folded_datasets = self.get_split_train_dataset_from_dataframe(self.dataframe)
        self.test_dataset = self.get_test_dataset_from_dataframe(self.dataframe)

    def prepare_tfdataset(self, ds, shuffle: bool = False, repeat: bool = False, aug: bool = False) -> tf.data.Dataset:
        return ds

    def get_train_and_val_for_fold(self, fold: int):
        logger.debug("Retrieving Fold %i" % fold)
        config = self.config

        train = []
        for i in range(config.hyperparameters.NFOLD):
            if i == fold:
                val_ds = self.folded_datasets[i]
            else:
                train.append(self.folded_datasets[i])

        train_ds = None
        for ds in train:
            train_ds = ds if train_ds is None else train_ds.concatenate(ds)

        train_ds = self.prepare_tfdataset(train_ds, shuffle=True, repeat=True, aug=True)
        val_ds = self.prepare_tfdataset(val_ds, shuffle=False)

        logger.debug("Successfully Retrieved Fold %i" % fold)
        return train_ds, val_ds

    @classmethod
    def get_tep_data_as_dataframe(cls):
        output = "tep_dataset.zip"
        gdown.download(cls.url, output, quiet=False)

        shell_exec("unzip -q -n tep_dataset.zip")

        # reading train data
        a1 = py.read_r("TEP_FaultFree_Training.RData")
        a2 = py.read_r("TEP_Faulty_Training.RData")
        b1 = cls.fix_column_types(a1["fault_free_training"])
        b2 = cls.fix_column_types(a2["faulty_training"])

        # reading test data
        a3 = py.read_r("TEP_FaultFree_Testing.RData")
        a4 = py.read_r("TEP_Faulty_Testing.RData")
        b3 = cls.fix_column_types(a3["fault_free_testing"])
        b4 = cls.fix_column_types(a4["faulty_testing"])

        b1["split"] = "train"
        b2["split"] = "train"
        b3["split"] = "test"
        b4["split"] = "test"

        df = pd.concat([b1, b2, b3, b4])

        df["id"] = df.faultNumber.apply(lambda x: int(x)) + df.simulationRun.apply(lambda x: int(x) * 100)

        scaler = preprocessing.MinMaxScaler()
        scaler.fit(df.iloc[:, 3:55][df.split == "train"].values)

        return df, scaler

    @classmethod
    def apply_scaler(cls, df, scaler):
        arr = df.iloc[:, 3:55].values
        arr = scaler.transform(arr)

        for i in tqdm(range(52)):
            df.iloc[:, i + 3] = arr[:, i]

        return df

    @staticmethod
    def get_train_dataset_from_dataframe(df):
        arr = df[df.split == "train"].iloc[:, :55].values
        arr = np.reshape(arr, (-1, 500, 55))
        ds = tf.data.Dataset.from_tensor_slices(arr)
        ds = ds.map(lambda x: {"input": x[:, 3:], "target": x[0, 0]})
        return ds

    @staticmethod
    def get_test_dataset_from_dataframe(df):
        arr = df[df.split == "test"].iloc[:, :55].values
        arr = np.reshape(arr, (-1, 960, 55))
        ds = tf.data.Dataset.from_tensor_slices(arr)
        ds = ds.map(lambda x: {"input": x[:, 3:], "target": x[0, 0]})
        return ds

    @staticmethod
    def fix_column_types(b1: pd.DataFrame):
        for col in b1.columns:
            b1.loc[:, col] = b1.loc[:, col].astype("float32")
        return b1

    def get_split_train_dataset_from_dataframe(self, df):

        arr = df[df.split == "train"].iloc[:, :55].values
        arr = np.reshape(arr, (-1, 500, 55))

        train_splits = []

        for train_split, val_split in KFold(5, shuffle=True, random_state=0).split([i + 1 for i in range(10500)]):
            ds = tf.data.Dataset.from_tensor_slices(arr[val_split])
            ds = ds.map(lambda x: {"input": x[:, 3:], "target": x[0, 0]})
            train_splits.append(ds)

        return train_splits
