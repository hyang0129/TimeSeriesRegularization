import gdown
import pandas as pd
import numpy as np
from sklearn import preprocessing
from tqdm.autonotebook import tqdm
import tensorflow as tf

from tsr.utils import shell_exec
from tsr.datasets.common import fix_type
from tsr.methods.augmentation.random_shift import RandomShifter


class NGAFID_DatasetManager:

    ngafid_urls = {
        "2021_IAAI_C28": "https://drive.google.com/uc?id=1R5q2s-QavuI6DKj9z2rNxQPIOrbJlwUM",
        "2021_IAAI_C37": "https://drive.google.com/uc?id=1RkEZnddzlwpAG5GCht0HBWBBIcvHfYlT",
    }

    input_columns = [
        "volt1",
        "volt2",
        "amp1",
        "amp2",
        "FQtyL",
        "FQtyR",
        "E1 FFlow",
        "E1 OilT",
        "E1 OilP",
        "E1 RPM",
        "E1 CHT1",
        "E1 CHT2",
        "E1 CHT3",
        "E1 CHT4",
        "E1 EGT1",
        "E1 EGT2",
        "E1 EGT3",
        "E1 EGT4",
        "OAT",
        "IAS",
        "VSpd",
        "NormAc",
        "AltMSL",
    ]

    def __init__(
        self,
        config,
        name="2021_IAAI_C28",
        scaler=None,
    ):
        self.config = config
        self.name = name
        self.scaler = scaler

        self.dataframe = self.get_ngafid_data_as_dataframe(name=name,
                                                    scaler=scaler)

        self.create_folded_datasets()

    def prepare_for_training(self, ds, shuffle=False, repeat=False, aug=False):

        ds = ds.map(fix_type)

        ds = ds.shuffle(512) if shuffle else ds
        ds = ds.repeat() if repeat else ds
        ds = ds.batch(self.config.model.batch_size, drop_remainder=True)
        ds = ds.map(RandomShifter.from_config(self.config))

        if aug:
            pass
            # batch_aug = get_batch_aug()
            # ds = ds.map(batch_aug)

        if self.config.model.num_class > 2:
            ds = ds.map(lambda x, y: (x, tf.one_hot(y, self.config.model.num_class)))
        else:
            ds = ds.map(lambda x, y: (x, tf.reshape(y, (self.config.model.batch_size, 1))))

        return ds

    def create_folded_datasets(self):

        self.folded_datasets = []
        df = self.dataframe
        for i in range(self.config.hyperparameters.NFOLD):
            self.folded_datasets.append(
                self.ngafid_dataframe_to_dataset(df[df.split == i], truncate_last_timesteps=self.config.hyperparameters.truncate_last_timesteps)
            )

    def get_train_and_val_for_fold(self, fold) -> (tf.data.Dataset, tf.data.Dataset):

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

        train_ds = self.prepare_for_training(train_ds, shuffle=True, repeat=True, predict=True, aug=True)
        val_ds = self.prepare_for_training(val_ds, shuffle=False, predict=True)

        return train_ds, val_ds

    @classmethod
    def get_ngafid_data_as_dataframe(cls, name: str, scaler: object = None, skip_scaler: bool = False) -> pd.DataFrame:


        url = cls.ngafid_urls[name]
        output = "data.csv.gz"
        gdown.download(url, output, quiet=False)

        shell_exec("yes | gzip -d data.csv.gz")

        filename = "data.csv"
        df_test = pd.read_csv(filename, nrows=100)

        float_cols = [c for c in df_test if df_test[c].dtype == "float64"]
        float32_cols = {c: np.float16 for c in float_cols}

        df = pd.read_csv(filename, engine="c", dtype=float32_cols)
        df["id"] = df.id.astype("int32  ")
        df = df.dropna()  # you can handle nans differently, but ymmv
        sources = df[["id", "plane_id", "split", "date_diff", "before_after"]].drop_duplicates()

        if not skip_scaler:
            df = cls.apply_scaler(df, scaler=scaler)

        return df, sources

    @classmethod
    def apply_scaler(cls, df, scaler=None, apply=True):

        if scaler is None:
            scaler = preprocessing.MinMaxScaler()
            scaler.fit(df.loc[:, cls.input_columns].sample(100000, random_state=0))

        if apply:
            arr = df.loc[:, cls.input_columns].values
            res = scaler.transform(arr)

            for i, col in tqdm(enumerate(cls.input_columns)):
                df.loc[:, col] = res[:, i]

        return df, scaler

    @classmethod
    def ngafid_dataframe_to_dataset(cls, df=None, truncate_last_timesteps=4096) -> tf.data.Dataset:

        ids = df.id.unique()

        sensor_datas = []
        afters = []

        for id in tqdm(ids):
            sensor_data = df[df.id == id].iloc[-truncate_last_timesteps:, :23].values

            sensor_data = np.pad(sensor_data, [[0, truncate_last_timesteps - len(sensor_data)], [0, 0]])

            sensor_data = tf.convert_to_tensor(sensor_data, dtype=tf.float32)

            after = df[df.id == id]["before_after"].iloc[0]

            sensor_datas.append(sensor_data)
            afters.append(after)

        sensor_datas = tf.stack(sensor_datas)
        afters = np.stack(afters)

        ds = tf.data.Dataset.from_tensor_slices((sensor_datas, afters))

        return ds
