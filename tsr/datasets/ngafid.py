import gdown
from tsr.utils import shell_exec
import pandas as pd
import numpy as np
from sklearn import preprocessing
from tqdm.autonotebook import tqdm
import tensorflow as tf


ngafid_urls = {
    "2021_IAAI_C28": "https://drive.google.com/uc?id=1R5q2s-QavuI6DKj9z2rNxQPIOrbJlwUM",
    "2021_IAAI_C37": "https://drive.google.com/uc?id=1RkEZnddzlwpAG5GCht0HBWBBIcvHfYlT",
}


class NGAFID_DatasetManager():

    def __init__(self):
        pass

    def get_train_and_val_for_fold(self, fold) -> (tf.data.Dataset, tf.data.Dataset):
        return


def list_ngafid_datasets():
    return ngafid_urls


def get_ngafid_dataframe_to_dataset(sources = None, df = None):

    pass


def get_ngafid_dataset_as_dataframe(name="2021_IAAI_C28", scaler = None, skip_scaler = False):

    url = ngafid_urls[name]
    output = "data.csv.gz"
    gdown.download(url, output, quiet=False)

    shell_exec("yes | gzip -d data.csv.gz")

    filename = 'data.csv'
    df_test = pd.read_csv(filename, nrows = 100)

    float_cols = [c for c in df_test if df_test[c].dtype == "float64"]
    float32_cols = {c: np.float16 for c in float_cols}

    df = pd.read_csv(filename, engine = 'c', dtype = float32_cols)
    df['id'] = df.id.astype('int32  ')
    df = df.dropna()  # you can handle nans differently, but ymmv
    sources = df[['id', 'plane_id', 'split', 'date_diff', 'before_after']].drop_duplicates()

    if not skip_scaler:
        df = apply_scaler(df, scaler=scaler)

    return df, sources


def apply_scaler(df, scaler = None):

    input_columns = ['volt1',
                     'volt2',
                     'amp1',
                     'amp2',
                     'FQtyL',
                     'FQtyR',
                     'E1 FFlow',
                     'E1 OilT',
                     'E1 OilP',
                     'E1 RPM',
                     'E1 CHT1',
                     'E1 CHT2',
                     'E1 CHT3',
                     'E1 CHT4',
                     'E1 EGT1',
                     'E1 EGT2',
                     'E1 EGT3',
                     'E1 EGT4',
                     'OAT',
                     'IAS',
                     'VSpd',
                     'NormAc',
                     'AltMSL']

    if scaler is None:
        scaler = preprocessing.MinMaxScaler()
        scaler.fit(df.loc[:, input_columns].sample(100000, random_state = 0))

    arr = df.loc[:, input_columns].values
    res = scaler.transform(arr)

    for i, col in tqdm(enumerate(input_columns)):
        df.loc[:, col] = res[:, i]

    return df


# def get_train_and_val_for_fold(folded_datasets, fold):
#     predict = True
#
#
#     train = []
#     for i in range(NFOLD):
#         if i == fold:
#             val_ds = folded_datasets[i]
#         else:
#             train.append(folded_datasets[i])
#
#     train_ds = None
#     for ds in train:
#         train_ds = ds if train_ds is None else train_ds.concatenate(ds)
#
#     train_ds = prepare_for_training(train_ds, shuffle = True, repeat = True, predict = PREDICT, aug = AUGMENT)
#     val_ds = prepare_for_training(val_ds, shuffle = False, predict = PREDICT)
#
#     return train_ds, val_ds