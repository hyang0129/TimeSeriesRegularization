from tsr.datasets.datasetmanager import DatasetManager
from tsr.config import Config
from tsr.utils import shell_exec
from sktime.utils.data_io import load_from_tsfile_to_dataframe
from sktime.datatypes._panel._convert import (
    from_nested_to_3d_numpy,
)
from loguru import logger

import pandas as pd
import numpy as np
import tensorflow as tf


class UAE_DatasetManager(DatasetManager):

    url = "http://www.timeseriesclassification.com/Downloads/Archives/Multivariate2018_ts.zip"

    directories = {
        "ArticularyWordRecognition": {
            "TEST": "Multivariate_ts/ArticularyWordRecognition/ArticularyWordRecognition_TEST.ts",
            "TRAIN": "Multivariate_ts/ArticularyWordRecognition/ArticularyWordRecognition_TRAIN.ts",
        },
        "AtrialFibrillation": {
            "TEST": "Multivariate_ts/AtrialFibrillation/AtrialFibrillation_TEST.ts",
            "TRAIN": "Multivariate_ts/AtrialFibrillation/AtrialFibrillation_TRAIN.ts",
        },
        "BasicMotions": {
            "TEST": "Multivariate_ts/BasicMotions/BasicMotions_TEST.ts",
            "TRAIN": "Multivariate_ts/BasicMotions/BasicMotions_TRAIN.ts",
        },
        "CharacterTrajectories": {
            "TEST": "Multivariate_ts/CharacterTrajectories/CharacterTrajectories_TEST.ts",
            "TRAIN": "Multivariate_ts/CharacterTrajectories/CharacterTrajectories_TRAIN.ts",
        },
        "Cricket": {
            "TEST": "Multivariate_ts/Cricket/Cricket_TEST.ts",
            "TRAIN": "Multivariate_ts/Cricket/Cricket_TRAIN.ts",
        },
        "DuckDuckGeese": {
            "TEST": "Multivariate_ts/DuckDuckGeese/DuckDuckGeese_TEST.ts",
            "TRAIN": "Multivariate_ts/DuckDuckGeese/DuckDuckGeese_TRAIN.ts",
        },
        "ERing": {"TEST": "Multivariate_ts/ERing/ERing_TEST.ts", "TRAIN": "Multivariate_ts/ERing/ERing_TRAIN.ts"},
        "EigenWorms": {
            "TEST": "Multivariate_ts/EigenWorms/EigenWorms_TEST.ts",
            "TRAIN": "Multivariate_ts/EigenWorms/EigenWorms_TRAIN.ts",
        },
        "Epilepsy": {
            "TEST": "Multivariate_ts/Epilepsy/Epilepsy_TEST.ts",
            "TRAIN": "Multivariate_ts/Epilepsy/Epilepsy_TRAIN.ts",
        },
        "EthanolConcentration": {
            "TEST": "Multivariate_ts/EthanolConcentration/EthanolConcentration_TEST.ts",
            "TRAIN": "Multivariate_ts/EthanolConcentration/EthanolConcentration_TRAIN.ts",
        },
        "FaceDetection": {
            "TEST": "Multivariate_ts/FaceDetection/FaceDetection_TEST.ts",
            "TRAIN": "Multivariate_ts/FaceDetection/FaceDetection_TRAIN.ts",
        },
        "FingerMovements": {
            "TEST": "Multivariate_ts/FingerMovements/FingerMovements_TEST.ts",
            "TRAIN": "Multivariate_ts/FingerMovements/FingerMovements_TRAIN.ts",
        },
        "HandMovementDirection": {
            "TEST": "Multivariate_ts/HandMovementDirection/HandMovementDirection_TEST.ts",
            "TRAIN": "Multivariate_ts/HandMovementDirection/HandMovementDirection_TRAIN.ts",
        },
        "Handwriting": {
            "TEST": "Multivariate_ts/Handwriting/Handwriting_TEST.ts",
            "TRAIN": "Multivariate_ts/Handwriting/Handwriting_TRAIN.ts",
        },
        "Heartbeat": {
            "TEST": "Multivariate_ts/Heartbeat/Heartbeat_TEST.ts",
            "TRAIN": "Multivariate_ts/Heartbeat/Heartbeat_TRAIN.ts",
        },
        "InsectWingbeat": {
            "TEST": "Multivariate_ts/InsectWingbeat/InsectWingbeat_TEST.ts",
            "TRAIN": "Multivariate_ts/InsectWingbeat/InsectWingbeat_TRAIN.ts",
        },
        "JapaneseVowels": {
            "TEST": "Multivariate_ts/JapaneseVowels/JapaneseVowels_TEST.ts",
            "TRAIN": "Multivariate_ts/JapaneseVowels/JapaneseVowels_TRAIN.ts",
        },
        "LSST": {"TEST": "Multivariate_ts/LSST/LSST_TEST.ts", "TRAIN": "Multivariate_ts/LSST/LSST_TRAIN.ts"},
        "Libras": {"TEST": "Multivariate_ts/Libras/Libras_TEST.ts", "TRAIN": "Multivariate_ts/Libras/Libras_TRAIN.ts"},
        "MotorImagery": {
            "TEST": "Multivariate_ts/MotorImagery/MotorImagery_TEST.ts",
            "TRAIN": "Multivariate_ts/MotorImagery/MotorImagery_TRAIN.ts",
        },
        "NATOPS": {"TEST": "Multivariate_ts/NATOPS/NATOPS_TEST.ts", "TRAIN": "Multivariate_ts/NATOPS/NATOPS_TRAIN.ts"},
        "PEMS-SF": {
            "TEST": "Multivariate_ts/PEMS-SF/PEMS-SF_TEST.ts",
            "TRAIN": "Multivariate_ts/PEMS-SF/PEMS-SF_TRAIN.ts",
        },
        "PenDigits": {
            "TEST": "Multivariate_ts/PenDigits/PenDigits_TEST.ts",
            "TRAIN": "Multivariate_ts/PenDigits/PenDigits_TRAIN.ts",
        },
        "PhonemeSpectra": {
            "TEST": "Multivariate_ts/PhonemeSpectra/PhonemeSpectra_TEST.ts",
            "TRAIN": "Multivariate_ts/PhonemeSpectra/PhonemeSpectra_TRAIN.ts",
        },
        "RacketSports": {
            "TEST": "Multivariate_ts/RacketSports/RacketSports_TEST.ts",
            "TRAIN": "Multivariate_ts/RacketSports/RacketSports_TRAIN.ts",
        },
        "SelfRegulationSCP1": {
            "TEST": "Multivariate_ts/SelfRegulationSCP1/SelfRegulationSCP1_TEST.ts",
            "TRAIN": "Multivariate_ts/SelfRegulationSCP1/SelfRegulationSCP1_TRAIN.ts",
        },
        "SelfRegulationSCP2": {
            "TEST": "Multivariate_ts/SelfRegulationSCP2/SelfRegulationSCP2_TEST.ts",
            "TRAIN": "Multivariate_ts/SelfRegulationSCP2/SelfRegulationSCP2_TRAIN.ts",
        },
        "SpokenArabicDigits": {
            "TEST": "Multivariate_ts/SpokenArabicDigits/SpokenArabicDigits_TEST.ts",
            "TRAIN": "Multivariate_ts/SpokenArabicDigits/SpokenArabicDigits_TRAIN.ts",
        },
        "StandWalkJump": {
            "TEST": "Multivariate_ts/StandWalkJump/StandWalkJump_TEST.ts",
            "TRAIN": "Multivariate_ts/StandWalkJump/StandWalkJump_TRAIN.ts",
        },
        "UWaveGestureLibrary": {
            "TEST": "Multivariate_ts/UWaveGestureLibrary/UWaveGestureLibrary_TEST.ts",
            "TRAIN": "Multivariate_ts/UWaveGestureLibrary/UWaveGestureLibrary_TRAIN.ts",
        },
    }

    def __init__(self, config: Config):
        self.download_and_unzip()


    def download_and_unzip(self):
        logger.info('Downloading UAE Archive for Multivariate TS Classification')
        shell_exec("http://www.timeseriesclassification.com/Downloads/Archives/Multivariate2018_ts.zip")
        shell_exec("unzip -q -n Multivariate2018_ts.zip")

    def get_dataset(self, dataset_name, split="TRAIN", format="TF"):

        raise NotImplementedError

    def get_dataset_as_tensorflow_dataset(self, dataset_name, split="TRAIN"):
        path = self.directories[dataset_name][split]

        x_train, y_train = load_from_tsfile_to_dataframe(path)

        return x_train, y_train



    def get_train_and_val_for_fold(self, fold: int) -> (tf.data.Dataset, tf.data.Dataset):
        raise NotImplementedError

    @staticmethod
    def to_numeric_classes(y_train):
        s = pd.get_dummies(pd.Series(y_train))
        fixed_classes = np.argmax(s.values, axis=1)
        return fixed_classes

    @staticmethod
    def convert_sktime_format_to_array(x_train):
        """
        This should end up with (instances, length, channels)

        """
        arr = from_nested_to_3d_numpy(x_train)
        arr = np.transpose(arr, (0, 2, 1))
        return arr
