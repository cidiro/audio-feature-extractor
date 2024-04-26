import os
import librosa
import numpy as np
import pandas as pd


class MFCCExtractor:
    def __init__(self, folder_path, sr=16000, n_mfcc=20):
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.data = {}

        print(f"Extracting MFCCs from {folder_path}...")

        for file in os.listdir(folder_path):
            if file.endswith(".wav"):
                file_path = os.path.join(folder_path, file)
                self.data[file] = self.extract_mfccs(file_path)

        print("Done!")

    def extract_mfccs(self, file_path):
        try:
            audio, sr = librosa.load(file_path, sr=self.sr)
            mfccs = librosa.feature.mfcc(
                y=audio,
                sr=self.sr,
                n_mfcc=self.n_mfcc
            )
            return mfccs
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return np.zeros((20, 126))

    def mean_to_csv(self, output_file, decimals=4):
        column_names = (
            ["instrument", "source"]
            + [f"mean_mfcc_{i + 1}" for i in range(self.n_mfcc)]
            + ["filename"]
        )

        data = []
        for file, mfccs in self.data.items():
            mean_mfccs = np.mean(mfccs, axis=1)
            data.append(
                [file.split("_")[0], file.split("_")[1]]
                + mean_mfccs.tolist()
                + [file]
            )

        df = pd.DataFrame(data, columns=column_names)
        df = df.round(decimals=decimals)
        df.to_csv(output_file, index=False)

        print(f"Mean MFCCs saved to {output_file}")

    def std_to_csv(self, output_file, decimals=4):
        column_names = (
            ["instrument", "source"]
            + [f"std_mfcc_{i + 1}" for i in range(self.n_mfcc)]
            + ["filename"]
        )

        data = []
        for file, mfccs in self.data.items():
            std_mfccs = np.std(mfccs, axis=1)
            data.append(
                [file.split("_")[0], file.split("_")[1]]
                + std_mfccs.tolist()
                + [file]
            )

        df = pd.DataFrame(data, columns=column_names)
        df = df.round(decimals=decimals)
        df.to_csv(output_file, index=False)

        print(f"Standard deviation of MFCCs saved to {output_file}")
