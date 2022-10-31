import os
import json
from random import sample
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import stft
from sklearn.model_selection import train_test_split
import pickle as pkl


def in_notebook():
    try:
        from IPython import get_ipython

        if "IPKernelApp" not in get_ipython().config:  # pragma: no cover
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True


if in_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


def find_all_h5s_in_dir(s_dir):
    """
    list all .h5 files in a directory
    """

    fileslist = []
    for root, dirs, files in os.walk(s_dir):
        for file in files:
            if file.endswith(".h5"):
                fileslist.append(file)
    return fileslist


class BoschCNCDataloader:
    def __init__(
        self,
        dataset_config_path,
        metadata_path,
        random_seed=42,
    ):
        self.random_seed = random_seed
        self.metadata_path = metadata_path

        with open(dataset_config_path) as f:
            self.dataset_config = json.load(f)

        self.machines = ["M01", "M02", "M03"]
        self.processes = [
            "OP00",
            "OP01",
            "OP02",
            "OP03",
            "OP04",
            "OP05",
            "OP06",
            "OP07",
            "OP08",
            "OP09",
            "OP10",
            "OP11",
            "OP12",
            "OP13",
            "OP14",
        ]
        self.periods = [
            "Feb_2019",
            "Aug_2019",
            "Feb_2020",
            "Aug_2020",
            "Feb_2021",
            "Aug_2021",
        ]

        self.sample_data_X = False
        self.sample_data_y = False

        self.train_part_ids = False
        self.val_part_ids = False
        self.test_part_ids = False

        self.train_sample_ids = False
        self.val_sample_ids = False
        self.test_sample_ids = False

        self.metadata = {
            "part_id": np.array([]),
            "part_id_machine": np.array([]),
            "part_id_process": np.array([]),
            "part_id_period": np.array([]),
            "part_id_label": np.array([]),
            "part_id_path": np.array([]),
            "part_id_samples": [],
        }

    def load_metadata(
        self,
    ):
        with open(self.metadata_path, "rb") as f:
            self.metadata = pkl.load(f)

    def plot_stats(self):
        fig, axs = plt.subplots(nrows=2, ncols=3, squeeze=True, figsize=(15, 4))

        train_n = []
        val_n = []
        test_n = []
        for machine_id in range(len(self.machines)):
            train_n.append(
                (
                    self.metadata["part_id_machine"][self.train_part_ids] == machine_id
                ).sum()
            )
            val_n.append(
                (
                    self.metadata["part_id_machine"][self.val_part_ids] == machine_id
                ).sum()
            )
            test_n.append(
                (
                    self.metadata["part_id_machine"][self.test_part_ids] == machine_id
                ).sum()
            )
        train_n = np.array(train_n)
        val_n = np.array(val_n)
        test_n = np.array(test_n)
        axs[0, 0].bar(self.machines, train_n, label="train")
        axs[0, 0].bar(self.machines, val_n, bottom=train_n, label="val")
        axs[0, 0].bar(self.machines, test_n, bottom=train_n + val_n, label="test")
        axs[0, 0].legend()

        train_n = []
        val_n = []
        test_n = []
        for process_id in range(len(self.processes)):
            train_n.append(
                (
                    self.metadata["part_id_process"][self.train_part_ids] == process_id
                ).sum()
            )
            val_n.append(
                (
                    self.metadata["part_id_process"][self.val_part_ids] == process_id
                ).sum()
            )
            test_n.append(
                (
                    self.metadata["part_id_process"][self.test_part_ids] == process_id
                ).sum()
            )
        train_n = np.array(train_n)
        val_n = np.array(val_n)
        test_n = np.array(test_n)
        axs[0, 1].bar(self.processes, train_n, label="train")
        axs[0, 1].bar(self.processes, val_n, bottom=train_n, label="val")
        axs[0, 1].bar(self.processes, test_n, bottom=train_n + val_n, label="test")
        axs[0, 1].legend()

        train_n = []
        val_n = []
        test_n = []
        for period_id in range(len(self.periods)):
            train_n.append(
                (
                    self.metadata["part_id_period"][self.train_part_ids] == period_id
                ).sum()
            )
            val_n.append(
                (self.metadata["part_id_period"][self.val_part_ids] == period_id).sum()
            )
            test_n.append(
                (self.metadata["part_id_period"][self.test_part_ids] == period_id).sum()
            )
        train_n = np.array(train_n)
        val_n = np.array(val_n)
        test_n = np.array(test_n)
        axs[0, 2].bar(self.periods, train_n, label="train")
        axs[0, 2].bar(self.periods, val_n, bottom=train_n, label="val")
        axs[0, 2].bar(self.periods, test_n, bottom=train_n + val_n, label="test")
        axs[0, 2].legend()

        train_n = []
        val_n = []
        test_n = []
        for label in range(2):
            train_n.append(
                (self.metadata["part_id_label"][self.train_part_ids] == label).sum()
            )
            val_n.append(
                (self.metadata["part_id_label"][self.val_part_ids] == label).sum()
            )
            test_n.append(
                (self.metadata["part_id_label"][self.test_part_ids] == label).sum()
            )
        train_n = np.array(train_n)
        val_n = np.array(val_n)
        test_n = np.array(test_n)

        axs[1, 0].bar(["NOK"], train_n[0], label="train")
        axs[1, 0].bar(["NOK"], val_n[0], bottom=train_n[0], label="val")
        axs[1, 0].bar(["NOK"], test_n[0], bottom=val_n[0] + train_n[0], label="test")
        axs[1, 0].legend()

        axs[1, 1].bar(["OK"], train_n[1], label="train")
        axs[1, 1].bar(["OK"], val_n[1], bottom=train_n[1], label="val")
        axs[1, 1].bar(["OK"], test_n[1], bottom=val_n[1] + train_n[1], label="test")
        axs[1, 1].legend()

        plt.show()

    @property
    def n_parts(self):
        return self.metadata["part_id_label"].shape[0]

    def period_sample_ids(self, period_ids):
        part_mask = np.ones_like(self.metadata["part_id_label"]).astype(np.bool)
        for period_id in range(len(self.periods)):
            if period_id not in period_ids:
                part_mask[self.metadata["part_id_period"] == period_id] = False

        sample_ids = []
        for part_id in np.nonzero(part_mask)[0]:
            sample_ids.extend(self.metadata["part_id_samples"][part_id][0].tolist())
        return sample_ids

    def generate_datasets_by_period(
        self, train_periods=["Feb_2019", "Aug_2019", "Feb_2020"], train_split=0.5
    ):

        train_mask = np.ones_like(self.metadata["part_id_label"]).astype(np.bool)
        for period_id, period in enumerate(self.periods):
            if period not in train_periods:
                train_mask[self.metadata["part_id_period"] == period_id] = False

        train_val_part_ids = np.nonzero(train_mask)[0]
        self.train_part_ids, self.val_part_ids = train_test_split(
            train_val_part_ids,
            train_size=train_split,
            stratify=self.metadata["part_id_label"][train_mask],
            random_state=self.random_seed,
        )

        self.test_part_ids = np.nonzero(np.logical_not(train_mask))[0]

        self.train_sample_ids = []
        for part_id in self.train_part_ids:
            self.train_sample_ids.extend(
                self.metadata["part_id_samples"][part_id][0].tolist()
            )

        self.test_sample_ids = []
        for part_id in self.test_part_ids:
            self.test_sample_ids.extend(
                self.metadata["part_id_samples"][part_id][0].tolist()
            )

        self.val_sample_ids = []
        for part_id in self.val_part_ids:
            self.val_sample_ids.extend(
                self.metadata["part_id_samples"][part_id][0].tolist()
            )


class TSFreshBoschCNCDataloader(BoschCNCDataloader):
    def __init__(
        self,
        dataset_config_path,
        metadata_path,
        random_seed=42,
    ):
        super().__init__(
            dataset_config_path=dataset_config_path,
            metadata_path=metadata_path,
            random_seed=random_seed,
        )

    def load_data(self, tsfresh_features_path, sample_data_y_path):
        self.sample_data_X = pd.read_pickle(tsfresh_features_path)
        self.sample_data_y = np.load(sample_data_y_path)


class STFTBoschCNCDataloader(BoschCNCDataloader):
    def __init__(
        self,
        dataset_config_path,
        metadata_path,
        random_seed=42,
    ):
        super().__init__(
            dataset_config_path=dataset_config_path,
            metadata_path=metadata_path,
            random_seed=random_seed,
        )

    def load_data(self, data_h5_path):
        with h5py.File(data_h5_path, "r") as f:
            self.sample_data_X = f["stft"][:]
            self.sample_data_y = f["data_y"][:]

        self.train_part_ids = False
        self.val_part_ids = False
        self.test_part_ids = False


class RawBoschCNCDataloader(BoschCNCDataloader):
    def __init__(
        self,
        dataset_config_path,
        metadata_path="",
        random_seed=42,
    ):
        super().__init__(
            dataset_config_path=dataset_config_path,
            metadata_path=metadata_path,
            random_seed=random_seed,
        )

        self.sample_data_X = np.empty(
            (0, self.dataset_config["window_length"], 3), float
        )
        self.sample_data_y = np.empty((0), bool)

    def create_stft_dataset(self, filename="cnc_dataset_stft_data.h5"):
        def preprocess(sample, fs=2000):
            specs = []
            for c in range(sample.shape[1]):
                _, _, Zxx = stft(sample[:, c], fs=fs)
                specs.append(np.abs(Zxx))
            specs = np.stack(specs, axis=0)
            return specs

        min_max_list = []
        for inds in self.metadata["part_id_samples"]:
            min_max_list.append([inds[0].min(), inds[0].max() + 1])

        min_max = np.array(min_max_list)

        hf = h5py.File(filename, "w")
        hf.create_dataset("sample_data_y", data=self.sample_data_y)
        hf.create_dataset("part_id_machine", data=self.metadata["part_id_machine"])
        hf.create_dataset("part_id_process", data=self.metadata["part_id_process"])
        hf.create_dataset("part_id_period", data=self.metadata["part_id_period"])
        hf.create_dataset("part_id", data=self.metadata["part_id"].astype(np.int32))
        hf.create_dataset("part_id_samples", data=min_max)
        hf.create_dataset("part_id_label", data=self.metadata["part_id_label"])
        dataset = hf.create_dataset("stft", (self.sample_data_X.shape[0], 3, 129, 33))
        for i in tqdm(range(self.sample_data_X.shape[0])):
            dataset[i] = preprocess(self.sample_data_X[i])

        hf.close()

    def load_raw_npy_data(
        self,
        sample_data_x_path,
        sample_data_y_path,
    ):
        self.sample_data_X = np.load(sample_data_x_path)
        self.sample_data_y = np.load(sample_data_y_path)

    def save_raw_data_to_files(
        self,
        sample_data_folder_path,
    ):
        np.save(
            os.path.join(
                sample_data_folder_path,
                "sample_data_x_raw_ws"
                + str(self.dataset_config["window_length"])
                + ".npy",
            ),
            self.sample_data_X,
        )
        np.save(
            os.path.join(
                sample_data_folder_path,
                "sample_data_y_raw_ws"
                + str(self.dataset_config["window_length"])
                + ".npy",
            ),
            self.sample_data_y,
        )

    def save_metadata_to_file(
        self,
        metadata_folder_path,
    ):
        with open(
            os.path.join(
                metadata_folder_path,
                "metadata_ws" + str(self.dataset_config["window_length"]) + ".pkl",
            ),
            "wb",
        ) as f:
            pkl.dump(self.metadata, f)

    def load_raw_h5_data(self, raw_h5_data_path):
        sample_count = 0
        with tqdm(
            total=2 * len(self.machines) * len(self.processes),
            desc="Loading dataset files",
        ) as pbar:
            for m_id, machine in enumerate(self.machines):
                for p_id, process in enumerate(self.processes):
                    machine_X_data = []
                    for label in ["good", "bad"]:
                        pbar.update(1)
                        data_path = os.path.join(
                            raw_h5_data_path, machine, process, label
                        )
                        list_paths = find_all_h5s_in_dir(data_path)
                        list_paths.sort()

                        for file_path in list_paths:
                            with h5py.File(
                                os.path.join(data_path, file_path), "r"
                            ) as f:
                                vibration_data = f["vibration_data"][:]

                            n_samples = (
                                vibration_data.shape[0]
                                // self.dataset_config["window_length"]
                            )

                            keep_samples = (
                                n_samples * self.dataset_config["window_length"]
                            )
                            vibration_data = vibration_data[:keep_samples].reshape(
                                [-1, self.dataset_config["window_length"], 3]
                            )

                            if label == "good":
                                labels = np.zeros((n_samples))
                                self.metadata["part_id_label"] = np.concatenate(
                                    [self.metadata["part_id_label"], [True]]
                                )
                            else:
                                labels = np.ones((n_samples))
                                self.metadata["part_id_label"] = np.concatenate(
                                    [self.metadata["part_id_label"], [False]]
                                )

                            machine_X_data.append(vibration_data)
                            self.sample_data_y = np.concatenate(
                                [self.sample_data_y, labels]
                            )

                            self.metadata["part_id_machine"] = np.concatenate(
                                [self.metadata["part_id_machine"], [m_id]]
                            )
                            self.metadata["part_id_process"] = np.concatenate(
                                [self.metadata["part_id_process"], [p_id]]
                            )
                            period = "_".join(file_path.split("_")[1:3])
                            self.metadata["part_id_period"] = np.concatenate(
                                [
                                    self.metadata["part_id_period"],
                                    [self.periods.index(period)],
                                ]
                            )
                            part_id = file_path.split("_")[-1].split(".")[0]
                            self.metadata["part_id"] = np.concatenate(
                                [self.metadata["part_id"], [part_id]]
                            )

                            self.metadata["part_id_samples"].append(
                                [
                                    np.arange(
                                        sample_count,
                                        sample_count + vibration_data.shape[0],
                                    )
                                ],
                            )
                            sample_count += vibration_data.shape[0]

                    if len(machine_X_data) > 0:
                        machine_X_data = np.concatenate(machine_X_data, axis=0)

                        if self.sample_data_X is False:
                            self.sample_data_X = machine_X_data
                        else:
                            self.sample_data_X = np.concatenate(
                                (self.sample_data_X, machine_X_data), axis=0
                            )


if __name__ == "__main__":
    loader = RawBoschCNCDataloader()
    loader.plot_stats()
