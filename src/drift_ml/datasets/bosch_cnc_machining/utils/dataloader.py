import os
import json
from random import sample
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import stft
from tqdm.contrib.concurrent import process_map
from sklearn.model_selection import train_test_split
import pickle as pkl

from drift_ml.utils.utils import in_notebook, find_all_h5s_in_dir

if in_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


class BoschCNCDataloader:
    def __init__(
        self, metadata_path="", window_length=4096, random_seed=42,
    ):
        self.random_seed = random_seed
        self.metadata_path = metadata_path

        self.window_length = window_length

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

    @property
    def X_train(self):
        return self.sample_data_X[self.train_sample_ids]

    @property
    def y_train(self):
        return self.sample_data_y[self.train_sample_ids]

    @property
    def X_val(self):
        return self.sample_data_X[self.val_sample_ids]

    @property
    def y_val(self):
        return self.sample_data_y[self.val_sample_ids]

    @property
    def X_test(self):
        return self.sample_data_X[self.test_sample_ids]

    @property
    def y_test(self):
        return self.sample_data_y[self.test_sample_ids]

    @staticmethod
    def _sample_stft(sample, fs=2000):
        specs = []
        for c in range(sample.shape[1]):
            _, _, Zxx = stft(sample[:, c], fs=fs)
            specs.append(np.abs(Zxx))
        specs = np.stack(specs, axis=0)
        return specs

    def get_windowed_samples_as_stft_dataloader(
        self, transform_fn=(lambda x: x), copy_sets=True
    ):
        stft = np.zeros((self.sample_data_X.shape[0], 3, 129, 33))
        # for i in tqdm(range(self.sample_data_X.shape[0])):
        #     stft[i] = BoschCNCDataloader._sample_stft(self.sample_data_X[i])
        stft = np.array(
            process_map(
                BoschCNCDataloader._sample_stft,
                transform_fn(self.sample_data_X),
                chunksize=10,
                max_workers=12,
            )
        )

        stft_dataloader = STFTBoschCNCDataloader(
            metadata_path=self.metadata_path,
            window_length=self.window_length,
            random_seed=self.random_seed,
        )
        stft_dataloader.sample_data_X = stft
        stft_dataloader.sample_data_y = self.sample_data_y

        if copy_sets:
            stft_dataloader.train_part_ids = self.train_part_ids
            stft_dataloader.val_part_ids = self.val_part_ids
            stft_dataloader.test_part_ids = self.test_part_ids
            stft_dataloader.train_sample_ids = self.train_sample_ids
            stft_dataloader.val_sample_ids = self.val_sample_ids
            stft_dataloader.test_sample_ids = self.test_sample_ids

        return stft_dataloader

    def save_windowed_samples_as_stft(self, filename="cnc_dataset_stft_data.h5"):
        hf = h5py.File(filename, "w")
        dataset = hf.create_dataset("stft", (self.sample_data_X.shape[0], 3, 129, 33))
        for i in tqdm(range(self.sample_data_X.shape[0])):
            dataset[i] = BoschCNCDataloader._sample_stft(self.sample_data_X[i])
        hf.close()

    def load_metadata(self,):
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

    def _generate_mask(self, periods, machines, processes):
        # Filter general dataset with global filters
        mask = np.ones_like(self.metadata["part_id_label"]).astype(np.bool)
        if periods is not None:
            for period_id, period in enumerate(self.periods):
                if period not in periods:
                    mask[self.metadata["part_id_period"] == period_id] = False

        if processes is not None:
            for op_id, op in enumerate(self.processes):
                if op not in processes:
                    mask[self.metadata["part_id_process"] == op_id] = False

        if machines is not None:
            for machine_id, machine in enumerate(self.machines):
                if machine not in machines:
                    mask[self.metadata["part_id_machine"] == machine_id] = False

        return mask

    def _update_sample_ids(self):
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

    def generate_datasets_by_size(
        self,
        periods=None,
        machines=None,
        processes=None,
        train_size=0.3,
        val_size=0.2,
        test_size=0.5,
    ):
        assert np.sum([train_size, val_size, test_size]) == 1.0
        mask = self._generate_mask(periods, machines, processes)

        part_ids = np.nonzero(mask)[0]
        train_val_part_ids, self.test_part_ids = train_test_split(
            part_ids,
            train_size=(train_size + val_size),
            stratify=self.metadata["part_id_label"][part_ids],
            random_state=self.random_seed,
        )
        self.train_part_ids, self.val_part_ids = train_test_split(
            train_val_part_ids,
            train_size=(train_size / (train_size + val_size)),
            stratify=self.metadata["part_id_label"][train_val_part_ids],
            random_state=self.random_seed,
        )
        self._update_sample_ids()

    def generate_datasets_by_train_filter(
        self,
        periods=None,
        machines=None,
        processes=None,
        train_periods=None,
        train_machines=None,
        train_processes=None,
        train_val_split=0.5,
    ):
        """Generate train/val/test datasets by setting filter criterias

        At least one of the train filters has to be set, otherwise
        there will be no test set generated.

        Args:
            periods list[string]: Defaults to None.
            machines list[string]: Defaults to None.
            processes list[string]: Defaults to None.
            train_periods list[string]: Defaults to None.
            train_machines list[string]: Defaults to None.
            train_ops list[string]: Defaults to None.
            train_val_split float: Defaults to 0.5.

        Raises:
            ValueError: _description_
        """
        if not np.any(
            np.array([train_periods, train_machines, train_processes], dtype=object)
        ):
            raise ValueError("A minimum of one train filter criterium has to be given.")

        # Filter general dataset with global filters
        global_mask = self._generate_mask(periods, machines, processes)

        # Filter training set with training filters
        train_mask = np.copy(global_mask)
        if train_periods is not None:
            for period_id, period in enumerate(self.periods):
                if period not in train_periods:
                    train_mask[self.metadata["part_id_period"] == period_id] = False

        if train_processes is not None:
            for op_id, op in enumerate(self.processes):
                if op not in train_processes:
                    train_mask[self.metadata["part_id_process"] == op_id] = False

        if train_machines is not None:
            for machine_id, machine in enumerate(self.machines):
                if machine not in train_machines:
                    train_mask[self.metadata["part_id_machine"] == machine_id] = False

        train_val_part_ids = np.nonzero(train_mask)[0]
        self.train_part_ids, self.val_part_ids = train_test_split(
            train_val_part_ids,
            train_size=train_val_split,
            stratify=self.metadata["part_id_label"][train_mask],
            random_state=self.random_seed,
        )

        self.test_part_ids = np.nonzero(
            np.logical_and(global_mask, np.logical_not(train_mask))
        )[0]

        self._update_sample_ids()


class SimpleTSFreshBoschCNCDataloader(BoschCNCDataloader):
    def __init__(
        self, metadata_path, window_length=4096, random_seed=42,
    ):
        super().__init__(
            window_length=window_length,
            metadata_path=metadata_path,
            random_seed=random_seed,
        )
        self.load_metadata()

    def load_data(self, tsfresh_features_path, sample_data_y_path):
        self.sample_data_X = pd.read_pickle(tsfresh_features_path)
        self.sample_data_y = np.load(sample_data_y_path)


class STFTBoschCNCDataloader(BoschCNCDataloader):
    def __init__(
        self, metadata_path, window_length=4096, random_seed=42,
    ):
        super().__init__(
            window_length=window_length,
            metadata_path=metadata_path,
            random_seed=random_seed,
        )
        self.load_metadata()

    def load_data(self, data_h5_path):
        with h5py.File(data_h5_path, "r") as f:
            self.sample_data_X = f["stft"][:]
            self.sample_data_y = f["sample_data_y"][:]


class NPYBoschCNCDataLoader(BoschCNCDataloader):
    def __init__(
        self, metadata_path, window_length=4096, random_seed=42,
    ):
        super().__init__(
            window_length=window_length,
            metadata_path=metadata_path,
            random_seed=random_seed,
        )
        self.load_metadata()

    def load_data(
        self, sample_data_x_path, sample_data_y_path,
    ):
        self.sample_data_X = np.load(sample_data_x_path)
        self.sample_data_y = np.load(sample_data_y_path)


class RawBoschCNCDataloader(BoschCNCDataloader):
    def __init__(
        self, window_length=4096, random_seed=42,
    ):
        super().__init__(
            window_length=window_length, random_seed=random_seed,
        )

        self.sample_data_X = np.empty((0, self.window_length, 3), float)
        self.sample_data_y = np.empty((0), bool)

    def save_windowed_samples_as_npy(
        self, sample_data_folder_path,
    ):
        np.save(
            os.path.join(
                sample_data_folder_path,
                "sample_data_x_raw_ws" + str(self.window_length) + ".npy",
            ),
            self.sample_data_X,
        )
        np.save(
            os.path.join(
                sample_data_folder_path,
                "sample_data_y_raw_ws" + str(self.window_length) + ".npy",
            ),
            self.sample_data_y,
        )

    def save_metadata_to_file(
        self, metadata_path,
    ):
        with open(os.path.join(metadata_path,), "wb",) as f:
            pkl.dump(self.metadata, f)
        self.metadata_path = metadata_path

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

                            n_samples = vibration_data.shape[0] // self.window_length

                            keep_samples = n_samples * self.window_length
                            vibration_data = vibration_data[:keep_samples].reshape(
                                [-1, self.window_length, 3]
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
