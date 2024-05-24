import os
import typing as tp

import h5py
import numpy as np
import pandas as pd
from mne import create_info
from mne.io.array import RawArray

from . import api, utils
from .textgrids import TextGrid


class Lebel2023Recording(api.Recording):
    data_url = "https://github.com/OpenNeuroDatasets/ds003020.git"
    paper_url = "https://www.nature.com/articles/s41597-023-02437-z"
    doi = "https://doi.org/10.1038/s41597-023-02437-z"
    licence = "CC0"
    modality = "audio"
    language = "english"
    device = "fmri"
    description = "fMRI of 8 subjects listening to stories."

    @classmethod
    def download(cls):
        path = utils.StudyPaths(Lebel2023Recording.study_name())
        command = f"datalad install {cls.data_url} {path.download}"
        print(f"Running shell command `{command}`")
        os.system(command)

    @classmethod
    def iter(cls) -> tp.Iterator["Lebel2023Recording"]:  # type: ignore
        """Returns a generator of all recordings"""
        path = utils.StudyPaths(Lebel2023Recording.study_name())
        preproc_path = path.download / "derivative" / "preprocessed_data"
        subjects = os.listdir(preproc_path)
        assert len(subjects) == 8

        for subject in subjects:
            hf5_files = os.listdir(preproc_path / subject)
            if len(hf5_files) == 84:
                for f in hf5_files:
                    recording = cls(
                        subject_uid=subject,
                        recording_uid=f.replace(".hf5", ""),
                    )
                    yield recording

    def __init__(self, subject_uid: str, recording_uid: str) -> None:
        super().__init__(subject_uid=subject_uid, recording_uid=recording_uid)
        self.tr = 2
        self.path = utils.StudyPaths(Lebel2023Recording.study_name())
        self.preproc_path = self.path.download / "derivative" / "preprocessed_data"
        self.img_path = (
            self.preproc_path / self.subject_uid / (self.recording_uid + ".hf5")
        )
        self.event_path = (
            self.path.download
            / "derivative"
            / "TextGrids"
            / (self.recording_uid + ".TextGrid")
        )

    def _load_raw(self):
        data = h5py.File(self.img_path, "r")["data"][...].T
        return RawArray(
            data,
            create_info(
                ch_names=data.shape[0],
                sfreq=1 / self.tr,
                ch_types="eeg",
                verbose=False,
            ),
            verbose=False,
        )

    def _load_events(self) -> pd.DataFrame:
        df = TextGrid(self.event_path).get_transcript()
        # blocks = pd.DataFrame(
        #     np.arange(
        #         df.start.min() // self.tr * self.tr,
        #         df.start.max() // self.tr * self.tr + self.tr,
        #         self.tr,
        #     ),
        #     columns=["start"],
        # )
        # blocks["stop"] = blocks.start + self.tr
        # blocks[["kind", "duration"]] = "block", self.tr
        # blocks["uid"] = blocks.index
        # block = pd.DataFrame(
        #     {
        #         "start": [df.start.min()],
        #         "stop": [df.stop.max()],
        #         "duration": [df.stop.max() - df.start.min()],
        #         "kind": ["block"],
        #         "uid": [self.recording_uid],
        #     }
        # )
        # df = pd.concat([blocks, df], ignore_index=True)
        filepath = self.path.download / "stimuli" / f"{self.recording_uid}.wav"
        df[["filepath", "language", "modality"]] = (
            str(filepath),
            "en",
            "audio",
        )
        df.event.validate()
        return df.sort_values("start")
