import os
import typing as tp
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from tqdm import tqdm

from ..events import extract_sequence_info
from . import api, utils

SFREQ = 500.0


def _read_meta(fname):
    proc = loadmat(
        fname,
        squeeze_me=True,
        chars_as_strings=True,
        struct_as_record=True,
        simplify_cells=True,
    )["proc"]

    # ref = proc["implicitref"]
    # ref_channels = proc["refchannels"]

    # subject_id = proc["subject"]
    meta = proc["trl"]

    # TODO artefacts, ica, rejected components etc
    assert len(meta) == proc["tot_trials"]
    assert proc["tot_chans"] == 61
    bads = list(proc["impedence"]["bads"])
    bads += list(proc["rejections"]["badchans"])
    # proc['varnames'], ['segment', 'tmin', 'Order']

    # summary = proc["rejections"]["final"]["artfctdef"]["summary"]
    # bad_segments = summary["artifact"]

    #     meta = pd.DataFrame(meta[:, 0].astype(int), columns=['start'])

    #     meta['start_offset'] = meta[:, 1].astype(int) # wave?
    #     meta['wav_file'] = meta[:, 3].astype(int)
    #     meta['start_sec'] = meta[:, 4]
    #     meta['mat_index'] = meta[:, 5].astype(int)
    columns = list(proc["varnames"])
    if len(columns) != meta.shape[1]:
        columns = ["start_sample", "stop_sample", "offset"] + columns
        assert len(columns) == meta.shape[1]
    meta = pd.DataFrame(meta, columns=["_" + i for i in columns])
    assert len(meta) == 2129  # FIXME retrieve subjects who have less trials?

    # Add Brennan's annotations
    paths = get_paths()
    story = pd.read_csv(paths.download / "AliceChapterOne-EEG.csv")
    events = meta.join(story)

    events["kind"] = "word"
    events["condition"] = "sentence"
    events["duration"] = events.offset - events.onset
    columns = dict(Word="word", Position="word_id", Sentence="sequence_id")
    events = events.rename(columns=columns)
    events["start"] = events["_start_sample"] / SFREQ

    # add audio events
    wav_file = paths.download / "audio" / "DownTheRabbitHoleFinal_SoundFile%i.wav"
    sounds = []
    for segment, d in events.groupby("Segment"):
        # Some wav files start BEFORE the onset of eeg recording...
        start = d.iloc[0].start - d.iloc[0].onset
        sound = dict(kind="sound", start=start, filepath=str(wav_file) % segment)
        sounds.append(sound)
    events = pd.concat([events, pd.DataFrame(sounds)], ignore_index=True)
    events = events.sort_values("start").reset_index()

    # clean up
    keep = [
        "start",
        "duration",
        "kind",
        "word",
        "word_id",
        "sequence_id",
        "condition",
        "filepath",
    ]
    events = events[keep]
    events[["language", "modality"]] = "english", "audio"
    events = extract_sequence_info(events)
    events = events.event.create_blocks(groupby="sentence")
    events = events.event.validate()

    return events


def _read_eeg(fname):
    fname = Path(fname)
    assert fname.exists()
    assert str(fname).endswith(".mat")
    mat = loadmat(
        fname,
        squeeze_me=True,
        chars_as_strings=True,
        struct_as_record=True,
        simplify_cells=True,
    )
    mat = mat["raw"]

    # sampling frequency
    sfreq = mat["hdr"]["Fs"]
    assert sfreq == 500.0
    assert mat["fsample"] == sfreq

    # channels
    n_chans = mat["hdr"]["nChans"]
    n_samples = mat["hdr"]["nSamples"]
    ch_names = list(mat["hdr"]["label"])
    assert len(ch_names) == n_chans

    # vertical EOG
    assert ch_names[60] == "VEOG"

    # audio channel
    add_audio_chan = False
    if len(ch_names) == 61:
        ch_names += ["AUD"]
        add_audio_chan = True
    assert ch_names[61] in ("AUD", "Aux5")

    # check name
    for i, ch in enumerate(ch_names[:-2]):
        assert ch == str(i + 1 + (i >= 28))

    # channel type
    assert set(mat["hdr"]["chantype"]) == set(["eeg"])
    ch_types = ["eeg"] * 60 + ["eog", "misc"]
    assert set(mat["hdr"]["chanunit"]) == set(["uV"])

    # create MNE info
    info = mne.create_info(ch_names, sfreq, ch_types, verbose=False)
    subject_id = fname.name.split(".mat")[0]
    info["subject_info"] = dict(his_id=subject_id, id=int(subject_id[1:]))

    # time
    diff = np.diff(mat["time"]) - 1 / sfreq
    tol = 1e-5
    assert np.all(diff < tol)
    assert np.all(diff > -tol)
    start, stop = mat["sampleinfo"]
    assert start == 1
    assert stop == n_samples
    assert mat["hdr"]["nSamplesPre"] == 0
    assert mat["hdr"]["nTrials"] == 1

    # data
    data = mat["trial"]
    assert data.shape[0] == n_chans
    assert data.shape[1] == n_samples
    if add_audio_chan:
        data = np.vstack((data, np.zeros_like(data[0])))

    # create mne objects
    info = mne.create_info(ch_names, sfreq, ch_types, verbose=False)
    raw = mne.io.RawArray(data * 1e-6, info, verbose=False)
    montage = mne.channels.make_standard_montage("easycap-M10")
    raw.set_montage(montage)

    assert raw.info["sfreq"] == SFREQ
    assert len(raw.ch_names) == 62

    return raw


class Lebel2023Recording(api.Recording):

    data_url = "https://github.com/OpenNeuroDatasets/ds003020.git"
    paper_url = "https://www.nature.com/articles/s41597-023-02437-z"
    doi = "https://doi.org/10.1038/s41597-023-02437-z"
    licence = "CC0"
    modality = "audio"
    language = "english"
    device = "fmri"

    def sync(cls, path):
        os.system(f"datalad install {data_url} {path}")

    @classmethod
    def iter(cls) -> tp.Iterator["Lebel2023Recording"]:  # type: ignore
        """Returns a generator of all recordings"""
        path = utils.StudyPaths(Lebel2023Recording.study_name())
        path.folder.mkdir(exist_ok=True, parents=True)
        cls.sync(path.download)

        subjects = os.listdir(path.download / "derivative" / "preprocessed_data")
        assert len(subjects) == 7

        for subject in subjects:
            recording = cls(subject_uid=str(subject))
            yield recording

    def __init__(self, subject_uid: str) -> None:
        super().__init__(subject_uid=subject_uid, recording_uid=subject_uid)

    def _load_raw(self) -> mne.io.RawArray:
        paths = get_paths()
        raw = _read_eeg(paths.download / f"{self.subject_uid}.mat")
        return raw

    def _load_events(self) -> pd.DataFrame:
        file = get_paths().download / "proc" / f"{self.subject_uid}.mat"
        events = _read_meta(file)
        return events
