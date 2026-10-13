from pathlib import Path

import numpy as np
from torch.utils.data import Subset


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def split_dataset_by_video(dataset, val_fraction=0.2, seed=2021):
    """Split a frame-window dataset by video, using training videos only.

    Splitting individual windows would put heavily overlapping windows from the
    same video in both subsets.  A video-level split prevents that leakage and
    keeps validation independent of the official test split and its labels.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")

    video_names = list(dataset.videos)
    if len(video_names) < 2:
        raise ValueError("At least two training videos are required for a validation split")

    rng = np.random.RandomState(seed)
    shuffled = list(video_names)
    rng.shuffle(shuffled)
    val_count = min(len(shuffled) - 1, max(1, int(round(len(shuffled) * val_fraction))))
    val_videos = set(shuffled[:val_count])
    train_videos = set(shuffled[val_count:])

    train_indices = []
    val_indices = []
    for index, (video_name, _) in enumerate(dataset.samples):
        if video_name in val_videos:
            val_indices.append(index)
        else:
            train_indices.append(index)

    if not train_indices or not val_indices:
        raise ValueError("The video-level split produced an empty subset")

    return (
        Subset(dataset, train_indices),
        Subset(dataset, val_indices),
        sorted(train_videos),
        sorted(val_videos),
    )


def _contains_video_frames(path):
    path = Path(path)
    if not path.is_dir():
        return False
    return any(
        child.is_dir() and any(frame.suffix.lower() in IMAGE_EXTENSIONS for frame in child.iterdir())
        for child in path.iterdir()
        if not child.name.lower().endswith('_gt')
    )


def resolve_split(dataset_path, dataset_type, split):
    """Resolve both the repository layout and the original UCSD Ped2 layout."""
    split = split.lower()
    if split not in {'training', 'testing'}:
        raise ValueError("split must be 'training' or 'testing'")

    candidates = []
    if dataset_path:
        root = Path(dataset_path).expanduser()
        candidates.extend([
            root,
            root / dataset_type / split / 'frames',
            root / dataset_type / ('Train' if split == 'training' else 'Test'),
            root / ('Train' if split == 'training' else 'Test'),
        ])

    project_root = Path(__file__).resolve().parent
    candidates.extend([
        project_root / 'data' / 'datasets' / dataset_type / split / 'frames',
        Path(r'E:\dataset\UCSD\UCSDped2') / ('Train' if split == 'training' else 'Test'),
    ])

    for candidate in candidates:
        if _contains_video_frames(candidate):
            return candidate.resolve()

    checked = '\n  - '.join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Could not find {} frames for {}. Checked:\n  - {}".format(split, dataset_type, checked)
    )


def load_ped2_labels(test_folder, label_path=None, time_step=4):
    """Return anomaly labels aligned with the predicted (future) frames."""
    test_folder = Path(test_folder)
    candidates = []
    if label_path:
        candidates.append(Path(label_path).expanduser())
    candidates.extend([
        Path(__file__).resolve().parent / 'data' / 'frame_labels_ped2.npy',
        test_folder.parent / 'label' / 'test.csv',
    ])

    npy_path = next((path for path in candidates if path.suffix.lower() == '.npy' and path.is_file()), None)
    if npy_path:
        labels = np.asarray(np.load(str(npy_path))).reshape(-1)
        expected = sum(max(0, len(list_frames(video)) - time_step) for video in list_videos(test_folder))
        if labels.size == expected:
            # Historical files in this project use 1=normal, 0=anomaly.
            return 1 - labels.astype(np.int64)

    csv_path = next((path for path in candidates if path.suffix.lower() == '.csv' and path.is_file()), None)
    if csv_path:
        rows = {}
        with csv_path.open('r', encoding='utf-8') as handle:
            for line in handle:
                fields = line.strip().split()
                if fields:
                    rows[Path(fields[0]).name] = np.asarray(fields[1:], dtype=np.int64)
        aligned = []
        for video in list_videos(test_folder):
            if video.name not in rows:
                raise ValueError("No labels found for {} in {}".format(video.name, csv_path))
            frame_count = len(list_frames(video))
            if rows[video.name].size != frame_count:
                raise ValueError("{} has {} frames but {} labels".format(
                    video.name, frame_count, rows[video.name].size))
            aligned.append(rows[video.name][time_step:])
        return np.concatenate(aligned)

    raise FileNotFoundError(
        "Ped2 labels were not found. Pass --label_path or place frame_labels_ped2.npy under data/."
    )


def list_videos(folder):
    return sorted(
        path for path in Path(folder).iterdir()
        if path.is_dir() and not path.name.lower().endswith('_gt')
    )


def list_frames(video):
    return sorted(
        path for path in Path(video).iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
