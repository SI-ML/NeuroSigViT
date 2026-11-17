
from utils.tools import digest_file as _sha256_file
import csv
import hashlib
import importlib.util
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


WEARABLE_DATASET_NAMES = ('PADS_11_task08_TouchIndex', 'Shimmer_10_session10_AFC')
WEARABLE_EXPECTED_SPLIT_SAMPLES = {
    "PADS_11_task08_TouchIndex": (280, 92, 97),
    "Shimmer_10_session10_AFC": (69, 23, 25),
}
WEARABLE_DYNAMIC_DATASET_SPECS = {
    # mPower 三个数据集的规格取自随数据包交付的加载器（AAAI_Data/data_provider/datasets.py）；
    # 一个 subject 有多条样本，先按 subject 划分再展开样本。
    "PADS_11_task08_TouchIndex": {
        "sequence_length": 976,
        "label_names": {
            0: "Healthy",
            1: "Parkinson",
            2: "OtherMovementDisorders",
        },
        "expected_subject_counts": (280, 92, 97),
        "expected_sample_count": 469,
    },
    "Shimmer_10_session10_AFC": {
        "sequence_length": 4096,
        "label_names": {0: "HC", 1: "MildPD", 2: "ModeratePD"},
        "expected_subject_counts": (69, 23, 25),
        "expected_sample_count": 117,
    },
}
WEARABLE_REFERENCE_DATASETS_SHA256 = (
    "0113d69736e9678a43b8e2c62b344bb34e6776c023085f1a80e7e81b0a512092"
)
_WEARABLE_MODULE_CACHE = {}


# EEG 数据集的固定协议。每个数据集记录：
#   channels / subject_count：通道数、标签表行数（=磁盘上的文件数）；
#   label_sha256 / content_sha256：标签文件、全部特征文件的摘要，用来确认读到的是同一版数据；
#   split_ids：写死的受试者编号名单，训练、验证、测试各一份，与随机种子无关；
#   expected_*：划分后应得到的人数、窗口数，加载时逐项核对。
EEG_MEDFORMER_DATASET_NAMES = ('TDBRAIN', 'APAVA')
EEG_MEDFORMER_NORMALIZATION = "per_window_per_channel_standard_scaler_ddof0"
EEG_MEDFORMER_SPECS = {
    "APAVA": {
        "channels": 16,
        "subject_count": 23,
        "class_names": {0: "healthy", 1: "alzheimers"},
        "label_sha256": (
            "57d98d5207e300999428ed0951f54bb4ef29f9ab67a547feedd07d35ed6da1f7"
        ),
        "content_sha256": (
            "96984c06c5d5d41e62b2dc751733c78f67002e990f6c7dc421f155b32dfaf0d6"
        ),
        # 0917 新划分：random.Random(20260917) 按类别分层抽取受试者。
        "protocol": "apava_subject_stratified_resplit_20260917",
        "split_ids": {
            "train": (1, 2, 4, 5, 8, 10, 11, 13, 15, 17, 18, 19, 20, 21, 22),
            "vali": (6, 7, 12, 14),
            "test": (3, 9, 16, 23),
        },
        "expected_subject_counts": {"train": 15, "vali": 4, "test": 4},
        "expected_window_counts": {"train": 4536, "vali": 909, "test": 522},
    },
    "TDBRAIN": {
        # TDBRAIN 主路径：磁盘共 72 个 legacy 文件编号；下面的固定名单仅选取 50 个。
        # subject_count 是标签表/文件清单的大小，不是实际参与 train/vali/test 的人数。
        "channels": 33,
        "subject_count": 72,
        "class_names": {0: "healthy", 1: "parkinsons_disease"},
        "label_sha256": (
            "12a23405ff73b720bc5075767d63c7e00b8a8e9d18fb9972407d10b025d1e069"
        ),
        "content_sha256": (
            "3ae88137c1bc4869ee4e5c265a409c0405932f3fec9e76d04aeca3538d7e3652"
        ),
        "protocol": "medformer_code_exact_official_legacy50_only",
        "split_ids": {
            "train": tuple(range(1, 18)) + tuple(range(29, 46)),
            "vali": (18, 19, 20, 21, 46, 47, 48, 49),
            "test": (22, 23, 24, 25, 50, 51, 52, 53),
        },
        "expected_subject_counts": {"train": 34, "vali": 8, "test": 8},
        # 预测单位是已经预处理好的 256 点窗口；每个窗口继承所属文件的类别标签。
        "expected_window_counts": {"train": 4320, "vali": 960, "test": 960},
        "expected_excluded_ids": (26, 27, 28) + tuple(range(54, 73)),
    },
}


@dataclass
class WearableDataBundle:
    dataset_name: str
    data_root: Path
    reference_root: Path
    train_loader: DataLoader
    train_labels: np.ndarray
    vali_loader: DataLoader
    vali_labels: np.ndarray
    test_loader: DataLoader
    test_labels: np.ndarray
    train_dataset: Any
    vali_dataset: Any
    test_dataset: Any
    label_mode: str = "original"
    label_mapping: dict[int, int] | None = None


@dataclass(frozen=True)
class EEGMedformerSubjectRecord:
    subject_id: int
    label: int
    feature_path: Path
    window_count: int


@dataclass(frozen=True)
class EEGMedformerInventory:
    dataset_name: str
    data_root: Path
    records: tuple[EEGMedformerSubjectRecord, ...]
    split_ids: dict[str, tuple[int, ...]]
    excluded_ids: tuple[int, ...]
    protocol: str
    label_sha256: str
    content_sha256: str


@dataclass
class EEGMedformerBundle:
    inventory: EEGMedformerInventory
    train_loader: DataLoader
    train_labels: np.ndarray
    vali_loader: DataLoader
    vali_labels: np.ndarray
    test_loader: DataLoader
    test_labels: np.ndarray
    normalization: str = EEG_MEDFORMER_NORMALIZATION


def _eeg_medformer_content_sha256(data_root):
    """Match `sha256sum Feature/*.npy Label/label.npy | sha256sum`."""
    data_root = Path(data_root)
    paths = sorted(
        (data_root / "Feature").glob("feature_*.npy"), key=lambda path: path.name
    )
    paths.append(data_root / "Label" / "label.npy")
    lines = "".join(
        f"{_sha256_file(path)}  {path.relative_to(data_root).as_posix()}\n"
        for path in paths
    )
    return hashlib.sha256(lines.encode("utf-8")).hexdigest()


def find_eeg_medformer_data_root(data_dir, dataset_name):
    # TDBRAIN 输入为已生成的 Feature/feature_<id>.npy 与 Label/label.npy。
    # 这里只定位处理后数据；本函数不读取原始 EEG，也不执行滤波或采样率转换。
    dataset_name = str(dataset_name).upper()
    if dataset_name not in EEG_MEDFORMER_DATASET_NAMES:
        raise ValueError(f"Unsupported Medformer EEG dataset: {dataset_name}")

    base = Path(data_dir).expanduser()
    candidates = (
        base,
        base / dataset_name,
        base / "processed" / dataset_name,
    )
    seen = set()
    for candidate in candidates:
        normalized = candidate.resolve()
        if normalized in seen:
            continue
        seen.add(normalized)
        if (
            (normalized / "Feature").is_dir()
            and (normalized / "Label" / "label.npy").is_file()
        ):
            return normalized
    raise FileNotFoundError(
        f"Could not find processed Medformer dataset {dataset_name} below "
        f"{data_dir!r}. Expected Feature/feature_*.npy and Label/label.npy."
    )


# 核对标签摘要、文件编号、窗口形状和固定受试者名单，防止加载到不同版本的数据。
def get_eeg_medformer_inventory(data_dir, dataset_name, verify_content=True):
    # 返回 EEGMedformerInventory 元数据对象；不将所有受试者窗口拼成训练张量。
    # TDBRAIN 标签表 [72,2] 的列顺序为 [class_id, legacy_subject_id]。
    # 第 1 步：找到数据目录，取出该数据集的固定协议。
    dataset_name = str(dataset_name).upper()
    if dataset_name not in EEG_MEDFORMER_SPECS:
        raise ValueError(f"Unsupported Medformer EEG dataset: {dataset_name}")
    spec = EEG_MEDFORMER_SPECS[dataset_name]
    data_root = find_eeg_medformer_data_root(data_dir, dataset_name)
    label_path = data_root / "Label" / "label.npy"

    # 第 2 步：核对标签文件摘要（断点 S1 之前），不一致说明标签文件被换过，直接报错。
    label_sha256 = _sha256_file(label_path)
    if label_sha256 != spec["label_sha256"]:
        raise ValueError(
            f"{dataset_name} label SHA-256 mismatch: {label_sha256}; "
            f"expected {spec['label_sha256']}"
        )

    # 第 3 步：读入标签表，TDBRAIN 为 [72,2]，每行是 (类别, 文件编号)。
    labels = np.load(label_path, allow_pickle=False)
    if labels.ndim != 2 or labels.shape != (spec["subject_count"], 2):
        raise ValueError(
            f"Expected {dataset_name} labels with shape "
            f"({spec['subject_count']}, 2), got {labels.shape}."
        )

    # 第 4 步：建立 文件编号 -> 类别 的字典，同时检查编号是整数、类别合法、编号不重复。
    label_by_id = {}
    for raw_label, raw_subject_id in labels.tolist():
        if not float(raw_label).is_integer() or not float(raw_subject_id).is_integer():
            raise ValueError(
                f"{dataset_name} label rows must contain integer-valued IDs."
            )
        label = int(raw_label)
        subject_id = int(raw_subject_id)
        if label not in spec["class_names"]:
            raise ValueError(f"Unexpected {dataset_name} class ID: {label}")
        if subject_id in label_by_id:
            raise ValueError(f"Duplicate {dataset_name} subject ID: {subject_id}")
        label_by_id[subject_id] = label

    # 第 5 步：扫描 Feature/ 目录，从文件名 feature_<编号>.npy 取出编号，建立 文件编号 -> 路径 的字典。
    feature_by_id = {}
    pattern = re.compile(r"^feature_(\d+)\.npy$")
    for path in sorted((data_root / "Feature").glob("*.npy")):
        match = pattern.fullmatch(path.name)
        if match is None:
            raise ValueError(f"Unexpected {dataset_name} feature file: {path.name}")
        subject_id = int(match.group(1))
        if subject_id in feature_by_id:
            raise ValueError(f"Duplicate {dataset_name} feature ID: {subject_id}")
        feature_by_id[subject_id] = path.resolve()

    # 用受试者编号匹配特征与标签，不依赖文件遍历顺序。
    if set(feature_by_id) != set(label_by_id):
        raise ValueError(
            f"{dataset_name} feature/label subject mismatch: "
            f"missing_features={sorted(set(label_by_id) - set(feature_by_id))}, "
            f"missing_labels={sorted(set(feature_by_id) - set(label_by_id))}"
        )

    # 第 6 步：按编号从小到大逐个打开特征文件（mmap，只读文件头不读全部数据），
    # 检查形状 [窗口数,256,通道数] 和 dtype=float64，记下每个文件的窗口数（断点 S3 时 records 已建好）。
    records = []
    for subject_id in sorted(label_by_id):
        feature_path = feature_by_id[subject_id]
        array = np.load(feature_path, mmap_mode="r", allow_pickle=False)
        # 单个 TDBRAIN 文件为 float64 [N_s,256,33]；N_s 随文件而变化。
        # 此处的编号核对证明文件与标签表在本地相符，不额外证明原始 participant 身份映射。
        expected_tail = (256, spec["channels"])
        if array.ndim != 3 or tuple(array.shape[1:]) != expected_tail:
            raise ValueError(
                f"Unexpected {dataset_name} feature shape for subject "
                f"{subject_id}: {array.shape}; expected [N,{expected_tail[0]},"
                f"{expected_tail[1]}]."
            )
        if array.dtype != np.float64:
            raise ValueError(
                f"Unexpected {dataset_name} source dtype for subject "
                f"{subject_id}: {array.dtype}; expected float64."
            )
        records.append(
            EEGMedformerSubjectRecord(
                subject_id=subject_id,
                label=label_by_id[subject_id],
                feature_path=feature_path,
                window_count=int(array.shape[0]),
            )
        )

    # 第 7 步：从模块常量里取出固定划分名单（断点 S4）。这里不读随机种子，也不调用任何随机数。
    split_ids = {
        split: tuple(int(value) for value in ids)
        for split, ids in spec["split_ids"].items()
    }
    # 检查固定划分互斥；排除的受试者也必须与协议记录一致。
    assigned = [value for ids in split_ids.values() for value in ids]
    if len(assigned) != len(set(assigned)):
        raise ValueError(f"{dataset_name} subject leakage detected between splits.")
    available = set(label_by_id)
    unknown = set(assigned) - available
    if unknown:
        raise ValueError(f"{dataset_name} split contains unknown subjects: {unknown}")
    excluded_ids = tuple(sorted(available - set(assigned)))
    expected_excluded = tuple(spec.get("expected_excluded_ids", ()))
    if excluded_ids != expected_excluded:
        raise ValueError(
            f"{dataset_name} excluded cohort mismatch: {excluded_ids}; "
            f"expected {expected_excluded}."
        )
    for split, expected in spec["expected_subject_counts"].items():
        if len(split_ids[split]) != expected:
            raise ValueError(
                f"{dataset_name} {split} subject count mismatch: "
                f"{len(split_ids[split])}; expected {expected}."
            )

    # 第 8 步：对全部特征文件 + 标签文件计算总摘要并与协议核对（断点 S5 时已算完）。
    content_sha256 = spec["content_sha256"]
    if verify_content:
        content_sha256 = _eeg_medformer_content_sha256(data_root)
        if content_sha256 != spec["content_sha256"]:
            raise ValueError(
                f"{dataset_name} aggregate content SHA-256 mismatch: "
                f"{content_sha256}; expected {spec['content_sha256']}"
            )

    # 第 9 步：返回清单对象，只含元数据（编号、路径、窗口数、划分名单），不含窗口数据本身。
    return EEGMedformerInventory(
        dataset_name=dataset_name,
        data_root=data_root,
        records=tuple(records),
        split_ids=split_ids,
        excluded_ids=excluded_ids,
        protocol=spec["protocol"],
        label_sha256=label_sha256,
        content_sha256=content_sha256,
    )


# EEG 按每个原始窗口、每个通道独立标准化，沿时间轴计算总体标准差（ddof=0）。
def _normalize_eeg_medformer_windows(windows):
    # 输入/输出均为 [N_s,T,C]；TDBRAIN T=256,C=33，输出 dtype=float32。
    # 每个窗口独立处理，mean/std 是 [N_s,1,33]，没有跨窗口或跨受试者拟合。
    # 步骤：检查形状与有限值 -> 沿时间轴（axis=1）求每个窗口每个通道的均值和标准差
    # -> 标准差为 0 的通道用 1 代替，避免除零 -> 标准化并转成 float32。
    windows = np.asarray(windows)
    if windows.ndim != 3:
        raise ValueError(f"Expected EEG windows [N,T,C], got {windows.shape}.")
    if not np.isfinite(windows).all():
        raise ValueError("Medformer EEG source contains NaN or infinite values.")
    mean = windows.mean(axis=1, keepdims=True)
    std = windows.std(axis=1, ddof=0, keepdims=True)
    std = np.where(std == 0.0, 1.0, std)
    normalized = ((windows - mean) / std).astype(np.float32)
    if not np.isfinite(normalized).all():
        raise ValueError("Medformer EEG normalization produced non-finite values.")
    return normalized


def _make_eeg_medformer_split_loader(inventory, split, batch_size):
    # TDBRAIN 的 split 总窗口数 S 分别为 train=4320、vali=960、test=960。
    # 函数返回 (DataLoader, labels[S])；loader 的每项只有输入张量，标签单独传递。
    # 第 1 步：取出这个划分的受试者名单，并按编号查到各自的文件记录。
    spec = EEG_MEDFORMER_SPECS[inventory.dataset_name]
    record_by_id = {record.subject_id: record for record in inventory.records}
    subject_ids = inventory.split_ids[split]
    # 保留选中受试者的全部窗口，并核对该集合的预期窗口总数。
    total_windows = sum(record_by_id[value].window_count for value in subject_ids)
    expected_windows = spec["expected_window_counts"][split]
    if total_windows != expected_windows:
        raise ValueError(
            f"{inventory.dataset_name} {split} window count mismatch: "
            f"{total_windows}; expected {expected_windows}."
        )

    # 第 2 步：预先分配整个划分的输入数组 [S,通道,256] 和标签数组 [S]（断点 S6）。
    inputs = np.empty(
        (total_windows, spec["channels"], 256), dtype=np.float32
    )
    labels = np.empty(total_windows, dtype=np.int64)
    # 以下两个 [S] 数组记录窗口来源；训练随机种子不会重新决定这里的受试者划分。
    sample_subject_ids = np.empty(total_windows, dtype=np.int64)
    sample_window_indices = np.empty(total_windows, dtype=np.int64)
    # 第 3 步：逐个受试者读入全部窗口，逐窗口标准化（断点 A1 在转置写入之前），
    # 再把这个人的所有窗口连续写进数组；标签、编号、窗口序号按同样的行位置写入，三者行对齐。
    cursor = 0
    for subject_id in subject_ids:
        record = record_by_id[subject_id]
        source = np.load(record.feature_path, mmap_mode="r", allow_pickle=False)
        normalized = _normalize_eeg_medformer_windows(source)
        next_cursor = cursor + record.window_count
        # 原始 [窗口数, 时间, 通道] 转为模型输入 [窗口数, 通道, 时间]。
        inputs[cursor:next_cursor] = normalized.transpose(0, 2, 1)
        labels[cursor:next_cursor] = record.label
        sample_subject_ids[cursor:next_cursor] = subject_id
        sample_window_indices[cursor:next_cursor] = np.arange(
            record.window_count, dtype=np.int64
        )
        cursor = next_cursor

    # 第 4 步：包装成 TensorDataset（断点 S7）。数据集中只放输入，标签作为单独的数组返回。
    tensor_dataset = TensorDataset(torch.from_numpy(inputs))
    # 单条 dataset[i][0] 为 [33,256]；一个 batch[0] 为 [B,33,256]。
    # 当前 NeuroSigVIA.sh 为 TDBRAIN 设 B=8；这不是分类内部 64 点块的数量。
    tensor_dataset.sample_subject_ids = sample_subject_ids
    # 保存窗口在原受试者文件中的位置，供划分审计与预测回溯使用。
    tensor_dataset.sample_window_indices = sample_window_indices
    tensor_dataset.split_name = split
    # 第 5 步：按顺序（shuffle=False）切批，保证第 i 行特征始终对应 labels[i]。
    loader = DataLoader(
        tensor_dataset,
        num_workers=0,
        batch_size=batch_size,
        shuffle=False,
    )
    # 静态特征提取保持窗口顺序与 labels 对齐；训练器随后对索引批次执行 shuffle。
    return loader, labels


def get_eeg_medformer_dataloaders(dataset_name, args):
    # 主方法 EEG 入口调用链：inventory -> 三次 split_loader -> EEGMedformerBundle。
    # TDBRAIN: X_train[4320,33,256], X_val/X_test[960,33,256]；y 均为一维 int64。
    # 第 1 步：确认使用的是固定的划分协议和逐窗口标准化方式。
    protocol = getattr(args, "eeg_protocol", "medformer_code_exact")
    if protocol != "medformer_code_exact":
        raise ValueError(f"Unsupported Medformer EEG protocol: {protocol}")
    normalization = getattr(
        args, "eeg_normalization", EEG_MEDFORMER_NORMALIZATION
    )
    if normalization != EEG_MEDFORMER_NORMALIZATION:
        raise ValueError(f"Unsupported Medformer EEG normalization: {normalization}")

    # 第 2 步：核对数据并取出固定划分名单（S1–S5）。
    inventory = get_eeg_medformer_inventory(
        args.data_dir, dataset_name, verify_content=True
    )
    # 第 3 步：依次组装训练、验证、测试三个划分（S6、A1、S7 各命中三次，先 train 后 vali、test）。
    train_loader, train_labels = _make_eeg_medformer_split_loader(
        inventory, "train", args.batch_size
    )
    vali_loader, vali_labels = _make_eeg_medformer_split_loader(
        inventory, "vali", args.batch_size
    )
    test_loader, test_labels = _make_eeg_medformer_split_loader(
        inventory, "test", args.batch_size
    )
    # 第 4 步：打包成 bundle，交给 exp_classification.main 使用。
    bundle = EEGMedformerBundle(
        inventory=inventory,
        train_loader=train_loader,
        train_labels=train_labels,
        vali_loader=vali_loader,
        vali_labels=vali_labels,
        test_loader=test_loader,
        test_labels=test_labels,
    )

    # 第 5 步：打印每个划分的类别分布，便于在日志里核对。
    distributions = []
    for split, labels in (
        ("train", train_labels),
        ("vali", vali_labels),
        ("test", test_labels),
    ):
        values, counts = np.unique(labels, return_counts=True)
        distribution = "/".join(
            f"{int(value)}:{int(count)}" for value, count in zip(values, counts)
        )
        distributions.append(f"{split}=[{distribution}]")
    print(
        f"EEG {inventory.dataset_name}: "
        f"train={len(train_labels)}, vali={len(vali_labels)}, "
        f"test={len(test_labels)}; {' '.join(distributions)}; "
        f"protocol={inventory.protocol}; normalization={normalization}; "
        "subject_split=PASS; metric_unit=window"
    )
    return bundle




def find_wearable_data_root(data_dir, dataset_name):
    if dataset_name not in WEARABLE_DATASET_NAMES:
        raise ValueError(f"Unsupported wearable dataset: {dataset_name}")

    base = Path(data_dir).expanduser()
    candidates = (
        base,
        base / "wearable",
        base / "Neuro" / "wearable",
        base / "Neuro",
        base / "med_data" / "wearable",
        base / "med_data",
    )
    seen = set()
    for candidate in candidates:
        normalized = candidate.resolve()
        if normalized in seen:
            continue
        seen.add(normalized)
        if not (
            (normalized / dataset_name / "Meta" / "subject_map.csv").is_file()
            and (normalized / dataset_name / "Feature").is_dir()
        ):
            continue
        return normalized

    raise FileNotFoundError(
        f"Could not find wearable/{dataset_name} below {data_dir!r}. "
        "Expected the dataset Feature/ and Meta/subject_map.csv files."
    )


def _find_wearable_reference_root(data_root):
    candidates = (
        Path(data_root) / "data_provider",
        Path(__file__).resolve().parents[1] / "data_provider",
    )
    for candidate in candidates:
        if (
            (candidate / "datasets.py").is_file()
            and (candidate / "split_reference_seed42.csv").is_file()
        ):
            return candidate
    raise FileNotFoundError(
        "Missing wearable reference loader and split file. Expected "
        "data_provider/datasets.py and data_provider/split_reference_seed42.csv "
        "either beside the dataset root or in the repository."
    )


def _load_wearable_reference(reference_root):
    module_path = Path(reference_root) / "datasets.py"
    digest = hashlib.sha256(module_path.read_bytes()).hexdigest()
    if digest != WEARABLE_REFERENCE_DATASETS_SHA256:
        raise ValueError(
            f"Unexpected SHA-256 for {module_path}: {digest}. "
            f"Expected {WEARABLE_REFERENCE_DATASETS_SHA256}."
        )

    cache_key = str(module_path.resolve())
    if cache_key in _WEARABLE_MODULE_CACHE:
        return _WEARABLE_MODULE_CACHE[cache_key]

    module_name = f"_wearable_reference_{digest[:12]}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load wearable reference module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    _WEARABLE_MODULE_CACHE[cache_key] = module
    return module


def _make_wearable_tensor_loader(source_dataset, batch_size):
    if source_dataset.X is None or source_dataset.y is None:
        raise ValueError("Wearable source dataset did not load samples")

    # 可穿戴参考加载器输出 [N,T,6]，模型接收 [N,6,T]；样本顺序不变。
    inputs = torch.from_numpy(source_dataset.X.transpose(0, 2, 1))
    tensor_dataset = TensorDataset(inputs)
    tensor_dataset.source_dataset = source_dataset
    tensor_dataset.sample_subject_ids = source_dataset.sample_subject_ids
    loader = DataLoader(
        tensor_dataset,
        num_workers=0,
        batch_size=batch_size,
        shuffle=False,
    )
    labels = np.asarray(source_dataset.y, dtype=np.int64)
    return loader, labels


def _standardize_wearable_bundle_from_train(bundle, batch_size):
    """Standardize selected task channels with training-split statistics only."""
    # 可穿戴数据仅用训练集估计各通道统计量，再应用于验证集和测试集。
    train_data = bundle.train_loader.dataset.tensors[0]
    mean = train_data.mean(dim=(0, 2), keepdim=True)
    std = train_data.std(dim=(0, 2), unbiased=False, keepdim=True)
    std = torch.where(std < 1e-8, torch.ones_like(std), std)

    for split in ("train", "vali", "test"):
        loader = getattr(bundle, f"{split}_loader")
        standardized = ((loader.dataset.tensors[0] - mean) / std).float()
        tensor_dataset = TensorDataset(standardized)
        tensor_dataset.source_dataset = loader.dataset.source_dataset
        tensor_dataset.sample_subject_ids = loader.dataset.sample_subject_ids
        setattr(
            bundle,
            f"{split}_loader",
            DataLoader(
                tensor_dataset,
                num_workers=0,
                batch_size=batch_size,
                shuffle=False,
            ),
        )


def _apply_wearable_label_protocol(bundle, label_mode, batch_size):
    family = "shimmer" if bundle.dataset_name.startswith("Shimmer_") else "pads"
    # Shimmer 合并两种 PD 严重程度；PADS 按任务筛选目标类别并重编码。
    protocols = {
        "shimmer_hc_vs_pd": ("shimmer", {0: 0, 1: 1, 2: 1}),
        "pads_pd_vs_hc": ("pads", {0: 0, 1: 1}),
        "pads_pd_vs_omd": ("pads", {2: 0, 1: 1}),
    }

    if label_mode == "original":
        labels = bundle.train_dataset.label_names
        mapping = {int(label): int(label) for label in labels}
    else:
        if label_mode not in protocols:
            raise ValueError(f"Unsupported wearable label mode: {label_mode}")
        expected_family, mapping = protocols[label_mode]
        if family != expected_family:
            raise ValueError(
                f"Label mode {label_mode} is only valid for {expected_family} "
                f"datasets, not {bundle.dataset_name}."
            )

    for split in ("train", "vali", "test"):
        source_dataset = getattr(bundle, f"{split}_dataset")
        loader = getattr(bundle, f"{split}_loader")
        original_labels = np.asarray(source_dataset.y, dtype=np.int64)
        keep = np.isin(original_labels, np.asarray(list(mapping), dtype=np.int64))
        if not np.any(keep):
            raise ValueError(
                f"{bundle.dataset_name} split={split} has no samples for {label_mode}."
            )

        inputs = loader.dataset.tensors[0][torch.as_tensor(keep)]
        tensor_dataset = TensorDataset(inputs)
        tensor_dataset.source_dataset = source_dataset
        tensor_dataset.sample_subject_ids = np.asarray(
            source_dataset.sample_subject_ids
        )[keep]
        mapped_labels = np.asarray(
            [mapping[int(label)] for label in original_labels[keep]],
            dtype=np.int64,
        )
        setattr(
            bundle,
            f"{split}_loader",
            DataLoader(
                tensor_dataset,
                num_workers=0,
                batch_size=batch_size,
                shuffle=False,
            ),
        )
        setattr(bundle, f"{split}_labels", mapped_labels)

    bundle.label_mode = label_mode
    bundle.label_mapping = mapping


def _validate_wearable_bundle(bundle, reference_module):
    reference_csv = bundle.reference_root / "split_reference_seed42.csv"
    reference = reference_module._read_reference_csv(reference_csv)
    reference_status = "NOT_AVAILABLE"
    if bundle.dataset_name in reference:
        reference_module._verify_reference_assignment(bundle.train_dataset, reference)
        reference_status = "PASS"
    reference_module._validate_label_file(bundle.train_dataset)

    split_items = (
        ("train", bundle.train_dataset, bundle.train_loader, bundle.train_labels),
        ("vali", bundle.vali_dataset, bundle.vali_loader, bundle.vali_labels),
        ("test", bundle.test_dataset, bundle.test_loader, bundle.test_labels),
    )
    split_sample_counts = []
    for split, source_dataset, loader, labels in split_items:
        if source_dataset.split != split:
            raise AssertionError(
                f"{bundle.dataset_name}: expected split={split}, got {source_dataset.split}"
            )
        expected_shape = (
            len(source_dataset),
            source_dataset.num_channels,
            source_dataset.sequence_length,
        )
        actual_shape = tuple(loader.dataset.tensors[0].shape)
        if actual_shape != expected_shape:
            raise AssertionError(
                f"{bundle.dataset_name} split={split}: expected NeuroSigVIA shape "
                f"{expected_shape}, got {actual_shape}"
            )
        if not np.array_equal(labels, source_dataset.y):
            raise AssertionError(
                f"{bundle.dataset_name} split={split}: adapter labels changed"
            )
        expected_labels = np.asarray(
            [
                source_dataset.subject_label[int(subject_id)]
                for subject_id in source_dataset.sample_subject_ids
            ],
            dtype=np.int64,
        )
        if not np.array_equal(labels, expected_labels):
            raise AssertionError(
                f"{bundle.dataset_name} split={split}: sample labels do not match subjects"
            )
        split_sample_counts.append(len(source_dataset))

    actual_split_samples = tuple(split_sample_counts)
    expected_split_samples = WEARABLE_EXPECTED_SPLIT_SAMPLES[bundle.dataset_name]
    if actual_split_samples != expected_split_samples:
        raise AssertionError(
            f"{bundle.dataset_name}: expected train/vali/test samples "
            f"{expected_split_samples}, got {actual_split_samples}"
        )
    total_samples = sum(actual_split_samples)
    if total_samples != bundle.train_dataset.expected_sample_count:
        raise AssertionError(
            f"{bundle.dataset_name}: expected "
            f"{bundle.train_dataset.expected_sample_count} total samples, "
            f"got {total_samples}"
        )
    return reference_status


def get_wearable_dataloaders(dataset_name, args):
    # 可穿戴数据入口（PADS / Shimmer / mPower）。步骤：
    # 1. 找数据目录和参考加载器 datasets.py（摘要固定，不可修改）；
    # 2. 用参考加载器按受试者划分出 train/vali/test 三个数据集；
    # 3. 转成 [N,通道,时间] 的 DataLoader，并逐项核对样本数、标签与受试者一致；
    # 4. 按任务筛选/重编码类别，再只用训练集统计量做通道标准化。
    data_root = find_wearable_data_root(args.data_dir, dataset_name)
    reference_root = _find_wearable_reference_root(data_root)
    reference_module = _load_wearable_reference(reference_root)
    dataset_classes = {
        dataset_class.dataset_name: dataset_class
        for dataset_class in reference_module.DATASET_CLASSES
    }
    for dynamic_name, spec in WEARABLE_DYNAMIC_DATASET_SPECS.items():
        dataset_classes[dynamic_name] = type(
            f"{dynamic_name}Dataset",
            (reference_module.SubjectMapDataset,),
            {
                "dataset_name": dynamic_name,
                "relative_directories": (dynamic_name,),
                **spec,
            },
        )
    dataset_class = dataset_classes[dataset_name]

    split_datasets = {
        split: dataset_class(
            data_root=data_root,
            split=split,
            normalize=False,
            verbose=False,
        )
        for split in ("train", "vali", "test")
    }
    train_loader, train_labels = _make_wearable_tensor_loader(
        split_datasets["train"], args.batch_size
    )
    vali_loader, vali_labels = _make_wearable_tensor_loader(
        split_datasets["vali"], args.batch_size
    )
    test_loader, test_labels = _make_wearable_tensor_loader(
        split_datasets["test"], args.batch_size
    )
    bundle = WearableDataBundle(
        dataset_name=dataset_name,
        data_root=data_root,
        reference_root=reference_root,
        train_loader=train_loader,
        train_labels=train_labels,
        vali_loader=vali_loader,
        vali_labels=vali_labels,
        test_loader=test_loader,
        test_labels=test_labels,
        train_dataset=split_datasets["train"],
        vali_dataset=split_datasets["vali"],
        test_dataset=split_datasets["test"],
    )
    reference_status = _validate_wearable_bundle(bundle, reference_module)

    label_mode = getattr(args, "wearable_label_mode", "original")
    # 先按任务筛选样本，再计算训练集统计量，避免无关类别影响归一化。
    _apply_wearable_label_protocol(bundle, label_mode, args.batch_size)
    _standardize_wearable_bundle_from_train(bundle, args.batch_size)

    split_distributions = []
    for split, labels in (
        ("train", bundle.train_labels),
        ("vali", bundle.vali_labels),
        ("test", bundle.test_labels),
    ):
        values, counts = np.unique(labels, return_counts=True)
        distribution = "/".join(
            f"{int(value)}:{int(count)}" for value, count in zip(values, counts)
        )
        split_distributions.append(f"{split}=[{distribution}]")
    print(
        f"Wearable {dataset_name}: "
        f"train={len(bundle.train_labels)}, vali={len(bundle.vali_labels)}, "
        f"test={len(bundle.test_labels)}; "
        f"label_mode={label_mode}; {' '.join(split_distributions)}; "
        f"subject_split=PASS; reference={reference_status}"
    )
    return bundle



def write_eeg_medformer_split_audit(bundle, result_dir):
    # 输出 splits/TDBRAIN_subject_split.csv（含被排除编号），供协议与预测来源追溯。
    split_dir = Path(result_dir) / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    inventory = bundle.inventory
    output_path = split_dir / f"{inventory.dataset_name}_subject_split.csv"
    spec = EEG_MEDFORMER_SPECS[inventory.dataset_name]
    split_by_id = {}
    for split, subject_ids in inventory.split_ids.items():
        for subject_id in subject_ids:
            split_by_id[subject_id] = split

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "dataset_name",
                "protocol",
                "normalization",
                "metric_unit",
                "legacy_subject_id",
                "label_id",
                "label_name",
                "window_count",
                "split",
                "label_sha256",
                "aggregate_content_sha256",
            ]
        )
        for record in inventory.records:
            writer.writerow(
                [
                    inventory.dataset_name,
                    inventory.protocol,
                    bundle.normalization,
                    "processed_one_second_window",
                    record.subject_id,
                    record.label,
                    spec["class_names"][record.label],
                    record.window_count,
                    split_by_id.get(record.subject_id, "excluded"),
                    inventory.label_sha256,
                    inventory.content_sha256,
                ]
            )
    return output_path


def write_wearable_split_audit(bundle, result_dir):
    split_dir = Path(result_dir) / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    output_path = split_dir / f"{bundle.dataset_name}_subject_split.csv"

    assignments = []
    for split, source_dataset in (
        ("train", bundle.train_dataset),
        ("vali", bundle.vali_dataset),
        ("test", bundle.test_dataset),
    ):
        assignments.extend(
            (
                int(subject_id),
                int(source_dataset.subject_label[int(subject_id)]),
                split,
            )
            for subject_id in source_dataset.selected_subject_ids
        )

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "dataset_name",
            "label_protocol",
            "numeric_subject_id",
            "original_label_id",
            "label_id",
            "split",
        ])
        for subject_id, original_label_id, split in sorted(assignments):
            if original_label_id not in bundle.label_mapping:
                continue
            label_id = bundle.label_mapping[original_label_id]
            writer.writerow(
                [
                    bundle.dataset_name,
                    bundle.label_mode,
                    subject_id,
                    original_label_id,
                    label_id,
                    split,
                ]
            )
    return output_path
