import os

import numpy as np
import scipy.io as sio

import config
from dataloader import DataProcessor


def _top_eigenvectors(data, n_components):
    covariance = data @ data.T / data.shape[1]
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1][:n_components]
    return eigenvectors[:, order]


def _estimate_snr(data, mean_spectrum, projected_data):
    bands, pixels = data.shape
    signal_dim = projected_data.shape[0]
    power_y = np.sum(data ** 2) / pixels
    power_x = np.sum(projected_data ** 2) / pixels + np.sum(mean_spectrum ** 2)
    numerator = power_x - signal_dim / bands * power_y
    denominator = power_y - power_x
    if numerator <= 0 or denominator <= 0:
        return 0.0
    return 10.0 * np.log10(numerator / denominator)


def vca(data, num_endmembers, random_seed=None, snr_input=0.0):
    """
    Vertex Component Analysis.

    Args:
        data: HSI data matrix with shape (bands, pixels).
        num_endmembers: Number of endmembers to extract.
        random_seed: Seed used by the random projection step.
        snr_input: If > 0, use this SNR; otherwise estimate SNR from data.

    Returns:
        endmembers: (bands, num_endmembers)
        indices: selected column indices in the input data matrix.
        snr: estimated or provided SNR value.
    """
    data = np.asarray(data, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError(f"VCA expects a 2D matrix (bands, pixels), got {data.shape}.")

    bands, pixels = data.shape
    if not 2 <= num_endmembers <= min(bands, pixels):
        raise ValueError(
            f"num_endmembers must be in [2, {min(bands, pixels)}], got {num_endmembers}."
        )

    rng = np.random.default_rng(random_seed)
    mean_spectrum = np.mean(data, axis=1, keepdims=True)
    centered_data = data - mean_spectrum

    Ud = _top_eigenvectors(centered_data, num_endmembers)
    projected = Ud.T @ centered_data
    snr = float(snr_input) if snr_input and snr_input > 0 else _estimate_snr(
        data, mean_spectrum, projected
    )

    snr_threshold = 15.0 + 10.0 * np.log10(num_endmembers)
    if snr < snr_threshold:
        projection_dim = num_endmembers - 1
        Ud = _top_eigenvectors(centered_data, projection_dim)
        projected = Ud.T @ centered_data
        scale = np.sqrt(np.max(np.sum(projected ** 2, axis=0)))
        projected = np.vstack((projected, scale * np.ones((1, pixels), dtype=np.float32)))
    else:
        projection_dim = num_endmembers
        Ud = _top_eigenvectors(data, projection_dim)
        projected = Ud.T @ data
        projected_mean = np.mean(projected, axis=1, keepdims=True)
        denominator = projected_mean.T @ projected
        denominator = np.where(np.abs(denominator) < 1e-12, 1e-12, denominator)
        projected = projected / denominator

    A = np.zeros((num_endmembers, num_endmembers), dtype=np.float32)
    A[-1, 0] = 1.0
    indices = np.zeros(num_endmembers, dtype=np.int64)

    for i in range(num_endmembers):
        w = rng.random((num_endmembers, 1), dtype=np.float32)
        f = w - A @ np.linalg.pinv(A) @ w
        norm = np.linalg.norm(f)
        if norm < 1e-12:
            f = rng.standard_normal((num_endmembers, 1)).astype(np.float32)
            norm = np.linalg.norm(f)
        f = f / norm

        scores = f.T @ projected
        idx = int(np.argmax(np.abs(scores)))
        indices[i] = idx
        A[:, i] = projected[:, idx]

    return data[:, indices], indices, snr


def _selected_global_positions(global_data, selected_vca_indices):
    height = int(global_data["height"])
    width = int(global_data["width"])
    if "vca_mask" in global_data:
        candidate_linear_indices = np.flatnonzero(global_data["vca_mask"].reshape(-1))
    else:
        candidate_linear_indices = np.arange(height * width)

    selected_linear_indices = candidate_linear_indices[selected_vca_indices]
    rows, cols = np.unravel_index(selected_linear_indices, (height, width))
    return selected_linear_indices, np.stack((rows, cols), axis=1)


def _output_path():
    output_dir = getattr(config, "save_dir", ".")
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_dir)
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, getattr(config, "vca_save_name", "vca_result.mat"))


def run_vca_estimate(save_result=True):
    processor = DataProcessor()
    global_data = processor.load_global_hsi(
        mask_background=bool(getattr(config, "vca_mask_background", True)),
        filter_rare_classes=getattr(config, "vca_filter_rare_classes", "auto"),
    )

    vca_input = global_data["HSI_vca"]
    num_endmembers = int(getattr(config, "vca_num_endmembers"))
    random_seed = getattr(config, "vca_random_seed", 0)
    snr_input = float(getattr(config, "vca_snr_input", 0.0))

    endmembers, selected_vca_indices, snr = vca(
        vca_input,
        num_endmembers=num_endmembers,
        random_seed=random_seed,
        snr_input=snr_input,
    )
    selected_linear_indices, selected_positions = _selected_global_positions(
        global_data, selected_vca_indices
    )

    result = {
        "endmembers": endmembers.astype(np.float32),
        "selected_vca_indices": selected_vca_indices.astype(np.int64),
        "selected_linear_indices": selected_linear_indices.astype(np.int64),
        "selected_positions_0based": selected_positions.astype(np.int64),
        "selected_positions_1based": (selected_positions + 1).astype(np.int64),
        "num_endmembers": np.array([[num_endmembers]], dtype=np.int32),
        "snr": np.array([[snr]], dtype=np.float32),
        "vca_pixels": np.array([[vca_input.shape[1]]], dtype=np.int64),
        "bands": np.array([[vca_input.shape[0]]], dtype=np.int32),
        "height": np.array([[global_data["height"]]], dtype=np.int32),
        "width": np.array([[global_data["width"]]], dtype=np.int32),
    }

    if "vca_mask" in global_data:
        result["vca_mask"] = global_data["vca_mask"].astype(np.uint8)
    if "kept_original_labels" in global_data:
        result["kept_original_labels"] = global_data["kept_original_labels"].astype(np.int32)

    if save_result:
        save_path = _output_path()
        sio.savemat(save_path, result)
        result["save_path"] = save_path

    return result

if __name__ == "__main__":
    vca_result = run_vca_estimate(save_result=True)
    print("VCA finished.")
    print(f"endmembers: {int(vca_result['num_endmembers'][0, 0])}")
    print(f"input shape: ({int(vca_result['bands'][0, 0])}, {int(vca_result['vca_pixels'][0, 0])})")
    print(f"SNR: {float(vca_result['snr'][0, 0]):.4f}")
    print(f"saved to: {vca_result['save_path']}")
