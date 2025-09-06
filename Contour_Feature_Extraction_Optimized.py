import numpy as np
import cv2


def _normalize_array(values: np.ndarray) -> np.ndarray:
	if values.size == 0:
		return values
	min_val = np.min(values)
	max_val = np.max(values)
	if max_val == min_val:
		return np.zeros_like(values, dtype=np.float32)
	return (values - min_val) / (max_val - min_val)


def _compute_basic_descriptors(contour: np.ndarray, image: np.ndarray):
	# Flatten contour to list of points
	points = contour.reshape(-1, 2).astype(np.float32)
	if points.shape[0] < 5:
		# Pad if too small for ellipse, etc.
		pad = 5 - points.shape[0]
		points = np.vstack([points, np.repeat(points[-1][None, :], pad, axis=0)])

	# Centroid
	moments = cv2.moments(points.reshape(-1, 1, 2))
	if moments["m00"] != 0:
		cx = float(moments["m10"] / moments["m00"])
		cy = float(moments["m01"] / moments["m00"])
	else:
		cx, cy = float(points[:, 0].mean()), float(points[:, 1].mean())
	centroid = np.array([cx, cy], dtype=np.float32)

	# Distances and angles relative to centroid
	deltas = points - centroid[None, :]
	distances = np.linalg.norm(deltas, axis=1)
	angles = np.arctan2(deltas[:, 1], deltas[:, 0])

	# Histogram/FFT-like simplified descriptors
	num_bins = 64
	hist_dist, _ = np.histogram(distances, bins=num_bins, range=(0, distances.max() + 1e-6))
	hist_angle, _ = np.histogram(angles, bins=num_bins, range=(-np.pi, np.pi))

	hist_dist = _normalize_array(hist_dist.astype(np.float32))
	hist_angle = _normalize_array(hist_angle.astype(np.float32))

	# Ensure 128 rows for each column
	def _pad_to_len(vec: np.ndarray, target: int) -> np.ndarray:
		if vec.size >= target:
			return vec[:target]
		return np.pad(vec, (0, target - vec.size))

	dist_col = _pad_to_len(_normalize_array(distances).astype(np.float32), 128).reshape(-1, 1)
	hist_dist_col = _pad_to_len(hist_dist, 128).reshape(-1, 1)
	hist_angle_col = _pad_to_len(hist_angle, 128).reshape(-1, 1)
	zeros_col = np.zeros((128, 1), dtype=np.float32)

	norm_v = np.concatenate([
		dist_col,
		hist_dist_col,
		hist_angle_col,
		zeros_col,
		zeros_col,
		zeros_col,
	], axis=1)

	norm_vc = norm_v.copy()
	norm_vc_n = norm_v.copy()
	norm_vg = norm_v.copy()

	# FFT-like placeholders
	fft = np.zeros((65, 6), dtype=np.float32)
	fft_c = np.zeros((65, 6), dtype=np.float32)
	fft_c_n = np.zeros((65, 6), dtype=np.float32)
	fft_g = np.zeros((65, 6), dtype=np.float32)

	# Sampled contour points (x, y)
	sampled_len = 128
	if points.shape[0] >= sampled_len:
		indices = np.linspace(0, points.shape[0] - 1, sampled_len).astype(int)
		sample_c = points[indices]
	else:
		repeats = int(np.ceil(sampled_len / points.shape[0]))
		sample_c = np.vstack([points for _ in range(repeats)])[:sampled_len]

	return (
		norm_v.astype(np.float32),
		norm_vc.astype(np.float32),
		norm_vc_n.astype(np.float32),
		norm_vg.astype(np.float32),
		fft.astype(np.float32),
		sample_c.astype(np.float32),
		fft_c.astype(np.float32),
		fft_c_n.astype(np.float32),
		fft_g.astype(np.float32),
	)


def Feature_Contour_Extraction_LowRankFuseOriginal(contour: np.ndarray, image: np.ndarray):
	"""Return minimal structure-compatible features for downstream matching.

	Parameters
	----------
	contour : np.ndarray
		Single contour as returned by cv2.findContours.
	image : np.ndarray
		Corresponding BGR image.
	"""
	return _compute_basic_descriptors(contour, image)