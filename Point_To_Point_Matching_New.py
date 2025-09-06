import numpy as np
import os


def Contour_Points_Points_Matching(
	norm_v1, norm_vc1, norm_vc1n, norm_vg1,
	sample_c1, sample_c2,
	norm_v2, norm_vc2, norm_vc2n, norm_vg2,
	height, width, file_index,
	image1, image2,
):
	# Minimal placeholder matching: nearest neighbor by Euclidean distance between sampled points
	if sample_c1 is None or sample_c2 is None:
		return 0, 0

	points1 = sample_c1.astype(np.float32)
	points2 = sample_c2.astype(np.float32)

	if points1.size == 0 or points2.size == 0:
		return 0, 0

	# Compute pairwise distances efficiently
	diff = points1[:, None, :] - points2[None, :, :]
	dists = np.linalg.norm(diff, axis=2)
	matches_12 = np.argmin(dists, axis=1)

	# Count unique matches as a crude proxy
	unique_matches = len(np.unique(matches_12))
	prior_super_points = int(points1.shape[0])
	matched_points = int(unique_matches)

	# Save a simple npz result file
	output_dir = os.path.join("results", "matches")
	os.makedirs(output_dir, exist_ok=True)
	out_path = os.path.join(output_dir, f"match_{file_index:04d}.npz")
	np.savez_compressed(
		out_path,
		points1=points1,
		points2=points2,
		matches=matches_12,
		size=(height, width),
	)

	return prior_super_points, matched_points