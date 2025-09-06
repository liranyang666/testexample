"""
优化的轮廓跟踪和特征提取程序
专门处理 test7.mp4 视频文件

主要功能：
- 使用 Contour_Feature_Extraction_Optimized.py 进行轮廓特征提取
- 使用 Point_To_Point_Matching_New.py 进行点点匹配
- 简化的代码结构，移除冗余代码
- 自动保存匹配结果

作者: AI Assistant
版本: v4.0 - 极简化优化版
"""

import cv2
import numpy as np
import time
import sys
import os
import copy
from Point_To_Point_Matching_New import Contour_Points_Points_Matching
from Contour_Feature_Extraction_Optimized import Feature_Contour_Extraction_LowRankFuseOriginal


def normalize(Narray):
	"""归一化数组到[0,1]区间"""
	if Narray.size == 0:
		return Narray
	return (Narray - np.min(Narray)) / (np.max(Narray) - np.min(Narray))


def _find_contours_compat(binary_image):
	res = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	if len(res) == 3:
		_, contours, hierarchy = res
	else:
		contours, hierarchy = res
	return contours, hierarchy


def process_contour_features_and_match(frame1, frame2, file_index):
	"""
	处理两帧图像的轮廓特征提取和点点匹配

	参数:
		frame1: 第一帧图像
		frame2: 第二帧图像
		file_index: 保存结果文件索引

	返回:
		success: 是否成功处理
		result_info: 处理结果信息
	"""

	try:
		# 对两帧分别进行轮廓检测和特征提取
		results = []
		for i, frame in enumerate([frame1, frame2], 1):
			# 转换为灰度图并进行边缘检测
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			edges = cv2.Canny(gray, 50, 150)

			# 寻找轮廓（兼容不同 OpenCV 版本）
			contours, _ = _find_contours_compat(edges)

			if len(contours) == 0:
				print(f"[WARNING] 帧{i} 未检测到轮廓")
				# 返回空结果
				empty_result = (np.zeros((128, 6)), np.zeros((128, 6)),
								  np.zeros((128, 6)), np.zeros((128, 6)),
								  np.zeros((64 + 1, 6)), np.zeros((128, 2)),
								  np.zeros((64 + 1, 6)), np.zeros((64 + 1, 6)),
								  np.zeros((64 + 1, 6)))
				results.append(empty_result)
				continue

			# 选择最大的轮廓
			max_contour = max(contours, key=cv2.contourArea)
			print(f"[INFO] 帧{i} 检测到 {len(contours)} 个轮廓，最大轮廓面积: {cv2.contourArea(max_contour)}")

			# 使用优化模块进行特征提取
			result = Feature_Contour_Extraction_LowRankFuseOriginal(max_contour, frame)
			results.append(result)

		if len(results) != 2:
			return False, {"error": "特征提取失败"}

		# 解包特征提取结果
		(norm_v1, norm_vc1, norm_vc1n, norm_vg1, fft1, sample_c1,
		 fft_c1, fft_c1n, fft_g1) = results[0]

		(norm_v2, norm_vc2, norm_vc2n, norm_vg2, fft2, sample_c2,
		 fft_c2, fft_c2n, fft_g2) = results[1]

		print("[INFO] 特征提取完成，开始点点匹配...")

		# 获取图像尺寸
		height, width = frame1.shape[:2]

		# 调用点点匹配函数
		prior_super_points, matched_points = Contour_Points_Points_Matching(
			norm_v1, norm_vc1, norm_vc1n, norm_vg1,  # 前帧特征
			sample_c1, sample_c2,  # 前后帧轮廓
			norm_v2, norm_vc2, norm_vc2n, norm_vg2,  # 后帧特征
			height, width, file_index,  # 图像尺寸和文件索引
			frame1, frame2  # 原始图像
		)

		result_info = {
			"prior_super_points": prior_super_points,
			"matched_points": matched_points,
			"feature_extraction_success": True,
			"point_matching_success": True
		}

		print(f"[INFO] 点点匹配完成!")
		print(f"[INFO] 前帧超点数量: {prior_super_points}")
		print(f"[INFO] 匹配点数量: {matched_points}")

		return True, result_info

	except Exception as e:
		print(f"[ERROR] 轮廓特征提取和匹配失败: {str(e)}")
		import traceback
		traceback.print_exc()
		return False, {"error": str(e)}


def main():
	"""主函数 - 只处理 test7.mp4"""

	print("=" * 60)
	print("优化的轮廓跟踪和特征提取程序")
	print("专门处理 test7.mp4 视频文件")
	print("=" * 60)

	# 视频文件路径
	video_path = "test7.mp4"

	# 检查视频文件是否存在
	if not os.path.exists(video_path):
		print(f"[ERROR] 视频文件不存在: {video_path}")
		return

	print(f"[INFO] 开始处理视频文件: {video_path}")

	# 打开视频文件
	video = cv2.VideoCapture(video_path)
	if not video.isOpened():
		print(f"[ERROR] 无法打开视频文件: {video_path}")
		return

	# 获取视频信息
	height = int(video.get(4))
	width = int(video.get(3))
	fps = int(video.get(5))
	total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

	print(f"[INFO] 视频信息:")
	print(f"  总帧数: {total_frames}")
	print(f"  FPS: {fps}")
	print(f"  分辨率: {width}x{height}")

	print("=" * 60)
	print("[INFO] 开始处理视频帧...")

	# 初始化变量
	prev_frame = None
	processed_pairs = 0
	k = 0  # 初始化帧计数器
	start_time = time.time()

	try:
		while True:
			ret, frame = video.read()

			if not ret:
				break

			k += 1

			# 每隔5帧进行一次处理以提高速度
			if k % 5 != 0:
				continue

			print(f"[INFO] 处理第 {k} 帧")

			if prev_frame is not None:
				try:
					# 使用轮廓特征提取和点点匹配
					success, result_info = process_contour_features_and_match(prev_frame, frame, processed_pairs)

					if success:
						print(f"[INFO] 第 {processed_pairs + 1} 对帧匹配完成")
						print(f"[INFO] 前帧超点数量: {result_info['prior_super_points']}")
						print(f"[INFO] 匹配点数量: {result_info['matched_points']}")
						processed_pairs += 1
					else:
						print(f"[WARNING] 第 {k} 帧匹配失败: {result_info.get('error', '未知错误')}")
				except Exception as e:
					print(f"[ERROR] 处理第 {k} 帧时出错: {str(e)}")
					continue

			prev_frame = frame.copy()

			# 显示进度
			if k % 50 == 0:
				elapsed_time = time.time() - start_time
				progress = k / total_frames * 100
				print(f"[INFO] 处理进度: {progress:.1f}%")

	except KeyboardInterrupt:
		print("[INFO] 用户中断程序")
	except Exception as e:
		print(f"[ERROR] 程序运行出错: {str(e)}")
	finally:
		# 释放资源
		video.release()

	# 显示最终结果
	total_time = time.time() - start_time
	print("[INFO] 处理完成!")
	print(f"[INFO] 总处理时间: {total_time:.2f} 秒")
	print(f"[INFO] 处理的帧对数: {processed_pairs}")

	print(" " + "=" * 60)
	print("✅ 程序执行完毕！")
	print("🎯 成功使用轮廓特征提取和点点匹配处理 test7.mp4")
	print("📁 匹配结果已由 Point_To_Point_Matching_New.py 保存")
	print("=" * 60)


if __name__ == "__main__":
	main()