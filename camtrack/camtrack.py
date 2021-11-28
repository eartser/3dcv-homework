#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple

from cv2 import cv2
import numpy as np

from corners import CornerStorage, FrameCorners
from data3d import CameraParameters, PointCloud, Pose
import frameseq
import sortednp as snp
from _camtrack import (
    PointCloudBuilder,
    create_cli,
    calc_point_cloud_colors,
    pose_to_view_mat3x4,
    to_opencv_camera_mat3x3,
    view_mat3x4_to_pose,
    build_correspondences,
    triangulate_correspondences,
    TriangulationParameters,
    rodrigues_and_translation_to_view_mat3x4,
    Correspondences,
    _remove_correspondences_with_ids,
    eye3x4,
)


def _calc_view_matrix(cloud: PointCloudBuilder, corners: FrameCorners, intrinsic_mat, rep_error):
    ids, (indices_1, indices_2) = snp.intersect(cloud.ids.flatten(), corners.ids.flatten(), indices=True)
    succ, rvec, tvec, inliers = cv2.solvePnPRansac(
        cloud.points[indices_1],
        corners.points[indices_2],
        intrinsic_mat,
        np.array([]),
        iterationsCount=108,
        reprojectionError=rep_error,
        flags=cv2.SOLVEPNP_EPNP
    )
    inliers = inliers.flatten()
    succ, rvec, tvec = cv2.solvePnP(
        cloud.points[indices_1][inliers],
        corners.points[indices_2][inliers],
        intrinsic_mat,
        np.array([]),
        rvec=rvec,
        tvec=tvec,
        useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    return rodrigues_and_translation_to_view_mat3x4(rvec, tvec)


def _view_from_correspondences(correspondences: Correspondences,
                               intrinsic_mat: np.ndarray,
                               triang_params: TriangulationParameters,
                               threshold_inl: float):
    if len(correspondences.ids) < 5:
        return None, 0
    prob = 0.999
    threshold = 1.0
    emat, inliers = cv2.findEssentialMat(correspondences.points_1,
                                         correspondences.points_2,
                                         intrinsic_mat,
                                         method=cv2.RANSAC,
                                         prob=prob,
                                         threshold=threshold)
    inliers = inliers.flatten()

    _, inliers_h = cv2.findHomography(correspondences.points_1,
                                      correspondences.points_2,
                                      cv2.RANSAC,
                                      triang_params.max_reprojection_error)
    if np.sum(inliers) / np.sum(inliers_h) < threshold_inl:
        return None, 0

    r1, r2, t_ = cv2.decomposeEssentialMat(emat)
    best_viewmat = None
    best_rt_points_number = 0
    for r in [r1, r2]:
        for t in [-t_, t_]:
            view_mat = np.hstack([r, t.reshape(-1, 1)])
            points3d, ids, median_cos = triangulate_correspondences(correspondences,
                                                                    eye3x4(),
                                                                    view_mat,
                                                                    intrinsic_mat,
                                                                    triang_params)
            if len(points3d) > best_rt_points_number:
                best_rt_points_number = len(points3d)
                best_viewmat = view_mat
    return best_viewmat, best_rt_points_number


def _get_known_views(corner_storage: CornerStorage, intrinsic_mat: np.ndarray, triang_params: TriangulationParameters):
    max_frame_dist = 30
    threshold_inl = 0.2
    points_number_threshold = 100
    while threshold_inl < 1:
        for i in range(len(corner_storage)):
            for j in range(i + 1, min(i + max_frame_dist + 1, len(corner_storage))):
                viewmat, points_number = _view_from_correspondences(
                    build_correspondences(corner_storage[i], corner_storage[j]),
                    intrinsic_mat,
                    triang_params,
                    threshold_inl
                )
                if viewmat is None or points_number < points_number_threshold:
                    continue
                return (
                    (i, view_mat3x4_to_pose(eye3x4())),
                    (j, view_mat3x4_to_pose(viewmat))
                )
        threshold_inl *= 2
    raise RuntimeError(f'Ошибочка, не смогла инициализировать((')


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:

    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    reproj_error = 7.5
    triangulate_params = TriangulationParameters(
        reproj_error,
        1.0,
        0.1
    )
    eps = 60

    if known_view_1 is None or known_view_2 is None:
        known_view_1, known_view_2 = _get_known_views(corner_storage, intrinsic_mat, triangulate_params)

    frame_count = len(corner_storage)
    view_mats: list = [None] * frame_count
    point_cloud_builder = PointCloudBuilder()

    frame1, frame2 = known_view_1[0], known_view_2[0]
    view_mats[frame1] = pose_to_view_mat3x4(known_view_1[1])
    view_mats[frame2] = pose_to_view_mat3x4(known_view_2[1])
    correspondences = build_correspondences(corner_storage[frame1], corner_storage[frame2])
    points3d, correspondence_ids, median_cos = triangulate_correspondences(
        correspondences, view_mats[frame1], view_mats[frame2], intrinsic_mat, triangulate_params
    )
    point_cloud_builder.add_points(correspondence_ids, points3d)
    print(f'''Processed frames {frame1} and {frame2}. 
{len(points3d)} points triangulated.
''')

    allowed_frames = set()
    for i in range(max(0, frame1 - eps), min(frame1 + eps + 1, frame_count)):
        allowed_frames.add(i)

    for i in range(max(0, frame2 - eps), min(frame2 + eps + 1, frame_count)):
        allowed_frames.add(i)

    while True:
        best_frame, best_inliers_cnt = None, 0
        for frame in allowed_frames:
            if view_mats[frame] is not None:
                continue
            ids, (indices_1, indices_2) = snp.intersect(
                point_cloud_builder.ids.flatten(),
                corner_storage[frame].ids.flatten(),
                indices=True
            )
            if len(ids) < 4:
                continue
            succ, _, _, inliers = cv2.solvePnPRansac(
                point_cloud_builder.points[indices_1],
                corner_storage[frame].points[indices_2],
                intrinsic_mat,
                np.array([]),
                iterationsCount=108,
                reprojectionError=reproj_error,
                flags=cv2.SOLVEPNP_EPNP
            )
            if succ and len(inliers) > best_inliers_cnt:
                best_frame = frame
                best_inliers_cnt = len(inliers)

        if best_frame is None:
            print(f'''Processing finished.
Points cloud size: {len(point_cloud_builder.ids)}.
''')
            break
        print(f'Chose frame {best_frame} with {best_inliers_cnt} inliers.')
        print(f'Processing frame {best_frame}...')
        view_mats[best_frame] = _calc_view_matrix(
            point_cloud_builder, corner_storage[best_frame], intrinsic_mat, reproj_error
        )

        for i in range(max(0, best_frame - eps), min(best_frame + eps + 1, frame_count)):
            allowed_frames.add(i)

        best_frame_second = None
        best_median_cos = 2
        for frame in range(max(0, best_frame - eps), min(best_frame + eps + 1, frame_count)):
            if view_mats[frame] is None or frame == best_frame:
                continue
            correspondences = build_correspondences(corner_storage[best_frame], corner_storage[frame])
            _, _, median_cos = triangulate_correspondences(
                correspondences,
                view_mats[best_frame],
                view_mats[frame],
                intrinsic_mat,
                triangulate_params
            )
            if median_cos < best_median_cos:
                best_frame_second = frame
                best_median_cos = median_cos
        if best_frame_second is None:
            best_frame_second = frame2
        correspondences = build_correspondences(corner_storage[best_frame], corner_storage[best_frame_second])
        points3d, correspondence_ids, median_cos = triangulate_correspondences(
            correspondences,
            view_mats[best_frame],
            view_mats[best_frame_second],
            intrinsic_mat,
            triangulate_params
        )
        point_cloud_builder.add_points(correspondence_ids, points3d)
        print(f'{len(points3d)} points triangulated. Current size of points cloud: {len(point_cloud_builder.ids)}.\n')

    calc_point_cloud_colors(
        point_cloud_builder,
        rgb_sequence,
        view_mats,
        intrinsic_mat,
        corner_storage,
        reproj_error
    )
    point_cloud = point_cloud_builder.build_point_cloud()
    poses = list(map(view_mat3x4_to_pose, view_mats))
    return poses, point_cloud


if __name__ == '__main__':
    # pylint:disable=no-value-for-parameter
    create_cli(track_and_calc_colors)()
