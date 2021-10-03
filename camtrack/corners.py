#! /usr/bin/env python3

__all__ = [
    'FrameCorners',
    'CornerStorage',
    'build',
    'dump',
    'load',
    'draw',
    'calc_track_interval_mappings',
    'calc_track_len_array_mapping',
    'without_short_tracks'
]

import click
import cv2
import numpy as np
import pims

from _corners import (
    FrameCorners,
    CornerStorage,
    StorageImpl,
    dump,
    load,
    draw,
    calc_track_interval_mappings,
    calc_track_len_array_mapping,
    without_short_tracks,
    create_cli
)


class _CornerStorageBuilder:

    def __init__(self, progress_indicator=None):
        self._progress_indicator = progress_indicator
        self._corners = dict()

    def set_corners_at_frame(self, frame, corners):
        self._corners[frame] = corners
        if self._progress_indicator is not None:
            self._progress_indicator.update(1)

    def build_corner_storage(self):
        return StorageImpl(item[1] for item in sorted(self._corners.items()))


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:

    def depth_to_255(img):
        img = img * 255
        return img.astype('uint8')

    block_size = 12
    quality = 0.05
    min_distance = 8
    win_size = 2 * block_size

    ids = np.array([], dtype=int).reshape([-1, 1])
    corners = np.array([], dtype=int).reshape([-1, 2])
    next_id = 0

    for frame, image in enumerate(frame_sequence):
        mask = np.full_like(image, 255, dtype='uint8')
        if corners.size != 0:
            prev_image = frame_sequence[frame - 1]
            corners, st, _ = cv2.calcOpticalFlowPyrLK(
                depth_to_255(prev_image),
                depth_to_255(image),
                corners.astype('float32'),
                None,
                winSize=(win_size, win_size)
            )
            st = st.flatten() == 1
            corners = corners[st].astype(int)
            ids = ids[st]
            for corner in corners:
                cv2.circle(mask, corner, min_distance, 0, -1)
        ext_corners = cv2.goodFeaturesToTrack(image, 0, quality, min_distance, mask=mask, blockSize=block_size)
        if ext_corners is not None:
            ext_corners = ext_corners.reshape([-1, 2]).astype(int)
            ext_ids = np.array(range(next_id, next_id + len(ext_corners))).reshape([-1, 1])
            next_id += len(ext_corners)
            corners = np.vstack((corners, ext_corners))
            ids = np.vstack((ids, ext_ids))
        frame_corners = FrameCorners(
            ids,
            corners,
            np.full((len(corners), 1), block_size)
        )
        builder.set_corners_at_frame(frame, frame_corners)


def build(frame_sequence: pims.FramesSequence,
          progress: bool = True) -> CornerStorage:
    """
    Build corners for all frames of a frame sequence.

    :param frame_sequence: grayscale float32 frame sequence.
    :param progress: enable/disable building progress bar.
    :return: corners for all frames of given sequence.
    """
    if progress:
        with click.progressbar(length=len(frame_sequence),
                               label='Calculating corners') as progress_bar:
            builder = _CornerStorageBuilder(progress_bar)
            _build_impl(frame_sequence, builder)
    else:
        builder = _CornerStorageBuilder()
        _build_impl(frame_sequence, builder)
    return builder.build_corner_storage()


if __name__ == '__main__':
    create_cli(build)()  # pylint:disable=no-value-for-parameter
