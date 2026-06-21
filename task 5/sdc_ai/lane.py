from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Deque, Dict, Tuple

import cv2
import numpy as np


@dataclass
class LaneMetrics:
    curvature_m: float
    offset_m: float
    confidence: float


class LaneDetector:
    """
    Lane detection using:
    1) color + gradient thresholding
    2) perspective transform
    3) sliding window polynomial fit
    4) temporal smoothing
    """

    def __init__(
        self,
        history: int = 8,
        ym_per_pix: float = 30 / 720,
        xm_per_pix: float = 3.7 / 700,
    ) -> None:
        self.left_fits: Deque[np.ndarray] = deque(maxlen=history)
        self.right_fits: Deque[np.ndarray] = deque(maxlen=history)
        self.ym_per_pix = ym_per_pix
        self.xm_per_pix = xm_per_pix
        self.M = None
        self.Minv = None
        self.last_metrics = LaneMetrics(curvature_m=0.0, offset_m=0.0, confidence=0.0)

    @staticmethod
    def _threshold_channel(channel: np.ndarray, lo: int, hi: int) -> np.ndarray:
        binary = np.zeros_like(channel, dtype=np.uint8)
        binary[(channel >= lo) & (channel <= hi)] = 1
        return binary

    def _binary_pipeline(self, img_bgr: np.ndarray) -> np.ndarray:
        hls = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HLS)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        l = hls[:, :, 1]
        s = hls[:, :, 2]

        s_binary = self._threshold_channel(s, 90, 255)
        l_binary = self._threshold_channel(l, 120, 255)

        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        abs_sobelx = np.absolute(sobelx)
        if abs_sobelx.max() > 0:
            scaled = np.uint8(255 * abs_sobelx / abs_sobelx.max())
        else:
            scaled = np.zeros_like(gray, dtype=np.uint8)
        grad_binary = self._threshold_channel(scaled, 20, 255)

        yellow_mask = cv2.inRange(hls, (10, 30, 80), (40, 255, 255)) // 255

        combined = np.zeros_like(gray, dtype=np.uint8)
        combined[((s_binary == 1) & (l_binary == 1)) | (grad_binary == 1) | (yellow_mask == 1)] = 1

        kernel = np.ones((5, 5), np.uint8)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
        return combined.astype(np.uint8)

    def _get_perspective(self, width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.M is not None and self.Minv is not None:
            return self.M, self.Minv

        src = np.float32([
            [0.43 * width, 0.65 * height],
            [0.58 * width, 0.65 * height],
            [0.10 * width, 1.00 * height],
            [0.95 * width, 1.00 * height],
        ])
        dst = np.float32([
            [0.25 * width, 0.0],
            [0.75 * width, 0.0],
            [0.25 * width, 1.0 * height],
            [0.75 * width, 1.0 * height],
        ])

        self.M = cv2.getPerspectiveTransform(src, dst)
        self.Minv = cv2.getPerspectiveTransform(dst, src)
        return self.M, self.Minv

    def _sliding_window_fit(self, binary_warped: np.ndarray):
        histogram = np.sum(binary_warped[binary_warped.shape[0] // 2 :, :], axis=0)
        midpoint = histogram.shape[0] // 2
        leftx_base = int(np.argmax(histogram[:midpoint]))
        rightx_base = int(np.argmax(histogram[midpoint:]) + midpoint)

        n_windows = 9
        window_height = binary_warped.shape[0] // n_windows
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        leftx_current = leftx_base
        rightx_current = rightx_base
        margin = 80
        minpix = 50

        left_lane_inds = []
        right_lane_inds = []

        for window in range(n_windows):
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            if len(good_left_inds) > minpix:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = int(np.mean(nonzerox[good_right_inds]))

        left_lane_inds = np.concatenate(left_lane_inds) if left_lane_inds else np.array([])
        right_lane_inds = np.concatenate(right_lane_inds) if right_lane_inds else np.array([])

        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        if len(leftx) < 500 or len(rightx) < 500:
            return None, None, 0.0, None

        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

        lane_width = float(np.mean(right_fitx - left_fitx))
        density_score = float(np.clip((len(leftx) + len(rightx)) / 20000.0, 0.0, 1.0))
        width_score = float(np.exp(-((lane_width - 700.0) ** 2) / (2 * 250.0 ** 2)))
        confidence = density_score * width_score
        return left_fit, right_fit, confidence, (ploty, left_fitx, right_fitx)

    def _curvature_and_offset(self, binary_warped: np.ndarray, left_fit: np.ndarray, right_fit: np.ndarray):
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        y_eval = np.max(ploty)

        left_fit_cr = np.polyfit(ploty * self.ym_per_pix,
                                 (left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]) * self.xm_per_pix,
                                 2)
        right_fit_cr = np.polyfit(ploty * self.ym_per_pix,
                                  (right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]) * self.xm_per_pix,
                                  2)

        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * self.ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / max(abs(2 * left_fit_cr[0]), 1e-6)
        right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * self.ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / max(abs(2 * right_fit_cr[0]), 1e-6)
        curvature = float((left_curverad + right_curverad) / 2.0)

        left_x = left_fit[0] * y_eval**2 + left_fit[1] * y_eval + left_fit[2]
        right_x = right_fit[0] * y_eval**2 + right_fit[1] * y_eval + right_fit[2]
        lane_center = (left_x + right_x) / 2.0
        frame_center = binary_warped.shape[1] / 2.0
        offset_m = float((frame_center - lane_center) * self.xm_per_pix)
        return curvature, offset_m

    def _draw_lane(self, original: np.ndarray, binary_warped: np.ndarray, left_fit: np.ndarray, right_fit: np.ndarray):
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

        warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right)).astype(np.int32)

        cv2.fillPoly(color_warp, [pts], (0, 255, 0))
        _, Minv = self._get_perspective(original.shape[1], original.shape[0])
        newwarp = cv2.warpPerspective(color_warp, Minv, (original.shape[1], original.shape[0]))
        return cv2.addWeighted(original, 1.0, newwarp, 0.3, 0)

    def process_frame(self, frame_bgr: np.ndarray):
        img = frame_bgr.copy()
        h, w = img.shape[:2]
        M, _ = self._get_perspective(w, h)
        binary = self._binary_pipeline(img)
        warped = cv2.warpPerspective(binary, M, (w, h), flags=cv2.INTER_LINEAR)

        left_fit, right_fit, confidence, _ = self._sliding_window_fit(warped)
        if left_fit is not None and right_fit is not None:
            self.left_fits.append(left_fit)
            self.right_fits.append(right_fit)
        elif self.left_fits and self.right_fits:
            left_fit = np.mean(np.stack(self.left_fits), axis=0)
            right_fit = np.mean(np.stack(self.right_fits), axis=0)
            confidence *= 0.5
        else:
            annotated = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            metrics = {"curvature_m": 0.0, "offset_m": 0.0, "confidence": 0.0}
            return annotated, metrics

        left_fit = np.mean(np.stack(self.left_fits), axis=0)
        right_fit = np.mean(np.stack(self.right_fits), axis=0)

        curvature, offset_m = self._curvature_and_offset(warped, left_fit, right_fit)
        overlay = self._draw_lane(img, warped, left_fit, right_fit)

        cv2.putText(overlay, f"Curvature: {curvature:,.0f} m", (40, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(overlay, f"Center offset: {offset_m:+.2f} m", (40, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(overlay, f"Lane confidence: {confidence:.2f}", (40, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

        self.last_metrics = LaneMetrics(curvature_m=float(curvature), offset_m=float(offset_m), confidence=float(confidence))
        metrics = {
            "curvature_m": float(curvature),
            "offset_m": float(offset_m),
            "confidence": float(confidence),
        }
        return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), metrics
