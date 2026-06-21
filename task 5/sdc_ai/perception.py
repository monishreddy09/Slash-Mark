from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple
import sys
import types
import pickle

import cv2
import numpy as np
from skimage.feature import hog


def _install_sklearn_pickle_shims() -> None:
    """Compatibility shims for older sklearn pickles bundled in the Udacity repo."""
    try:
        import sklearn.svm._classes as svm_classes
        import sklearn.preprocessing._data as prep_data
    except Exception:
        return

    mod = types.ModuleType("sklearn.svm.classes")
    for name in ["LinearSVC", "SVC", "NuSVC", "LinearSVR", "NuSVR"]:
        if hasattr(svm_classes, name):
            setattr(mod, name, getattr(svm_classes, name))
    sys.modules["sklearn.svm.classes"] = mod

    mod2 = types.ModuleType("sklearn.preprocessing.data")
    for name in ["StandardScaler", "MinMaxScaler", "RobustScaler", "Normalizer"]:
        if hasattr(prep_data, name):
            setattr(mod2, name, getattr(prep_data, name))
    sys.modules["sklearn.preprocessing.data"] = mod2


def _hog_features(img, orient, pix_per_cell, cell_per_block, feature_vec=True):
    return hog(
        img,
        orientations=orient,
        pixels_per_cell=(pix_per_cell, pix_per_cell),
        cells_per_block=(cell_per_block, cell_per_block),
        transform_sqrt=True,
        visualize=False,
        feature_vector=feature_vec,
    )


def _bin_spatial(img, size=(32, 32)):
    return cv2.resize(img, size).ravel()


def _color_hist(img, nbins=32, bins_range=(0, 256)):
    c1 = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)[0]
    c2 = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)[0]
    c3 = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)[0]
    return np.concatenate((c1, c2, c3))


def _convert_color(image, color_space):
    if color_space == "RGB":
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if color_space == "HSV":
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    if color_space == "LUV":
        return cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
    if color_space == "HLS":
        return cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    if color_space == "YUV":
        return cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    if color_space == "YCrCb":
        return cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    return image


def _extract_features(image, params):
    feature_image = _convert_color(image, params["color_space"])
    features = []
    if params.get("spatial_feat", True):
        features.append(_bin_spatial(feature_image, size=params.get("spatial_size", (32, 32))))
    if params.get("hist_feat", True):
        features.append(_color_hist(feature_image, nbins=params.get("hist_bins", 32)))
    if params.get("hog_feat", True):
        hog_channel = params.get("hog_channel", "ALL")
        if hog_channel == "ALL":
            hog_feats = []
            for ch in range(feature_image.shape[2]):
                hog_feats.append(_hog_features(
                    feature_image[:, :, ch],
                    orient=params["orient"],
                    pix_per_cell=params["pix_per_cell"],
                    cell_per_block=params["cell_per_block"],
                ))
            features.append(np.ravel(hog_feats))
        else:
            features.append(_hog_features(
                feature_image[:, :, hog_channel],
                orient=params["orient"],
                pix_per_cell=params["pix_per_cell"],
                cell_per_block=params["cell_per_block"],
            ))
    return np.concatenate(features)


def _slide_windows(img, x_start_stop=None, y_start_stop=None, xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    if x_start_stop is None:
        x_start_stop = [None, None]
    if y_start_stop is None:
        y_start_stop = [None, None]
    if x_start_stop[0] is None:
        x_start_stop[0] = 0
    if x_start_stop[1] is None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] is None:
        y_start_stop[0] = 0
    if y_start_stop[1] is None:
        y_start_stop[1] = img.shape[0]

    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    nx_pix_per_step = int(xy_window[0] * (1 - xy_overlap[0]))
    ny_pix_per_step = int(xy_window[1] * (1 - xy_overlap[1]))
    nx_windows = max(1, int(xspan / nx_pix_per_step) - 1)
    ny_windows = max(1, int(yspan / ny_pix_per_step) - 1)

    windows = []
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            startx = xs * nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys * ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            if endx <= img.shape[1] and endy <= img.shape[0]:
                windows.append(((startx, starty), (endx, endy)))
    return windows


@dataclass
class TrafficLightResult:
    state: str
    confidence: float


class TrafficLightDetector:
    """Heuristic traffic light detector for demo purposes."""

    def detect(self, frame_bgr: np.ndarray) -> TrafficLightResult:
        h, w = frame_bgr.shape[:2]
        crop = frame_bgr[: max(1, int(h * 0.22)), int(w * 0.40): int(w * 0.60)]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        red1 = cv2.inRange(hsv, (0, 100, 120), (10, 255, 255))
        red2 = cv2.inRange(hsv, (170, 100, 120), (180, 255, 255))
        yellow = cv2.inRange(hsv, (15, 100, 120), (35, 255, 255))
        green = cv2.inRange(hsv, (35, 80, 120), (90, 255, 255))

        scores = {
            "RED": float((red1.sum() + red2.sum()) / 255.0),
            "YELLOW": float(yellow.sum() / 255.0),
            "GREEN": float(green.sum() / 255.0),
        }
        state = max(scores, key=scores.get)
        best = scores[state]
        sorted_scores = sorted(scores.values(), reverse=True)
        second = sorted_scores[1] if len(sorted_scores) > 1 else 0.0

        # Require a clear winner and enough signal, otherwise report UNKNOWN.
        if best < 25.0 or (best / max(second, 1.0)) < 1.6:
            return TrafficLightResult(state="UNKNOWN", confidence=0.0)

        total = sum(scores.values()) + 1e-6
        confidence = best / total
        return TrafficLightResult(state=state, confidence=float(confidence))


class VehicleDetector:
    """
    HOG + linear SVM vehicle detector using the pretrained model bundled in the Udacity repo.
    If the pretrained files are unavailable, the detector gracefully returns no boxes.
    """

    def __init__(self, model_dir: Optional[Path] = None, history: int = 6):
        _install_sklearn_pickle_shims()
        self.history: Deque[List[Tuple[Tuple[int, int], Tuple[int, int]]]] = deque(maxlen=history)
        self.svc = None
        self.scaler = None
        self.params = None

        if model_dir is not None:
            self._load_models(model_dir)

    def _load_models(self, model_dir: Path) -> None:
        try:
            self.svc = pickle.loads((model_dir / "svm_trained.pickle").read_bytes())
            self.scaler = pickle.loads((model_dir / "feature_scaler.pickle").read_bytes())
            self.params = pickle.loads((model_dir / "feat_extraction_params.pickle").read_bytes())
        except Exception:
            self.svc = self.scaler = self.params = None

    @staticmethod
    def _draw_boxes(img, bboxes, color=(255, 0, 0), thick=3):
        out = img.copy()
        for bbox in bboxes:
            cv2.rectangle(out, bbox[0], bbox[1], color, thick)
        return out

    @staticmethod
    def _heatmap(frame_shape, boxes, threshold=1):
        heat = np.zeros(frame_shape[:2], dtype=np.float32)
        for (x1, y1), (x2, y2) in boxes:
            heat[y1:y2, x1:x2] += 1.0
        heat[heat <= threshold] = 0
        return heat

    def _find_cars(self, img_bgr: np.ndarray, ystart: int, ystop: int, scale: float) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        if self.svc is None or self.scaler is None or self.params is None:
            return []

        img_tosearch = img_bgr[ystart:ystop, :, :]
        ctrans_tosearch = _convert_color(img_tosearch, self.params["color_space"])

        if scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch, (int(imshape[1] / scale), int(imshape[0] / scale)))

        ch1, ch2, ch3 = cv2.split(ctrans_tosearch)
        nxblocks = (ch1.shape[1] // self.params["pix_per_cell"]) - self.params["cell_per_block"] + 1
        nyblocks = (ch1.shape[0] // self.params["pix_per_cell"]) - self.params["cell_per_block"] + 1
        window = 64
        nblocks_per_window = (window // self.params["pix_per_cell"]) - self.params["cell_per_block"] + 1
        cells_per_step = 2
        nxsteps = max(1, (nxblocks - nblocks_per_window) // cells_per_step)
        nysteps = max(1, (nyblocks - nblocks_per_window) // cells_per_step)

        hog1 = _hog_features(ch1, self.params["orient"], self.params["pix_per_cell"], self.params["cell_per_block"], feature_vec=False)
        hog2 = _hog_features(ch2, self.params["orient"], self.params["pix_per_cell"], self.params["cell_per_block"], feature_vec=False)
        hog3 = _hog_features(ch3, self.params["orient"], self.params["pix_per_cell"], self.params["cell_per_block"], feature_vec=False)

        hot_windows = []
        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb * cells_per_step
                xpos = xb * cells_per_step
                xleft = xpos * self.params["pix_per_cell"]
                ytop = ypos * self.params["pix_per_cell"]

                hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

                xleft = int(xleft * scale)
                ytop = int(ytop * scale)

                subimg = cv2.resize(img_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))
                if subimg.shape[:2] != (64, 64):
                    continue

                converted = _convert_color(subimg, self.params["color_space"])
                spatial_features = _bin_spatial(converted, self.params["spatial_size"])
                hist_features = _color_hist(converted, nbins=self.params["hist_bins"])
                features = np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1)
                test_features = self.scaler.transform(features)
                prediction = self.svc.predict(test_features)
                if int(prediction[0]) == 1:
                    xbox_left = int(xleft * scale)
                    ytop_draw = int(ytop * scale)
                    win_draw = int(window * scale)
                    hot_windows.append(((xbox_left, ytop_draw + ystart), (xbox_left + win_draw, ytop_draw + win_draw + ystart)))
        return hot_windows

    def detect(self, frame_bgr: np.ndarray) -> Dict[str, object]:
        if self.svc is None:
            return {"boxes": [], "label": "vehicle detector unavailable"}

        boxes = []
        boxes.extend(self._find_cars(frame_bgr, 380, min(frame_bgr.shape[0], 660), 1.0))
        boxes.extend(self._find_cars(frame_bgr, 380, min(frame_bgr.shape[0], 660), 1.5))
        boxes.extend(self._find_cars(frame_bgr, 380, min(frame_bgr.shape[0], 660), 2.0))

        if boxes:
            self.history.append(boxes)
        if self.history:
            boxes = [b for history_boxes in self.history for b in history_boxes]

        heat = self._heatmap(frame_bgr.shape, boxes, threshold=max(1, len(self.history) - 1))
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats((heat > 0).astype(np.uint8), connectivity=8)
        final_boxes = []
        for label in range(1, num_labels):
            x, y, w, h, area = stats[label]
            if area >= 20:
                final_boxes.append(((x, y), (x + w, y + h)))

        annotated = self._draw_boxes(frame_bgr, final_boxes, (255, 0, 0), 3)
        return {"boxes": final_boxes, "annotated": annotated, "heatmap": heat}
