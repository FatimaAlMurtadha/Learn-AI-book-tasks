"""Preprocessing for phone photos containing one handwritten digit.

The output follows MNIST conventions: a bright, anti-aliased digit on a black
28x28 canvas, fitted into a 20x20 box and centred by its centre of mass.

Three things matter most for photos, and all three are handled here:

1. Only the pixels that belong to the digit are kept. Paper texture inside the
   bounding box is removed, not just outside it.
2. The strokes stay grayscale. A hard binary mask looks nothing like MNIST,
   whose strokes are soft-edged.
3. The stroke thickness is normalised. A thin ballpoint line shrunk to 20x20
   is far thinner than an MNIST stroke, which is the single biggest reason a
   model trained on MNIST misclassifies phone photos.
"""

from pathlib import Path

import cv2
import numpy as np

MNIST_INK_TARGET = 0.12  # mean pixel value (0-1) of a typical MNIST digit


def _read_gray(image):
    """Read a path or an RGB/BGR/grayscale NumPy image as grayscale uint8."""
    from_path = isinstance(image, (str, Path))
    if from_path:
        raw = np.fromfile(str(image), dtype=np.uint8)
        decoded = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
        if decoded is None:
            raise FileNotFoundError(f"Could not read image: {image}")
        image = decoded

    image = np.asarray(image)
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    elif image.ndim == 3 and image.shape[2] == 3:
        code = cv2.COLOR_BGR2GRAY if from_path else cv2.COLOR_RGB2GRAY
        gray = cv2.cvtColor(image, code)
    else:
        raise ValueError("Expected a file path or a grayscale/RGB image array.")

    if gray.size == 0:
        raise ValueError("The image is empty.")
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return gray


def _ink_image(gray):
    """Remove uneven lighting and return an image where ink is bright."""
    h, w = gray.shape
    blur_size = max(15, int(round(min(h, w) * 0.10)) | 1)
    background = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
    normalized = cv2.divide(gray, np.maximum(background, 1), scale=255)

    # Paper is usually light, so ink is the dark side. Decide from the border.
    border = np.concatenate((normalized[0, :], normalized[-1, :],
                             normalized[:, 0], normalized[:, -1]))
    ink = 255 - normalized if float(np.median(border)) >= 127 else normalized
    return cv2.GaussianBlur(ink, (3, 3), 0)


def _digit_mask(ink):
    """Binary mask containing only the components that make up the digit."""
    h, w = ink.shape
    _, mask = cv2.threshold(ink, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    min_area = max(20, int(h * w * 0.00003))
    candidates = []
    for i in range(1, count):
        x, y, cw, ch, area = stats[i]
        if area < min_area:
            continue
        score = float(area)
        # Components touching the frame are usually the table or a page edge.
        if x <= 1 or y <= 1 or x + cw >= w - 1 or y + ch >= h - 1:
            score *= 0.2
        candidates.append((score, i))

    if not candidates:
        raise ValueError(
            "No digit was detected. Use a well-lit photo with one dark digit "
            "on a plain background, filling a reasonable part of the frame."
        )

    _, main = max(candidates)
    x, y, cw, ch, main_area = stats[main]

    # Keep detached strokes that clearly belong to the same digit, such as the
    # bar of a 7 or a gap in a 5, but nothing further away than that.
    keep = np.zeros(count, dtype=bool)
    keep[main] = True
    x1, y1 = x - 0.35 * cw, y - 0.35 * ch
    x2, y2 = x + 1.35 * cw, y + 1.35 * ch
    for _, i in candidates:
        bx, by, bw, bh, area = stats[i]
        if i == main or area < 0.05 * main_area:
            continue
        if bx >= x1 and by >= y1 and bx + bw <= x2 and by + bh <= y2:
            keep[i] = True

    return (keep[labels] * 255).astype(np.uint8)


def _to_mnist_canvas(strokes):
    """Fit strokes into a 20x20 box on a 28x28 canvas, centred by mass."""
    ys, xs = np.where(strokes > 0)
    if not len(ys):
        return None
    crop = strokes[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    h, w = crop.shape
    scale = min(20.0 / w, 20.0 / h)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    resized = cv2.resize(crop, (new_w, new_h), interpolation=interpolation)

    canvas = np.zeros((28, 28), dtype=np.uint8)
    top, left = (28 - new_h) // 2, (28 - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized

    moments = cv2.moments(canvas)
    if moments["m00"] > 0:
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
        shift = np.float32([[1, 0, 13.5 - cx], [0, 1, 13.5 - cy]])
        canvas = cv2.warpAffine(
            canvas, shift, (28, 28), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )

    if canvas.max() > 0:
        canvas = np.clip(
            canvas.astype(np.float32) * (255.0 / canvas.max()), 0, 255
        ).astype(np.uint8)
    return canvas


def preprocess_phone_digit(image, return_debug=False, ink_target=MNIST_INK_TARGET):
    """Convert a phone photo of one handwritten digit into 1x784 MNIST input.

    Args:
        image: Image path or RGB/grayscale NumPy array.
        return_debug: Also return grayscale, mask and the 28x28 image.
        ink_target: Mean pixel value (0-1) the 28x28 result should end up near.
            MNIST digits sit around 0.12-0.13; a thin pen line lands near 0.06
            without this correction.

    Returns:
        Float32 array shaped (1, 784), or that array plus intermediate images
        when ``return_debug=True``.
    """
    gray = _read_gray(image)

    # A bounded resolution keeps sensor noise and runtime down.
    longest = max(gray.shape)
    if longest > 1000:
        factor = 1000.0 / longest
        gray = cv2.resize(gray, None, fx=factor, fy=factor,
                          interpolation=cv2.INTER_AREA)

    ink = _ink_image(gray)
    mask = _digit_mask(ink)

    ys, xs = np.where(mask > 0)
    y1, y2 = ys.min(), ys.max() + 1
    x1, x2 = xs.min(), xs.max() + 1

    # Grayscale strokes with a truly black background: paper texture inside the
    # bounding box is masked away, soft stroke edges are kept.
    strokes = cv2.bitwise_and(ink, mask)[y1:y2, x1:x2].astype(np.float32)
    if strokes.max() > 0:
        strokes = strokes / strokes.max() * 255.0
    strokes = strokes.astype(np.uint8)

    # Try several stroke thicknesses and keep the one whose 28x28 result has
    # about as much ink as an MNIST digit.
    base_scale = min(20.0 / strokes.shape[1], 20.0 / strokes.shape[0])
    step = max(1, int(round(0.5 / base_scale)) | 1)
    canvas, best_error = None, None
    for level in range(-2, 9):
        if level == 0:
            candidate = strokes
        else:
            size = max(3, abs(level) * step | 1)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
            candidate = (cv2.dilate(strokes, kernel) if level > 0
                         else cv2.erode(strokes, kernel))
        if candidate.max() == 0:
            continue
        result = _to_mnist_canvas(candidate)
        if result is None:
            continue
        error = abs(result.mean() / 255.0 - ink_target)
        if best_error is None or error < best_error:
            canvas, best_error = result, error

    if canvas is None:
        raise ValueError("No digit was detected.")

    features = (canvas.astype(np.float32) / 255.0).reshape(1, 784)
    if return_debug:
        return features, gray, mask, canvas
    return features


def predict_digit(model, image, return_debug=False):
    """Predict one digit with test-time augmentation.

    The photo is processed at three stroke thicknesses and each version is
    shifted by one pixel in four directions; the probabilities of all fifteen
    variants are averaged. This costs nothing at training time and removes most
    of the coin-flip predictions you get from a single fragile 28x28 image.

    Returns:
        (digit, probabilities) or (digit, probabilities, debug) where debug is
        the (grayscale, mask, canvas) tuple of the middle variant.
    """
    shifts = ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1))
    targets = (ink_target := MNIST_INK_TARGET), ink_target - 0.02, ink_target + 0.02

    probabilities, debug = [], None
    for target in targets:
        features, gray, mask, canvas = preprocess_phone_digit(
            image, return_debug=True, ink_target=target
        )
        if target == MNIST_INK_TARGET:
            debug = (gray, mask, canvas)
        for dx, dy in shifts:
            matrix = np.float32([[1, 0, dx], [0, 1, dy]])
            moved = cv2.warpAffine(canvas, matrix, (28, 28), flags=cv2.INTER_LINEAR)
            variant = (moved.astype(np.float32) / 255.0).reshape(1, 784)
            probabilities.append(model.predict_proba(variant)[0])

    mean_probabilities = np.mean(probabilities, axis=0)
    digit = int(np.argmax(mean_probabilities))
    if return_debug:
        return digit, mean_probabilities, debug
    return digit, mean_probabilities
