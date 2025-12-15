import numpy as np

# Estimate number of courser grid
n_grid_h = 24
n_grid_w = 50

# convert corser grid to mm
grid_to_mm = 5

# convert corser grid to mm
mm_to_mv = 0.5 / 5
mm_to_s= 0.2 / 5

# Start
duration_short = 12.5
duration_long = 50
n_rows = 4
n_cols = 4
lead_positions = {
    'DI': (0, 0),
    'DII': (1, 0),
    'DIII': (2, 0),
    'AVR': (0, 1),
    'AVL': (1, 1),
    'AVF': (2, 1),
    'V1': (0, 2),
    'V2': (1, 2),
    'V3': (2, 2),
    'V4': (0, 3),
    'V5': (1, 3),
    'V6': (2, 3),
    'long DII': (3, 0)
}
y_offset = - (n_grid_h / n_rows) * grid_to_mm
x_offset = (n_grid_w / n_cols) * grid_to_mm


def estimate_pixels_per_mm(img):
    height, width = img.shape
    pixels_per_mm_h = height / (n_grid_h * grid_to_mm)
    pixels_per_mm_w = width / (n_grid_w * grid_to_mm)
    return pixels_per_mm_h, pixels_per_mm_w

def signal_to_ecg(signal, pixels_per_mm_h, pixels_per_mm_w):
    y = (signal - y_offset) * mm_to_mv / pixels_per_mm_h
    sample_frequency = pixels_per_mm_w / (mm_to_s)
    return y, sample_frequency

def vectorize(lead_img, mask = None):
    if mask is None:
        mask = lead_img < 200
    H, W = mask.shape[0], mask.shape[1]
    signal_pixels = np.zeros(W)
    for i in range(W):
        ys = np.nonzero(mask[:, i])[0]  # y-locations of the signal
        if len(ys) > 0:
            v = 255 - lead_img[ys, i]
            signal_pixels[i] = H - np.mean(ys[v==max(v)])  # take mean pixel coordinate
        else:
            signal_pixels[i] = np.nan
    # might be good to do something smarter to potentially remove some artifacts

    return signal_pixels
