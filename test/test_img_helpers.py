from ecgprep.img_helpers import vectorize, signal_to_ecg
from ecgprep.read_ecg import read_ecg
from ecgprep.preprocess import preprocess_ecg
import cv2
import numpy as np



# Test vectorize
def test_vectorize():
    ecg_from_signal, freq_signal, lead_names = read_ecg('ptbxl/00001_hr', format='wfdb')

    lead_src_img = cv2.imread('img/ptbxl-individual-leads/long DII.png')
    mask_src_img = cv2.imread('img/ptbxl-individual-leads-mask/long DII.png')

    lead_img = cv2.cvtColor(lead_src_img, cv2.COLOR_RGB2GRAY)
    mask_img = cv2.cvtColor(mask_src_img, cv2.COLOR_RGB2GRAY)>240
    signal_from_image = vectorize(lead_img, mask_img)

    assert signal_from_image .shape[0] == lead_img.shape[1]


    ecg_from_image, freq_img_extracted = signal_to_ecg(signal_from_image, 4, 4)


    ecg_signal, sample_frequency, lead_names = preprocess_ecg(ecg_from_signal, freq_signal, lead_names, new_freq=freq_img_extracted, remove_baseline=True)

    # remove the mean
    ecg_1 = ecg_from_image - np.mean(ecg_from_image)
    ecg_2 = ecg_signal[1, :] - np.mean(ecg_signal[1, :])

    # make both same shape
    l1 = ecg_1.shape
    l2 = ecg_2.shape
    l = min(l1, l2)[0]
    ecg_1 = ecg_1[:l]
    ecg_2 = ecg_2[:l]

    # compute mean_absolute error:
    error = np.mean(np.abs(ecg_1 - ecg_2))
    assert error < 0.05

    # see visually
    import matplotlib.pyplot as plt

    plt.plot(ecg_1)
    plt.plot(ecg_2)
    plt.show()

