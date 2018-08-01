from astroscrappy import detect_cosmics
from ccdproc import CCDData
import glob
import matplotlib.pyplot as plt
import numpy as np

data_list = glob.glob('data/fzsto_*')

for image in data_list:
    print(image)
    ccd = CCDData.read(image, unit='adu')
    ccd.data = np.rot90(ccd.data)
    c0, c1 = ccd.data.min(), ccd.data.mean()
    # print(ccd.data.min(), ccd.data.max(), ccd.data.mean())
    original = ccd.copy()

    mask, ccd.data = detect_cosmics(ccd.data)
    mask[mask == True] = 1
    mask[mask == False] = 0
    print(mask, True in mask)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 9))
    fig.canvas.set_window_title(image)

    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()

    ax1.set_title('Original Data')
    ax1.imshow(original.data, clim=(c0, c1))
    ax2.set_title('Mask')
    ax2.imshow(mask, clim=(0, 1))
    ax3.set_title('Cleaned Data')
    ax3.imshow(ccd.data, clim=(c0, c1))
    plt.tight_layout()

    plt.savefig(image + '.png', dpi=300)

    plt.show()
