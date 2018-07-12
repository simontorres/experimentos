from ccdproc import CCDData
import matplotlib.pyplot as plt
import sys

ccd = CCDData.read(sys.argv[1], unit='adu')
print(ccd.data.max())
print(ccd.data.min())

data_only = ccd.data.copy()

data_only[data_only > 66558] = 1
# data_only[data_only < 66558] = 0

manager = plt.get_current_fig_manager()
manager.window.showMaximized()

plt.title(sys.argv[1])
plt.imshow(ccd.data, clim=(int(sys.argv[2]), int(sys.argv[3])))
# plt.contour(ccd.data)
plt.tight_layout()
plt.show()
