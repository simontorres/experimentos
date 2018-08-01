from ccdproc import ImageFileCollection
from pipeline.wcs import WCS
from ccdproc import CCDData
import os

import matplotlib.pyplot as plt

wcs = WCS()

_directory = '/data/simon/development/soar/goodman/pipeline/data/ref_comp'

_keywords = ['OBJECT',
             'INSTCONF',
             'CAM_ANG',
             'GRT_ANG',
             'CAM_TARG',
             'GRT_TARG',
             'FILTER2',
             'GRATING',
             'SLIT',
             'WAVMODE',
             'EXPTIME',
             'GSP_VERS',
             'GSP_FUNC',
             'GSP_ORDR',
             'GSP_NPIX',
             'GSP_C000',
             'GSP_C001',
             'GSP_C002',
             'GSP_C003']

ic = ImageFileCollection(_directory, keywords=_keywords)

pic = ic.summary.to_pandas()

grouped = pic.groupby(['WAVMODE']).size().reset_index().rename(columns={0: 'count'})

print(grouped.index)
for i in grouped.index:
    print(" ")
    print(grouped.iloc[i]['WAVMODE'])
    print('=====\n')
    conf_group = pic[(pic['WAVMODE'] == grouped.iloc[i]['WAVMODE'])]

    for _file in conf_group['file']:
        ccd = CCDData.read(os.path.join(_directory, _file), unit='adu')
        wav, inte = wcs.read_gsp_wcs(ccd)
        plt.plot(wav, inte, label=_file)

    plt.legend()
    plt.show()
    print(conf_group)