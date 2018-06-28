from astropy import units as u

from astropy.modeling import models, fitting
from astropy.stats import sigma_clip
from ccdproc import CCDData
from scipy import signal

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import sys

sys.path.append('/user/simon/development/soar/goodman')
from pipeline.wcs.wcs import WCS


class WavelengthCalibration(object):

    def __init__(self,
                 pixel_size=15,
                 pixel_scale=0.15,
                 focal_length=377.3,
                 max_division=12):
        self._pixel_size = pixel_size * u.micrometer
        self._pixel_scale = pixel_scale * u.arcsec
        self._focal_length = focal_length * u.mm
        self._max_division = float(int(max_division))
        self._file_name = None
        self._ccd = None
        self._nist_df = None
        self._filtered_nist_df = None
        self._spec_dict = None
        self._linear_model = None
        self._linear_fitter = None
        self._non_linear_model = None
        self._non_linear_fitter = None
        self._data_mask = None
        self._nist_mask = None
        self._global_shift = None
        self._bin_limits = None
        self._local_shifts = None
        self._data_peaks = None
        self.ref = None
        self.wcs = WCS()

    def __call__(self, spectrum, *args, **kwargs):
        if not isinstance(spectrum, CCDData):
            self.file_name = spectrum
            self.ccd = self._read_file()
        else:
            self.ccd = spectrum

        if self._ccd is not None and isinstance(self._ccd, CCDData):
            self.spec = self._get_spectral_limits()
            self._data_peaks = self._get_peaks_in_data()


        lrms1 = self._fit_linear_model()
        print("Linear Fit 1: {:f}".format(lrms1))
        self._create_mask_from_data()
        self._create_mask_from_nist()

        self._global_shift = self._global_cross_correlate()

        spec_dict = self.spec

        spec_dict['blue'] = self._linear_model(0 + self._global_shift) * u.angstrom
        spec_dict['center'] = self._linear_model(
            0.5 * len(self._ccd.data) + self._global_shift) * u.angstrom
        spec_dict['red'] = self._linear_model(
            len(self._ccd.data) + self._global_shift) * u.angstrom

        self.spec = spec_dict

        self._fit_linear_model()
        self._clean_nist_df(model=self._linear_model)
        # rebuild mask from nist using global shift
        self._create_mask_from_nist()

        self.ref = self.wcs.read_gsp_wcs(ccd=self._ccd)
        self._non_linear_model = models.Chebyshev1D(degree=3)
        self._non_linear_fitter = fitting.LevMarLSQFitter()
        # self._match_peaks_to_nist()

        # print(self._global_cross_correlate())

        # for k in self._data_peaks:
        #     plt.axvline(k, color='c')

        # plt.plot(self._data_mask, color='k')
        # plt.plot(self._nist_mask, color='r', alpha=0.3)
        # plt.show()

        self._local_shifts = self._local_cross_correlate()
        print(self._local_shifts)
        rms_l = self._fit_non_linear_model()
        print("first fit: {:f}".format(rms_l))
        self._clean_nist_df(model=self._non_linear_model)
        rms_nl = self._fit_non_linear_model()
        print("second fit: {:f}".format(rms_nl))

        # wav =self._non_linear_model(peaks)

        plt.title('RMS: {:f}'.format(rms_nl))
        # for v in wav:
        #     plt.axvline(v, color='k')
        for line in self._filtered_nist_df.wavelength.values:
            plt.axvline(line, color='g')
        f_df = self.nist_df[self.nist_df.rel_int > 3]
        print(f_df)
        new_angstrom = []
        for k in self._non_linear_model(self._data_peaks):
            index = np.argmin(np.abs(f_df.wavelength.values - k))
            plt.axvline(k, color='c', alpha=0.3)
            plt.axvline(f_df.iloc[index].wavelength, color='r')
            new_angstrom.append(f_df.iloc[index].wavelength)

        # self._non_linear_model = self.__fit(self._non_linear_model, self._non_linear_fitter, self._data_peaks, new_angstrom)
        # last_rms = self.__get_rms(self._data_peaks, self._non_linear_model, new_angstrom)
        # plt.title("RMS: {:f}".format(last_rms))

        plt.plot(self.ref[0], self.ref[1], color='k', label='Reference Data')
        plt.axhline(self._ccd.data.mean(), color='r')

        # plt.plot(self._linear_model(range(len(self._nist_mask))), self._nist_mask * self.ccd.data.max(), color='r')
        # plt.plot(self._linear_model(range(len(self._data_mask))), self._data_mask * self.ccd.data.max(), color='c')
        # plt.plot(self._linear_model(range(len(self._ccd.data))), self._ccd.data, color='b', label='Linear Model')
        plt.plot(self._non_linear_model(range(len(self._ccd.data))), self._ccd.data, color='m', label='Non-Linear Model')
        plt.legend()
        plt.show()

    def _match_peaks_to_nist(self):
        fdf = self.nist_df[self.nist_df.rel_int > 5]
        print(fdf)
        pixel = self._data_peaks
        angstrom = self._linear_model(pixel)
        new_angstrom = []

        for val in self.nist_df.wavelength.values:
            plt.axvline(val, color='r', alpha=0.2)

        for i in range(len(angstrom)):
            plt.axvline(angstrom[i], color='g')
            index = np.argmin(np.abs(fdf.wavelength.values - angstrom[i]))
            new_angstrom.append(fdf.iloc[index].wavelength)
            plt.axvline(fdf.iloc[index].wavelength, color='r')
        self._non_linear_model = self.__fit(model=self._non_linear_model,
                                            fitter=self._non_linear_fitter,
                                            pixel=pixel,
                                            wavelength=new_angstrom)
        plt.plot(self.ref[0], self.ref[1], color='k')
        for p in self._linear_model(pixel):
            plt.axvline(p, color='g')
        for a in new_angstrom:
            plt.axvline(a, color='m')
        plt.plot(self._non_linear_model(range(len(self._ccd.data))), self._ccd.data, color='b')
        plt.show()

    @staticmethod
    def __fit(model, fitter, pixel, wavelength):
        fitted_model = fitter(model, pixel, wavelength)
        return fitted_model

    @staticmethod
    def __get_rms(pixels, model, reference_wavelength):
        new_wavelength = model(pixels)
        results = []
        for value in new_wavelength:
            index = np.argmin(np.abs(reference_wavelength - value))
            results.append(reference_wavelength[index] - value)
        rms = [i ** 2 for i in results]
        return np.sqrt(np.sum(rms) / len(rms))

    def _read_file(self):
        if self._file_name is not None:
            ccd = CCDData.read(self.file_name, unit=u.adu)
            return ccd

    def _clean_nist_df(self, model):
        data_mean = self._ccd.data.mean()
        wavelength_axis = model(range(len(self._ccd.data)))
        index_to_remove = []
        # print(self._filtered_nist_df.index.tolist())
        # print(self._filtered_nist_df.iloc[190])
        for i in range(len(self._filtered_nist_df)): #.index.tolist():

            pix = int(
                np.argmin(abs(wavelength_axis -
                              self._filtered_nist_df.iloc[i]['wavelength'])))
            if np.max(self._ccd.data[pix - 10:pix + 10]) < data_mean:
                index_to_remove.append(i)
                # print(self._filtered_nist_df.iloc[i])
        self._filtered_nist_df = self._filtered_nist_df.drop(
            self._filtered_nist_df.index[index_to_remove])

    def _count_features_in_mask(self):
        data_mask = list(self._data_mask)

        data_mask = [' ' if e == float(0) else '1' for e in data_mask]
        str_mask = ''.join(data_mask)
        data_mask = ' '.join(str_mask.split())
        number_of_features = len(data_mask.split())

        widths = []
        for s in data_mask.split():
            widths.append(len(s))

        mean_width = np.mean(widths)
        return number_of_features, mean_width

    def _create_mask_from_data(self):
        _mean = self._ccd.data.mean()
        mask = self._ccd.data.copy()
        mask[mask <= _mean] = 0
        mask[mask > _mean] = 1
        self._data_mask = mask

    def _create_mask_from_nist(self):
        features_in_data, mean_width = self._count_features_in_mask()
        feature_half_width = int((mean_width - 1) / 2.)

        self._get_nist_lines()
        sorted_nist_df = self.nist_df.sort_values('rel_int', ascending=False)
        self._filtered_nist_df = sorted_nist_df[
                           0:features_in_data].sort_values('wavelength')

        mask = np.zeros(len(self._ccd.data))
        wavelength_axis = self._linear_model(range(len(self._ccd.data)))

        for wavelength in self._filtered_nist_df.wavelength.values:
            arg = np.argmin(np.abs(wavelength_axis - wavelength))
            mask[arg - feature_half_width:arg + feature_half_width] = 1

        self._nist_mask = mask

    @staticmethod
    def _cross_correlate(reference_array, compared_array, mode='full'):
        if np.mean(reference_array) == 0. or np.mean(compared_array) == 0.:
            return 1000
        else:
            cross_correlation = signal.correlate(reference_array,
                                                 compared_array,
                                                 mode=mode)
            correlation_shifts = np.linspace(-int(len(cross_correlation) / 2.),
                                             int(len(cross_correlation) / 2.),
                                             len(cross_correlation))
            max_correlation_index = np.argmax(cross_correlation)
            # plt.title("Arrays")
            # plt.plot(reference_array, color='k')
            # plt.plot(compared_array, color='r')
            # plt.show()
            # plt.title("Cross Correlation {:f}".format(correlation_shifts[max_correlation_index]))
            # plt.plot(correlation_shifts, cross_correlation)
            # plt.show()
            return correlation_shifts[max_correlation_index]

    def _fit_linear_model(self, offset=0):
        self._linear_model = models.Linear1D()
        self._linear_fitter = fitting.LinearLSQFitter()
        pixel_axis = [0,
                      0.5 * len(self._ccd.data),
                      len(self._ccd.data)]
        wavelength_axis = [self._spec_dict['blue'].value,
                           self._spec_dict['center'].value,
                           self._spec_dict['red'].value]
        self._linear_model = self.__fit(self._linear_model,
                                        self._linear_fitter,
                                        pixel_axis,
                                        wavelength_axis)
        return self.__get_rms(pixel_axis, self._linear_model, wavelength_axis)

    def _fit_non_linear_model(self):

        angstrom = self._filtered_nist_df.wavelength.values
        pixel = self._linear_model.inverse(angstrom)

        for b in range(len(self._bin_limits) - 1):
            for a in range(len(angstrom)):
                if self._bin_limits[b] < pixel[a] < self._bin_limits[b + 1]:
                    pixel[a] = pixel[a] - self._local_shifts[b]

        if self._non_linear_model is None:
            self._non_linear_model = models.Chebyshev1D(degree=3)
            self._non_linear_model.c0.value = self._linear_model.intercept.value
            self._non_linear_model.c1.value = self._linear_model.slope.value

            self._non_linear_fitter = fitting.LevMarLSQFitter()
        self._non_linear_model = self.__fit(self._non_linear_model,
                                            self._non_linear_fitter,
                                            pixel,
                                            angstrom)
        return self.__get_rms(pixel, self._non_linear_model, angstrom)

    def _get_nist_lines(self):
        _nist_dir = 'data/nist/'
        df_list = []
        lamp_elements = self._ccd.header['OBJECT']
        if len(lamp_elements) % 2 == 0:
            for e in range(0, len(lamp_elements), 2):
                _file = 'nist_air_strong_lines_{:s}.txt' \
                        ''.format(lamp_elements[e:e+2])
                _full_file = os.path.join(_nist_dir, _file)
                if os.path.isfile(_full_file):
                    df = pd.read_csv(_full_file,
                                     names=['rel_int',
                                            'wavelength',
                                            'ion',
                                            'reference'])
                    filtered_df = df[((df.wavelength > self.spec['blue'].value) &
                                      (df.wavelength < self.spec['red'].value))]

                    if not filtered_df.empty:
                        df_list.append(filtered_df)
                else:
                    raise Exception(
                        "File {:s} does not exist".format(_full_file))
            if len(df_list) > 1:
                self.nist_df = pd.concat(df_list)
                self.nist_df = self.nist_df.sort_values('wavelength')
                self.nist_df = self.nist_df.reset_index(drop=True)
            elif len(df_list) == 1:
                self.nist_df = df_list[0]
            else:
                raise Exception("No NIST data was recovered")

        else:
            raise Exception("Wrong OBJECT keyword")

    def _get_peaks_in_data(self):
        serial_binning, parallel_binning = [
            int(x) for x in self._ccd.header['CCDSUM'].split()]
        slit_size = np.float(re.sub('[a-zA-Z" ]', '', self._ccd.header['slit']))
        no_nan_data = np.asarray(np.nan_to_num(self._ccd.data))

        filtered_data = np.where(np.abs(no_nan_data > no_nan_data.min() + 0.03 * no_nan_data.max()), no_nan_data, None)
        none_to_zero = [0 if i is None else i for i in filtered_data]
        filtered_data = np.array(none_to_zero)

        order = int(round(float(slit_size) /
                          (self._pixel_scale.value * serial_binning)))

        peaks = signal.argrelmax(filtered_data, axis=0, order=order)[0]
        return peaks

    def _get_spectral_limits(self):

        grating_frequency = float(re.sub('[A-Za-z_-]',
                                         '',
                                         self._ccd.header['GRATING'])) / u.mm

        grating_angle = float(self._ccd.header['GRT_ANG']) * u.deg
        camera_angle = float(self._ccd.header['CAM_ANG']) * u.deg

        serial_binning, parallel_binning = [
            int(x) for x in self._ccd.header['CCDSUM'].split()]

        self._pixel_size *= serial_binning

        pixel_count = len(self._ccd.data)

        alpha = grating_angle.to(u.rad)
        beta = camera_angle.to(u.rad) - grating_angle.to(u.rad)

        center_wavelength = (np.sin(alpha) +
                             np.sin(beta)) / grating_frequency
        center_wavelength = center_wavelength.to(u.angstrom)

        limit_angle = np.arctan(
            pixel_count *
            (self._pixel_size / self._focal_length) / 2)

        blue_limit = ((np.sin(alpha) +
                       np.sin(beta - limit_angle.to(u.rad))) /
                      grating_frequency).to(u.angstrom)

        red_limit = ((np.sin(alpha) +
                      np.sin(beta +
                             limit_angle.to(u.rad))) /
                     grating_frequency).to(u.angstrom)

        spectral_limits = {'center': center_wavelength,
                           'blue': blue_limit,
                           'red': red_limit,
                           'alpha': alpha,
                           'beta': beta}

        return spectral_limits

    def _global_cross_correlate(self):
        return self._cross_correlate(self._nist_mask, self._data_mask)

    def _local_cross_correlate(self):
        data_length = len(self._ccd.data)
        self._bin_limits = range(0,
                           data_length + 1,
                           int(np.floor(data_length / self._max_division)))
        all_shifts = []
        for e in range(len(self._bin_limits) - 1):
            new_shift = self._cross_correlate(
                self._nist_mask[self._bin_limits[e]:self._bin_limits[e+1]],
                self._data_mask[self._bin_limits[e]:self._bin_limits[e+1]])
            all_shifts.append(new_shift)
        clipped_shifts = sigma_clip(all_shifts, sigma=1, iters=2)
        final_shifts = self._repopulate_masked(masked_array=clipped_shifts)
        return final_shifts

    @staticmethod
    def _repopulate_masked(masked_array):
        x_axis = [i for i in range(len(masked_array.mask)) if masked_array.mask[i] == False]
        y_axis = [masked_array[i] for i in range(len(masked_array)) if masked_array.mask[i] == False]

        masked_index = [i for i in range(len(masked_array.mask)) if masked_array.mask[i] == True]

        _model = models.Polynomial1D(degree=2)
        _fitter = fitting.LevMarLSQFitter()
        _fitted_model = _fitter(_model, x_axis, y_axis)

        masked_array = [masked_array[i] if masked_array.mask[i] == False else _fitted_model(i) for i in range(len(masked_array))]
        return masked_array



    @property
    def nist_df(self):
        return self._nist_df

    @nist_df.setter
    def nist_df(self, value):
        if isinstance(value, pd.DataFrame):
            self._nist_df = value
        else:
            self._nist_df = None

    @property
    def data_mask(self):
        return self._data_mask

    @data_mask.setter
    def data_mask(self, value):
        if (len(value) == len(self._ccd.data)) and \
                (np.max(value) <= 1) and (np.min(value) >= 0):
            self._data_mask = value

    @property
    def spec(self):
        return self._spec_dict

    @spec.setter
    def spec(self, value):
        if isinstance(value, dict):
            self._spec_dict = value

    @property
    def file_name(self):
        return self._file_name

    @file_name.setter
    def file_name(self, value):
        if os.path.isfile(value):
            self._file_name = value

    @property
    def ccd(self):
        return self._ccd

    @ccd.setter
    def ccd(self, value):
        if isinstance(value, CCDData):
            self._ccd = value




if __name__ == '__main__':
    # _file = 'data/fits/goodman_comp_400M2_GG455_HgArNe.fits'
    _file = 'data/fits/goodman_comp_400M2_GG455_Ne.fits'

    wav = WavelengthCalibration()
    wav(spectrum=_file)
