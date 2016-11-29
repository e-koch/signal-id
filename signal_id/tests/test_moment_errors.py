
'''
Moment errors testing
'''

import pytest
import numpy.testing as npt

import numpy as np
from spectral_cube import SpectralCube
from astropy.wcs import WCS
from astropy.io import fits

from signal_id._moment_errs import (moment0_error, moment1_error,
                                    moment2_error, linewidth_sigma_err,
                                    linewidth_fwhm_err)


def moment_cube_and_errs():

    cube = np.array([2, 1, 1, 1]).reshape((4, 1, 1)).astype(np.float)
    cube_errs = \
        np.array([0.5, 0.5, 0.25, 0.25]).reshape((4, 1, 1)).astype(np.float)

    wcs = WCS(naxis=3)
    wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN', 'VELO']
    # choose values to minimize spherical distortions
    wcs.wcs.cdelt = np.array([1, 1, 1], dtype='float32')
    wcs.wcs.crpix = np.array([0, 0, 0], dtype='float32')
    wcs.wcs.crval = np.array([0, 0, 0], dtype='float32')
    wcs.wcs.cunit = ['deg', 'deg', 'm/s']

    header = wcs.to_header()
    header['BUNIT'] = 'K'

    hdu = fits.PrimaryHDU(data=cube, header=header)
    hdu_err = fits.PrimaryHDU(data=cube_errs, header=header)
    return hdu, hdu_err


def wt_moment(data, wt, errors=None, oversample=1.):
    """

    Borrowing from Jonathan Foster's RAMPS scripts:
    https://github.com/jfoster17/ramps

    Using to verify, since this function is verified against the IDL version.

    Adapted to python from wt_moment.pro by Erik Rosolowksy
    Verified to be consistent with that code on 2010-12-03
    In [33]: wm.wt_moment([1,2,3,4],[2,1,1,1],errors=[0.5,0.5,0.25,0.25],oversample=2.)
    Out[33]:
    {'errmn': 0.22135943621178658,
     'errsd': 0.10076910123994302,
     'mean': 2.2000000000000002,
     'stdev': 1.16619037896906}
    IDL> ya = wt_moment([1,2,3,4],[2,1,1,1],errors=[0.5,0.5,0.25,0.25],oversample=2)
    IDL> help,ya,/struct
    ** Structure <1db9344>, 4 tags, length=16, data length=16, refs=1:
       MEAN            FLOAT           2.20000
       STDEV           FLOAT           1.16619
       ERRMN           FLOAT          0.221359
       ERRSD           FLOAT          0.100769


    The output values using oversample=1 are:
    {'errmn': 0.15652475842498528,
     'errsd': 0.10076910123994302,
     'mean': 2.2000000000000002,
     'stdev': 1.16619037896906}
    """

    # wt = np.array(wt, dtype='float64')
    # data = np.array(data, dtype='float64')
    # errors = np.array(errors, dtype='float64')

    osf = np.sqrt(oversample)  # Note this doesn't change anything for osf=1
    tot = np.sum(wt)
    mean = np.sum(wt * data) / tot
    stdev = np.sqrt(np.abs(np.sum((data - mean)**2 * wt) / tot))
    if errors is not None:
        mean_err = np.sqrt(np.sum(((tot * data - np.sum(wt * data)) /
                                   (tot**2))**2 * errors**2)) * osf

        sig2err = \
            np.sqrt(np.sum(((tot * (data - mean)**2 -
                             np.sum(wt * (data - mean)**2)) / tot**2)**2 *
                           errors**2) +
                    (2 * np.sum(wt * (data - mean)) / tot)**2 * mean_err**2)
        sd_err = 1. / (2 * stdev) * sig2err * osf
        return({"mean": mean, "stdev": stdev, "errmn": mean_err,
                "errsd": sd_err, "sig2err": sig2err})
    else:
        return({"mean": mean, "stdev": stdev})


def comparison_values():

    hdu, hdu_err = moment_cube_and_errs()

    test_cube = SpectralCube.read(hdu)
    test_cube_err = SpectralCube.read(hdu_err)

    out = wt_moment(test_cube.spectral_axis,
                    test_cube.filled_data[:].squeeze(),
                    errors=test_cube_err.filled_data[:].squeeze())

    # Add in the moment 0 error.
    mom0_err = test_cube._pix_size_slice(2) * \
        np.sqrt(((test_cube_err ** 2).sum()))

    out["mom0_err"] = mom0_err

    return out


def test_slice_errs():

    hdu, hdu_err = moment_cube_and_errs()

    test_cube = SpectralCube.read(hdu)
    test_cube_err = SpectralCube.read(hdu_err)

    comp_vals = comparison_values()

    mom0_err = moment0_error(test_cube, test_cube_err, axis=0, how='slice')
    mom1_err = moment1_error(test_cube, test_cube_err, axis=0, how='slice')
    mom2_err = moment2_error(test_cube, test_cube_err, axis=0, how='slice')

    assert mom0_err == comp_vals["mom0_err"]
    assert mom1_err == comp_vals["errmn"]
    assert mom2_err == comp_vals["sig2err"]


def test_cube_errs():

    hdu, hdu_err = moment_cube_and_errs()

    test_cube = SpectralCube.read(hdu)
    test_cube_err = SpectralCube.read(hdu_err)

    comp_vals = comparison_values()

    mom0_err = moment0_error(test_cube, test_cube_err, axis=0, how='cube')
    mom1_err = moment1_error(test_cube, test_cube_err, axis=0, how='cube')
    mom2_err = moment2_error(test_cube, test_cube_err, axis=0, how='cube')

    assert mom0_err == comp_vals["mom0_err"]
    assert mom1_err == comp_vals["errmn"]
    assert mom2_err == comp_vals["sig2err"]


def test_linewidth_sigma_err():
    '''
    '''

    hdu, hdu_err = moment_cube_and_errs()

    test_cube = SpectralCube.read(hdu)
    test_cube_err = SpectralCube.read(hdu_err)

    comp_vals = comparison_values()

    sig_cube = linewidth_sigma_err(test_cube, test_cube_err, how='cube')
    sig_slice = linewidth_sigma_err(test_cube, test_cube_err, how='slice')

    npt.assert_almost_equal(sig_cube.squeeze().value, comp_vals['errsd'].value)
    npt.assert_almost_equal(sig_slice.squeeze().value,
                            comp_vals['errsd'].value)


def test_linewidth_fwhm_err():
    '''
    '''

    SIGMA2FWHM = 2. * np.sqrt(2. * np.log(2.))

    hdu, hdu_err = moment_cube_and_errs()

    test_cube = SpectralCube.read(hdu)
    test_cube_err = SpectralCube.read(hdu_err)

    comp_vals = comparison_values()

    sig_cube = linewidth_fwhm_err(test_cube, test_cube_err, how='cube')
    sig_slice = linewidth_fwhm_err(test_cube, test_cube_err, how='slice')

    npt.assert_almost_equal(sig_cube.squeeze().value,
                            comp_vals['errsd'].value * SIGMA2FWHM)
    npt.assert_almost_equal(sig_slice.squeeze().value,
                            comp_vals['errsd'].value * SIGMA2FWHM)
