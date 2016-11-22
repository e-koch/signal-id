
import pytest
import numpy as np
from numpy.testing import assert_allclose
from astropy.wcs import WCS
from astropy.io import fits
from spectral_cube import SpectralCube

from ..mask import RadioMask


def radiomask():
    test_mask = np.ones((5, 5)).astype(bool)
    test_mask[:1] = False
    test_mask[-1:] = False
    test_mask[:, -1] = False
    test_mask[:, 0] = False

    test_mask = test_mask[np.newaxis, :, :]

    test_data = np.ones_like(test_mask) * 5.0

    wcs = WCS(naxis=3)
    wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN', 'VELO']
    # choose values to minimize spherical distortions
    wcs.wcs.cdelt = np.array([-1, 2, 3], dtype='float32') / 1e5
    wcs.wcs.crpix = np.array([1, 1, 1], dtype='float32')
    wcs.wcs.crval = np.array([0, 1e-3, 2e-3], dtype='float32')
    wcs.wcs.cunit = ['deg', 'deg', 'km/s']
    header = wcs.to_header()
    header['BUNIT'] = 'K'

    hdu = fits.PrimaryHDU(data=test_data, header=header)

    cube = SpectralCube.read(hdu)
    cube = cube.with_mask(test_mask)

    return RadioMask(cube), test_data, wcs, test_mask


def test_mask_props():

    mask, data, wcs, test_mask = radiomask()

    filled = data * test_mask
    filled[filled == 0] = np.NaN

    flattened = filled[np.isfinite(filled)]

    assert_allclose(mask.include(data, wcs), test_mask)
    assert_allclose(mask.exclude(data, wcs), ~test_mask)
    assert_allclose(mask._filled(data, wcs), filled)
    assert_allclose(mask._flattened(data, wcs), flattened)

    assert_allclose(m.include(data, wcs, view=(0, 0, slice(1, 4))), [1, 1, 0])
    assert_allclose(m.exclude(data, wcs, view=(0, 0, slice(1, 4))), [0, 0, 1])
    assert_allclose(m._filled(data, wcs, view=(0, 0, slice(1, 4))), [1, 2, np.nan])
    assert_allclose(m._flattened(data, wcs, view=(0, 0, slice(1, 4))), [1, 2])