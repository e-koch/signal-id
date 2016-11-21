
import numpy as np

from spectral_cube._moments import _moment_shp
# from spectral_cube.np_compat import allbadtonan

'''
Functions for making moment error maps.

Borrows heavily from the functionality in _moments.py from spectral-cube.

Functions require, at minimum, a SpectralCube object and a scale value that
characterizes the noise.
'''


def _slice0(cube, axis, scale):
    """
    0th moment along an axis, calculated slicewise

    Parameters
    ----------
    cube : SpectralCube
    axis : int
    scale : float

    Returns
    -------
    moment0_error : array
    """
    shp = _moment_shp(cube, axis)
    result = np.zeros(shp)

    view = [slice(None)] * 3

    valid = np.zeros(shp, dtype=np.bool)
    for i in range(cube.shape[axis]):
        view[axis] = i
        plane = cube._mask.include(data=cube._data, wcs=cube._wcs, view=view)
        valid |= plane
        result += plane
    result = scale * np.sqrt(result)
    result[~valid] = np.nan
    return result


def _slice1(cube, axis, scale, moment0, moment1):
    """
    1st moment along an axis, calculated slicewise

    Parameters
    ----------
    cube : SpectralCube
    axis : int
    scale : float
    moment0 : 0th moment
    moment1 : 1st moment

    Returns
    -------
    moment1_error : array
    """
    shp = _moment_shp(cube, axis)
    result = np.zeros(shp)

    view = [slice(None)] * 3
    pix_cen = cube._pix_cen()[axis]

    for i in range(cube.shape[axis]):
        view[axis] = i
        result += np.power((pix_cen[view] - moment1)/moment0, 2)
    return (scale / moment0) * np.sqrt(result)


def _slice2(cube, axis, scale, moment0, moment1, moment2,
            moment1_err):
    """
    2nd moment error along an axis, calculated slicewise

    Parameters
    ----------
    cube : SpectralCube
    axis : int
    scale : float
    moment0 : 0th moment
    moment1 : 1st moment
    moment2 : 2nd moment
    moment1_err : 1st moment error

    Returns
    -------
    moment1_error : array
    """
    shp = _moment_shp(cube, axis)
    term1 = np.zeros(shp)
    term2 = np.zeros(shp)

    view = [slice(None)] * 3
    pix_cen = cube._pix_cen()[axis]

    for i in range(cube.shape[axis]):
        view[axis] = i
        plane = cube._mask.include(data=cube._data, wcs=cube._wcs, view=view)
        term1 += scale**2 * \
            np.power((np.power((pix_cen[view] - moment1), 2) -
                     moment2), 2)

        term2 += plane * (pix_cen - moment1)

    return (1/moment0) * np.sqrt(term1 + 4*np.power(moment1_err*term2, 2))


def moment_slicewise(cube, order, axis, scale, moment0=None, moment1=None,
                     moment2=None):
    """
    Compute moments by accumulating the result 1 slice at a time
    """
    if order == 0:
        return _slice0(cube, axis, scale)
    if order >= 1:
        if moment0 is None or moment1 is None:
            raise TypeError("Both moment0 and moment1 must be specified for"
                            "the moment1 error.")
        if order == 1:
            return _slice1(cube, axis, moment0, moment1)
        else:
            mom1_err = _slice1(cube, axis, moment0, moment1)

    shp = _moment_shp(cube, axis)
    result = np.zeros(shp)

    view = [slice(None)] * 3
    pix_size = cube._pix_size()[axis]
    pix_cen = cube._pix_cen()[axis]
    weights = np.zeros(shp)

    for i in range(cube.shape[axis]):
        view[axis] = i
        plane = cube._get_filled_data(fill=0, view=view)
        result += (plane *
                   (pix_cen[view] - mom1) ** order *
                   pix_size[view])
        weights += plane * pix_size[view]

    return (result / weights)


def moment_raywise(cube, order, axis):
    """
    Compute moments by accumulating the answer one ray at a time
    """
    shp = _moment_shp(cube, axis)
    out = np.zeros(shp) * np.nan

    pix_cen = cube._pix_cen()[axis]
    pix_size = cube._pix_size()[axis]

    for x, y, slc in cube._iter_rays(axis):
        # the intensity, i.e. the weights
        include = cube._mask.include(data=cube._data, wcs=cube._wcs,
                                     view=slc)
        if not include.any():
            continue

        data = cube.flattened(slc).value * pix_size[slc][include]

        if order == 0:
            out[x, y] = data.sum()
            continue

        order1 = (data * pix_cen[slc][include]).sum() / data.sum()
        if order == 1:
            out[x, y] = order1
            continue

        ordern = (data * (pix_cen[slc][include] - order1) ** order).sum()
        ordern /= data.sum()

        out[x, y] = ordern
    return out


def moment_cubewise(cube, order, axis):
    """
    Compute the moments by working with the entire data at once
    """

    pix_cen = cube._pix_cen()[axis]
    data = cube._get_filled_data() * cube._pix_size()[axis]

    if order == 0:
        return allbadtonan(np.nansum)(data, axis=axis)

    if order == 1:
        return (np.nansum(data * pix_cen, axis=axis) /
                np.nansum(data, axis=axis))
    else:
        mom1 = moment_cubewise(cube, 1, axis)

        # insert an axis so it broadcasts properly
        shp = list(_moment_shp(cube, axis))
        shp.insert(axis, 1)
        mom1 = mom1.reshape(shp)

        return (np.nansum(data * (pix_cen - mom1) ** order, axis=axis) /
                np.nansum(data, axis=axis))
