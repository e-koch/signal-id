# Generate and manipulate three-d Boolean arrays for purposes of
# signal identification.

# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
# IMPORTS
# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

# python
import time
import copy
import logging
import warnings

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as nd
from astropy import units as u
from astropy.extern import six

# radio tools
from spectral_cube import SpectralCube, BooleanArrayMask, LazyMask
from spectral_cube.masks import MaskBase, is_broadcastable_and_smaller
from spectral_cube.wcs_utils import slice_wcs
from radio_beam import Beam

from .utils import get_pixel_scales

# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
# BASE CLASS
# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%


class RadioMask(MaskBase):
    """
    Create and manipulate a binary mask for a spectral datacube. `RadioMask`
    understands and contains the associated metadata, leveraging the abilities
    of the `SpectralCube <http://spectral-cube.readthedocs.org/en/latest/>`_
    package. `RadioMask` is intended for interactively manipulating masks and
    includes abilities to log and backup previous versions of the mask, as well
    as applying the changed mask to a `SpectralCube` object or writing the mask
    to disk. Most common morphological operations are availabled.


    Parameters
    ----------
    data : {SpectralCube, str, np.ndarray}
        Datacube to create the mask from.
    thresh : float, optional
        Threshold level used to create the mask.
    backup : bool, optional
        Enables a backup after performing an operation on the mask. Changes
        can then be reverted to the previous step.
    args : tuple or keyword arguments, optional
        Passed to loading functions. If `data` given as an array, a `WCS`
        object can be supplied by specifying `wcs`. If loading from a file,
        the file format can be supplied through `format`. `signal-id` supports
        the same formats as
        `spectral-cube <http://spectral-cube.readthedocs.org/en/latest/>`_.

    Example
    -------

    Starting from a `SpectralCube` object:
    >>> from spectral_cube import SpectralCube
    >>> from signal_id import RadioMask
    >>> cube = SpectralCube.read("test.fits")
    >>> mask = RadioMask(cube)
    By default, a backup of the mask is created whenever an operation is
    applied. That way, the previous version can be recovered.

    Operations can then be applied to the mask. For example, to perform a
    binary dilation then erosion,
    >>> mask.binary_dilation()
    >>> mask.binary_erosion()

    If the outcome of the erosion was unintended, the backup can be restored
    using:
    >>> mask.undo()

    The mask can be attached to a `SpectralCube` object using:
    >>> cube2 = mask.attach_to_cube()
    By default, this will use the `SpectralCube` object from
    `mask.linked_data`.

    The mask can be written using:
    >>> mask.write("test_mask.fits")



    """

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # Attributes and Properties
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    # Current version of the array
    _value = None

    # Previous version of the array
    _backup = None

    # Toggles
    _implicit_backup = True

    # The associated cube
    _linked_data = None

    # TBD: noise / SNR cube

    # TBD: beam

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # Construction
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    def __init__(self, data, thresh=None, backup=True, struct=None,
                 *args):

        if isinstance(data, SpectralCube):
                self.from_spec_cube(data, thresh=thresh, *args)

        elif isinstance(data, six.string_types):
            self.from_file(data, *args)

        elif isinstance(data, np.ndarray):
            self.from_array(data, *args)

        else:
            raise TypeError("Input of type %s is not accepted." % (type(data)))

        # Apply specified threshold to create an initial mask
        if thresh is not None:
            self._mask *= self.linked_data > thresh

        self.struct = struct

        self._get_shape()

        # Start log of method calls
        logging.basicConfig()
        self._log = logging.getLogger("method_calls")
        self._log.info("Object created on : ",
                       time.asctime(time.localtime(time.time())))

        # Switch on backup
        if backup:
            self.enable_backup()
        else:
            self.disable_backup()

    def _validate_wcs(self, data, wcs):
        pass

    def _include(self, data=None, wcs=None, view=()):
        self._validate_wcs(data, wcs)
        return self._mask._include(data=data, wcs=wcs, view=view)

    def __getitem__(self, view):
        return self._mask._include(view=view)

    def _iter_slices(self, axis):
        """
        Iterate over the cube one slice at a time,
        replacing masked elements with fill
        """
        view = [slice(None)] * 3
        for x in range(self.shape[axis]):
            view[axis] = x
            yield self._mask._include(view=view)

    def from_file(self, fname, thresh=None, format='fits'):
        cube = SpectralCube.read(fname, format=format)
        self.from_spec_cube(cube, thresh=None)

    def from_spec_cube(self, cube, thresh=None):
        self._linked_data = cube
        self._mask = cube.mask
        self._wcs = cube.wcs

    def from_array(self, array, thresh=None, wcs=None):
        self._linked_data = array
        self._mask = np.isfinite(array)
        self._wcs = wcs

    @property
    def linked_data(self):
        return self._linked_data

    @property
    def wcs(self):
        return self._wcs

    @property
    def shape(self):
        return self._shape

    def _get_shape(self):
        if isinstance(self._mask, LazyMask):
            ax_shape = 0
            view = [slice(None)] * 3
            while True:
                view[0] = ax_shape
                try:
                    arr = self.include(view=view)
                except IndexError:
                    break
                if ax_shape == 0:
                    shape = arr.shape
                ax_shape += 1

            self._shape = (ax_shape,) + shape

        else:
            self._shape = self._mask.shape

    @property
    def log(self):
        return self._log

    @property
    def struct(self):
        return self._struct

    @struct.setter
    def struct(self, input_struct):
        if input_struct is not None:
            if not isinstance(input_struct, np.ndarray):
                raise TypeError("input_struct must be a numpy array.")

            if input_struct.ndim < 2 or input_struct.ndim > 3:
                raise ValueError("struct must be 2- or 3-dimensional.")
            elif input_struct.ndim == 2:
                input_struct = input_struct[np.newaxis, :, :]

        self._struct = input_struct

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # Output
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    def copy(self):
        """
        Return a copy.
        """
        return copy.deepcopy(self)

    def as_spec_cube(self, dtype=bool):
        """
        Return a spectral cube.

        Parameters
        ----------
        dtype : valid data type, optional
            Specify the data type to be used in the mask in the
            SpectralCube object. Defaults to boolean.
        """
        if isinstance(self._linked_data, SpectralCube):
            return SpectralCube(self._mask.astype(dtype),
                                wcs=self._wcs)
        return SpectralCube(self._mask.astype(dtype),
                            wcs=self._wcs)

    def write(self, fname, dtype=int):
        """
        Write to a file. Default to using ints.

        Parameters
        ----------
        fname : str
            Name of the file to output to.
        dtype : valid data type, optional
            The data type the mask is converted to when writing out. Due to
            FITS not supporting boolean arrays, the default is 'int'.
        """
        # So wasteful...
        cube = self.as_spec_cube(dtype=dtype)
        cube.write(fname)

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # Expose the mask in various ways
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    def as_indices(self):
        """
        As a tuple of indices where True, useful for indexing.
        """
        return np.where(self._mask)

    def as_index_array(self, coordaxis=0):
        """
        As a numpy array of True indices. Optionally specify the axis for
        the coordinates (0 or 1)
        """
        if coordaxis == 0:
            return np.vstack(np.where(self._mask))
        else:
            return np.vstack(np.where(self._mask)).transpose()

    def oned(self, axis=0, sum=False):
        raise NotImplementedError()

    def twod(self, axis=0, sum_axis=False):
        """
        Return a two-dimensional version of the mask.
        """
        if self._mask.ndim == 2:
            return self._mask
        if sum_axis:
            return np.sum(self._mask, axis=axis)
        else:
            return np.max(self._mask, axis=axis)

    def independent_channels(self, struct=None):
        raise NotImplementedError()

    def independent_areas(self, struct=None):
        raise NotImplementedError()

    def independent_cubes(self, struct=None):
        raise NotImplementedError()

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # Undo/Redo
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    def enable_backup(self):
        self.is_backup_enabled = True
        if not hasattr(self, '_backup'):
            self._backup = None

    def disable_backup(self):
        self.is_backup_enabled = False

    def log_and_backup(self, func):
        '''
        Back-up and log method calls.
        '''
        self._log.info(func.__name__)
        if self.is_backup_enabled:
            self._backup = self._mask.copy()

    def undo(self):
        self._log.info("UNDO")
        temp = self._backup.copy()
        self._backup = self._mask.copy()
        self._mask = temp

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # Operators
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    def _precheck_operator(self, other, wcs=None):
        if isinstance(other, np.ndarray):
            if wcs is None:
                raise TypeError("wcs object must be given when other is "
                                "a numpy array.")
            other = BooleanArrayMask(other, wcs)
        # Check if arrays are broadcastable
        if not is_broadcastable_and_smaller(self.shape, other.shape):
            raise ValueError("Mask shapes are not broadcastable.")

        return other

    # Union
    def union(self, other, wcs=None):
        self.log_and_backup(self.union)
        self._precheck_operator(other, wcs=wcs)
        self._mask = self._mask | other

    # Intersection
    def intersection(self, other, wcs=None):
        self.log_and_backup(self.intersection)
        self._precheck_operator(other, wcs=wcs)
        self._mask = self._mask and other

    # Exclusive or
    def xor(self, other, wcs=None):
        self.log_and_backup(self.xor)
        self._precheck_operator(other, wcs=wcs)
        self._mask = self._mask ^ other

    # Inversion
    def invert(self):
        self.log_and_backup(self.invert)
        self._mask = ~self._mask

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # Manipulation
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    def _check_use_struct(self, struct):
        '''
        Use global structure if another structure is not given.
        Raise error when no global structure is defined.
        '''

        if struct is None:
            if self.struct is None:
                warnings.warn("There are no defined structure elements. "
                              "The default square-connectivity element will "
                              "be used.")
            return self.struct
        else:
            return struct

    def _apply_manipulation(self, function, args=(), kwargs={},
                            iteraxis=None):
        '''
        Function for general mask manipulations. Returns new mask
        as a BooleanArrayMask.
        '''

        if iteraxis is not None:
            struct = self.struct.squeeze()
            kwargs['structure'] = struct

            # Add the sliced dimension back in to restore cube shape
            add_axis = [slice(None)] * 3
            add_axis[iteraxis] = np.newaxis

            for i, plane in enumerate(self._iter_slices(iteraxis)):

                out_plane = function(plane, *args, **kwargs)

                if i == 0:
                    output = out_plane[add_axis]
                else:
                    output = np.append(output, out_plane[add_axis],
                                       axis=iteraxis)

        else:
            kwargs['structure'] = self.struct
            output = function(self._mask.include(), *args, **kwargs)

        self._mask = BooleanArrayMask(output, self.wcs)

    # Dilation
    def dilate(self, struct=None, iterations=1, iteraxis=None):
        struct = self._check_use_struct(struct)
        self.log_and_backup(self.dilate)
        self._mask = nd.binary_dilation(self._mask, structure=struct,
                                        iterations=iterations)

    # Erosion
    def erode(self, struct=None, iterations=1, iteraxis=None):
        struct = self._check_use_struct(struct)
        self.log_and_backup(self.erode)
        self._mask = nd.binary_erosion(self._mask, structure=struct,
                                       iterations=iterations)

    # Opening
    def open(self, struct=None, iterations=1, iteraxis=None):
        struct = self._check_use_struct(struct)
        self.log_and_backup(self.open)
        self._mask = nd.binary_opening(self._mask, structure=struct,
                                       iterations=iterations)

    # Closing
    def close(self, struct=None, iterations=1, iteraxis=None):
        struct = self._check_use_struct(struct)
        self.log_and_backup(self.close)
        self._mask = nd.binary_closing(self._mask, structure=struct,
                                       iterations=iterations)

    def remove_small_regions(self, area_threshold=None, beam=None,
                             verbose=False):
        '''
        Remove 2D regions (per channel) based on their area. By default, this
        removed regions smaller than the beam area.

        Parameters
        ----------
        area_threshold : float, optional
            Minimum pixel area to keep. Overrides beam argument.
        beam : radio_beam.Beam, optional
            Provide a Beam object to define the area threshold. By default,
            a Beam object will be created from the information in the cube
            WCS. Specifying a Beam object will override using the default
            Beam object from the cube WCS.

        '''

        self.log_and_backup(self.remove_small_regions)

        # Attempt to get beam area from cube WCS info.
        if area_threshold is None:
            if beam is None:
                beam = Beam.from_fits_header(self._linked_data.header)

            # Get pixelscale, add unit on (ignore the per pixel)
            # Eventually the unit should be included in get_pixel_scales
            # but this requires additional workarounds when used for Beam
            # objects
            pixscale = get_pixel_scales(self._wcs) * u.deg
            # Now get the pixel beam area
            area_threshold = (beam.sr.to(u.deg**2) / pixscale**2.).value

        def area_thresh_func(arr, size_thresh):
            label_arr, num = nd.label(arr, np.ones((3, 3)))

            pixel_area = nd.sum(arr, label_arr, range(1, num+1))

            remove_labels = np.where(pixel_area < size_thresh)[0]

            for lab in remove_labels:
                arr[np.where(label_arr == lab)] = 0

            return arr

        self.reject_region(area_thresh_func, iteraxis='spectral',
                           func_args=(area_threshold),
                           log_call=False, verbose=verbose)

        return self

    # Reject on property
    def reject_region(self, func, iteraxis='spectral', func_args=(),
                      log_call=True, verbose=False):
        '''
        Remove 2D regions from the mask based on the given criteria.
        The specified function should operate on two-dimensional planes.
        The axis to be iterated over can be specified using the iteraxis
        argument. Additional inputs should be specified as a tuple in the args
        argument.

        Parameters
        ----------
        func : function
            Contains rejection criteria operating on 2D planes.
        iteraxis : int or 'spectral', optional
            Axis to iterate over. This defaults to the spectral axis.
        func_args : tuple, optional
            Arguments passed to func.
        log_call : bool, optional
            Turns off the logging, since this function is called within
            remove_small_regions.
        verbose : bool, optional
            Enabling prints out the plane currently being altered.
        '''
        if log_call:
            self.log_and_backup(self.reject_region)

        if iteraxis == 'spectral':
            iteraxis = self._wcs.naxis - \
                self._wcs.wcs.spec - 1
        elif isinstance(iteraxis, int):
            pass
        else:
            raise TypeError("iteraxis must be an integer or 'spectral'.")

        if not isinstance(func_args, tuple):
            func_args = (func_args, )

        nplanes = self.shape[0]
        # Now iterate through the planes
        for i, plane in enumerate(self._iter_slices(0)):
            if verbose:
                print("On plane "+str(i)+" of "+str(nplanes))
            plane_slice[iteraxis] = plane
            self._mask[plane_slice] = \
                func(self._mask[plane_slice], *func_args)

    # Reject on volume (special case)
    def reject_on_volume(self, thresh=None, struct=None):
        # self.log_and_backup(self.reject_volume)
        raise NotImplementedError()

    def apply_custom_func(self, func, *args):
        self.log_and_backup(self.apply_custom_func)
        self._mask = func(self._mask, *args)

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # Mask generation
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    # Structured thresholding
    def threshold(self, thresh=None, struct=None):
        raise NotImplementedError()

    # Projected 2d prior ("drop down" a twod mask)

    # Projected 3d prior ("inflate" a velocity field)

    # Define line-free channels

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # RECIPES
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    # Convolve-then-threshold (2+1d)

    # High-reject-grow-low

    # Autotune thresholding

# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
# STRUCTURING ELEMENTS
# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
# VISUALIZATION AND DIAGNOSTICS
# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

