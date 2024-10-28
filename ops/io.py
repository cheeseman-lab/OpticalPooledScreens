"""
Image Processing and File Handling Utilities
This module provides a comprehensive set of functions for file I/O. It includes functions for:

1. File Format Conversion: Converting between various image formats (e.g., ND2 to TIFF).
2. Image Reading and Writing: Functions to read and write images in different formats (TIFF, HDF5, Zarr).
3. Image Manipulation: Tools for slicing, tiling, and grid view generation of image data.
4. Metadata Handling: Functions for creating and manipulating ImageJ-compatible metadata.
5. Color Management: Utilities for handling color Look-Up Tables (LUTs) and display ranges.
6. Multi-dimensional Data Processing: Support for handling multi-channel and time-series image data.
7. Stitching and Tiling: Functions to assist with image stitching and tiling operations.
8. Performance Optimization: Options for memoization and parallel processing to improve performance.

"""

import numpy as np
import os
import pandas as pd
import six
import struct
import warnings

# Importing utility functions from 'ops.utils', constants from 'ops.constants',
# and filename handling functions from 'ops.filenames'
import ops.utils
import ops.constants
import ops.filenames

# Importing image reading functions from external modules
from ops.external.tifffile_new import imread
# Importing image saving function from an older version of the TIFF file handling module
from ops.external.tifffile_old import imsave

# Importing ND2Reader for reading ND2 files
from nd2reader import ND2Reader

# Importing tables for HDF5 file handling
import tables
# Importing zarr (commented out, but usually used for Zarr file format handling)
# import zarr

# ImageJ metadata description format
imagej_description = ''.join(['ImageJ=1.49v\nimages=%d\nchannels=%d\nslices=%d',
                              '\nframes=%d\nhyperstack=true\nmode=composite',
                              '\nunit=\\u00B5m\nspacing=8.0\nloop=false\n',
                              'min=764.0\nmax=38220.0\n'])

# Color LUTs (Look-Up Tables) for image visualization
ramp = list(range(256))
ZERO = [0]*256
RED     = ramp + ZERO + ZERO
GREEN   = ZERO + ramp + ZERO
BLUE    = ZERO + ZERO + ramp
MAGENTA = ramp + ZERO + ramp
GRAY    = ramp + ramp + ramp
CYAN    = ZERO + ramp + ramp

# Default color LUTs
DEFAULT_LUTS = GRAY, GREEN, RED, MAGENTA, CYAN, GRAY, GRAY

def read_lut(lut_string):
    """Reads a LUT (Look-Up Table) from a string and returns it as a flattened numpy array."""
    return (pd.read_csv(six.StringIO(lut_string), sep='\s+', header=None)
        .values.T.flatten())

# Read the GLASBEY inverted LUT
GLASBEY = read_lut(ops.constants.GLASBEY_INVERTED)

def save_zarr(filename, image, chunks=False, compressor=None, store_type='default'):
    """Saves an image to a Zarr file with optional compression and chunking.
    
    Args:
        filename (str): Path to the Zarr file.
        image (np.ndarray): Image data to be saved.
        chunks (tuple or bool): Chunk sizes for Zarr storage.
        compressor (zarr.Compressor): Optional compressor for Zarr storage.
        store_type (str): Type of Zarr store ('default' or 'lmdb').
    """
    if store_type == 'default':
        z = zarr.open(filename, mode='w', shape=image.shape, chunks=chunks, compressor=compressor)
        z[:] = image
    elif store_type == 'lmdb':
        with zarr.LMDBStore(filename) as store:
            z = zarr.open(store=store, mode='w', shape=image.shape, chunks=chunks, compressor=compressor)
            z[:] = image
    else:
        print('zarr store type unknown or not implemented')

def read_zarr(filename, bbox=None, memoize=False):
    """Reads an image from a Zarr file with optional bounding box and memoization.

    Args:
        filename (str): Path to the Zarr file.
        bbox (tuple): Optional bounding box (i0, j0, i1, j1) to read a subset of the image.
        memoize (bool): If True, enables caching of the Zarr file.

    Returns:
        np.ndarray: The image data read from the file.
    """
    if memoize:
        open_zarr_file.keys['active'] = True

    store = open_zarr_store(filename)
    array = zarr.open(store=store, mode='r')
    if bbox is not None:
        # Check if bbox is in image bounds
        i0, j0 = max(bbox[0], 0), max(bbox[1], 0)
        i1, j1 = min(bbox[2], array.shape[-2]), min(bbox[3], array.shape[-1])
        image = array[..., i0:i1, j0:j1]
    else:
        image = array[:]
    return image

def slice_array(array, bbox):
    """Slices a 4D array using a bounding box.

    Args:
        array (np.ndarray): Input array.
        bbox (tuple): Bounding box (i0, j0, i1, j1) for slicing.

    Returns:
        np.ndarray: Sliced array.
    """
    i0, j0 = max(bbox[0], 0), max(bbox[1], 0)
    i1, j1 = min(bbox[2], array.shape[-2]), min(bbox[3], array.shape[-1])
    return array[..., i0:i1, j0:j1]

@ops.utils.memoize(active=False, copy_numpy=False)
def open_zarr_store(filename):
    """Opens a Zarr file store for reading.

    Args:
        filename (str): Path to the Zarr file.

    Returns:
        zarr.Store: Zarr store object.
    """
    if filename.split('.')[-1] == 'lmdb':
        store = zarr.LMDBStore(filename, buffers=True)
    else:
        store = zarr.DirectoryStore(filename)
    return store

def nd2_to_tif(file, mag='10X', zproject=False, fov_axes='cxy', n_threads=1, tqdm=False, file_pattern=None, sites='all'):
    """Converts ND2 files to TIFF format and saves them, with options for z-projection and multithreading.

    Args:
        file (str): Path to the ND2 file.
        mag (str): Magnification information for the TIFF metadata.
        zproject (bool): If True, performs z-projection.
        fov_axes (str): Axes for the field of view in ND2 data.
        n_threads (int): Number of threads for parallel processing.
        tqdm (bool): If True, displays a progress bar.
        file_pattern (list): List of regular expression patterns for filename parsing.
        sites (str or list): Specifies which sites to process ('all' or specific site indices).

    Returns:
        dict: Metadata for each processed file.
    """
    if file_pattern is None:
        file_pattern = [
            (r'(?P<cycle>c[0-9]+)?/?'
            '(?P<dataset>.*)?/?'
            'Well(?P<well>[A-H][0-9]*)_'
            '(Point[A-H][0-9]+_(?P<site>[0-9]*)_)?'
            'Channel((?P<channel_1>[^_,]+)(_[^,]*)?)?,?'
            '((?P<channel_2>[^_,]+)(_[^,]*)?)?,?'
            '((?P<channel_3>[^_,]+)(_[^,]*)?)?,?'
            '((?P<channel_4>[^_,]+)(_[^,]*)?)?,?'
            '((?P<channel_5>[^_,]+)(_[^,]*)?)?,?'
            '((?P<channel_6>[^_,]+)(_[^_]*)?)?'
            '_Seq([0-9]+).nd2')
        ]

    description = ops.filenames.parse_filename(file, custom_patterns=file_pattern)
    description['ext'] = 'tif'
    description['mag'] = mag

    if 'site' in description.keys():
        description['site'] = str(int(description['site']))

    try:
        description['subdir'] = 'preprocess/' + description['cycle']
    except:
        description['subdir'] = 'preprocess'
    description['dataset'] = None

    channels = [ch for key, ch in description.items() if key.startswith('channel')]

    if len(channels) == 1:
        fov_axes = fov_axes.replace('c', '')

    def process_site(site, image):
        if zproject:
            z_axis = fov_axes.find('z')
            image = image.max(axis=z_axis)
        filename = ops.filenames.name_file(description, site=str(site))
        save_stack(filename, image[:])

        metadata = {'filename': filename,
                    'field_of_view': site,
                    'x': image.metadata['x_data'][site],
                    'y': image.metadata['y_data'][site],
                    'z': image.metadata['z_data'][site],
                    'pfs_offset': image.metadata['pfs_offset'][site],
                    'pixel_size': image.metadata['pixel_microns']}
        return metadata

    with ND2Reader(file) as images:
        try:
            images.iter_axes = 'v'
        except:
            images.bundle_axes = fov_axes
            if zproject:
                z_axis = fov_axes.find('z')
                images = images.max(axis=z_axis)
            filename = ops.filenames.name_file(description)
            save_stack(filename, images.get_frame(0))

            metadata = {'filename': filename,
                        'field_of_view': description['site'],
                        'x': images.metadata['x_data'][0],
                        'y': images.metadata['y_data'][0],
                        'z': images.metadata['z_data'],
                        'pfs_offset': images.metadata['pfs_offset'][0],
                        'pixel_size': images.metadata['pixel_microns']}
            return metadata

        images.bundle_axes = fov_axes

        if sites == 'all':
            sites = slice(None)

        if tqdm:
            import tqdm.notebook
            tqdn = tqdm.notebook.tqdm
            work = tqdn(zip(images.metadata['fields_of_view'][sites], images[sites]))
        else:
            work = zip(images.metadata['fields_of_view'][sites], images[sites])

        if n_threads != 1:
            from joblib import Parallel, delayed
            well_metadata = Parallel(n_jobs=n_threads, backend='threading')(delayed(process_site)(site, image) 
                for site, image in work)
        else:
            well_metadata = []
            for site, image in work:
                well_metadata.append(process_site(site, image))

    metadata_filename = ops.filenames.name_file(description, subdir='metadata', tag='metadata', ext='pkl')
    pd.DataFrame(well_metadata).to_pickle(metadata_filename)

def tile_config(df, output_filename):
    """Generates a tile configuration file for FIJI grid collection stitching from a DataFrame of site positions.

    Args:
        df (pd.DataFrame): DataFrame containing site position information.
        output_filename (str): Path to the output configuration file.
    """
    config = ['dim = 2\n\n']
    df = df.pipe(ops.plates.to_pixel_xy)
    for filename, x, y in zip(df.filename.tolist(), df.x_px.tolist(), df.y_px.tolist()):
        config.append(filename + "; ; (" + str(y) + ", " + str(-x) + ")\n")
    with open(output_filename, 'x') as f:
        f.writelines(config)

def grid_view(files, bounds, padding=40, with_mask=False, im_func=None, memoize=True):
    """Generates a grid view of images from given files and bounds.
    
    Args:
        files (list): List of filenames to read images from.
        bounds (list): List of bounding boxes for each file.
        padding (int): Padding to add around each image. Default is 40.
        with_mask (bool): If True, generates a mask image. Default is False.
        im_func (function): Function to apply to each image read from file. Default is None.
        memoize (bool): If True, caches file reads. Default is True.
    Returns:
        np.ndarray: Stacked image array.
        np.ndarray (optional): Stacked mask array if with_mask is True.
    """
    padding = int(padding)
    arr = []
    Is = {}  # Cache for loaded images
    
    if im_func is None:
        im_func = lambda x: x
        
    for filename, bounds_ in zip(files, bounds):
        if filename.endswith('hdf'):
            bounds_ = np.array(bounds_) + np.array((-padding, -padding, padding, padding))
            I_cell = im_func(read_hdf_image(filename, bbox=bounds_, memoize=memoize))
        else:
            try:
                I = Is[filename]
            except KeyError:
                I = im_func(read_stack(filename, copy=False))
                Is[filename] = I
            I_cell = ops.utils.subimage(I, bounds_, pad=padding)
        arr.append(I_cell.copy())

    if with_mask:
        arr_m = []
        for i, (i0, j0, i1, j1) in enumerate(bounds):
            shape = (i1 - i0 + padding, j1 - j0 + padding)
            img = np.zeros(shape, dtype=np.uint16) + i + 1
            arr_m.append(img)
        return ops.utils.pile(arr), ops.utils.pile(arr_m)
        
    return ops.utils.pile(arr)

def grid_view_timelapse(filename, frames, bounds, xy_shape=(80, 80), memoize=False):
    """Generates a grid view timelapse of images from an HDF file.

    Args:
        filename (str): Path to the HDF file.
        frames (list): List of frame indices to read.
        bounds (list): List of bounding boxes for each frame.
        xy_shape (tuple): Shape of each image in (height, width). Default is (80, 80).
        memoize (bool): If True, caches file reads. Default is False.

    Returns:
        np.ndarray: 4D array with timelapse images.
    """
    max_cells = max(len(bound) for bound in bounds)
    I = np.zeros((len(frames), 1, xy_shape[0], xy_shape[1] * max_cells))

    for frame_count, (frame, timelapse_bounds) in enumerate(zip(frames, bounds)):
        leading_dims = (slice(frame, frame + 1), slice(None))
        for num, bound in enumerate(timelapse_bounds):
            data = read_hdf_image(filename, leading_dims=leading_dims, bbox=bound, memoize=memoize)
            I[(slice(frame_count, frame_count + 1), slice(None)) + (slice(0, data.shape[-2]), slice(xy_shape[0] * num, xy_shape[0] * num + data.shape[-1]))] = data
    return I

def format_input(input_table, n_jobs=1, **kwargs):
    """Formats input from an Excel table and processes files.

    Args:
        input_table (str): Path to the Excel input table.
        n_jobs (int): Number of parallel jobs to use. Default is 1.
        **kwargs: Additional arguments for joblib.Parallel.

    Returns:
        None
    """
    df = pd.read_excel(input_table)

    def process_site(output_file, df_input):
        stacked = np.array([read_stack(input_file) for input_file in df_input.sort_values('channel')['original filename']])
        save_stack(output_file, stacked)

    if n_jobs != 1:
        from joblib import Parallel, delayed
        Parallel(n_jobs=n_jobs, **kwargs)(delayed(process_site)(output_file, df_input) for output_file, df_input in df.groupby('snakemake filename'))
    else:
        for output_file, df_input in df.groupby('snakemake filename'):
            process_site(output_file, df_input)

@ops.utils.memoize(active=False)
def read_stack(filename, copy=True, maxworkers=None, fix_axes=False):
    """Reads a TIFF file into a numpy array, with optional memory mapping and axis fixing.

    Args:
        filename (str): Path to the TIFF file.
        copy (bool): If True, returns a copy of the data. Default is True.
        maxworkers (int): Number of threads for decompression. Default is None.
        fix_axes (bool): If True, fixes incorrect axis orders. Default is False.

    Returns:
        np.ndarray: Image data.
    """
    data = imread(filename, multifile=False, is_ome=False, maxworkers=maxworkers)
    while data.shape[0] == 1:
        data = np.squeeze(data, axis=(0,))

    if copy:
        data = data.copy()

    if fix_axes:
        if data.ndim != 4:
            raise ValueError('`fix_axes` only tested for data with 4 dimensions')
        data = np.array([data.reshape((-1,) + data.shape[-2:])[n::data.shape[-4]] for n in range(data.shape[-4])])

    return data

@ops.utils.memoize(active=False)
def open_hdf_file(filename, mode='r'):
    """Opens an HDF file for reading or writing.

    Args:
        filename (str): Path to the HDF file.
        mode (str): Mode for opening the file ('r' for read, 'w' for write). Default is 'r'.

    Returns:
        tables.File: HDF file object.
    """
    return tables.file.open_file(filename, mode=mode)

def read_hdf_image(filename, bbox=None, leading_dims=None, array_name='image', pad=False, memoize=False):
    """Reads an image from an HDF file with optional bounding box and padding.

    Args:
        filename (str): Path to the HDF file.
        bbox (tuple): Bounding box (i0, j0, i1, j1) for cropping. Default is None.
        leading_dims (tuple): Leading dimensions to select from the HDF file. Default is None.
        array_name (str): Name of the array in the HDF file. Default is 'image'.
        pad (bool): If True, pads the image to the bounding box size. Default is False.
        memoize (bool): If True, caches file reads. Default is False.

    Returns:
        np.ndarray: Cropped and optionally padded image data.
    """
    hdf_file = open_hdf_file(filename, mode='r')

    try:
        image_node = hdf_file.get_node('/', name=array_name)
        if bbox is not None:
            i0, j0 = max(bbox[0], 0), max(bbox[1], 0)
            i1, j1 = min(bbox[2], image_node.shape[-2]), min(bbox[3], image_node.shape[-1])
            if leading_dims is None:
                image = image_node[..., i0:i1, j0:j1]
            else:
                image = image_node[leading_dims + (slice(i0, i1), slice(j0, j1))]
            if pad:
                pads = ((i0 - bbox[0], bbox[2] - i1), (j0 - bbox[1], bbox[3] - j1))
                image = np.pad(image, tuple((0, 0) for _ in range(image.ndim - 2)) + pads)
        else:
            image = image_node[...]
    except:
        print(f'Error in reading image array from HDF file for {filename}')
        image = None
    if not memoize:
        hdf_file.close()
    return image

def save_stack(name, data, luts=None, display_ranges=None, resolution=1., compress=0, dimensions=None, display_mode='composite', photometric='minisblack'):
    """Saves image data to a TIFF file with optional LUTs and display ranges.

    Args:
        name (str): Path to the output TIFF file.
        data (np.ndarray): Image data to be saved.
        luts (list): List of LUTs for each channel. Default is None.
        display_ranges (list): List of (min, max) pairs for each channel. Default is None.
        resolution (float): Resolution in microns per pixel. Default is 1.
        compress (int): Compression level. Default is 0 (no compression).
        dimensions (str): String specifying dimensions (e.g., 'TZC'). Default is None.
        display_mode (str): Display mode for the image. Default is 'composite'.
        photometric (str): Photometric interpretation ('minisblack' or 'rgb'). Default is 'minisblack'.

    Returns:
        None
    """
    if name.split('.')[-1] != 'tif':
        name += '.tif'
    name = os.path.abspath(name)

    if isinstance(data, list):
        data = np.array(data)

    if not (2 <= data.ndim <= 5):
        raise ValueError(f'Input has shape {data.shape}, but number of dimensions must be in range [2, 5]')

    if data.dtype == np.int64:
        if (data >= 0).all() and (data < 2**16).all():
            data = data.astype(np.uint16)
        else:
            data = data.astype(np.float32)
            print('Cast int64 to float32')
    if data.dtype == np.float64:
        data = data.astype(np.float32)
        # print('Cast float64 to float32')

    if data.dtype == np.bool_:
        data = 255 * data.astype(np.uint8)

    if data.dtype == np.int32:
        if data.min() >= 0 and data.max() < 2**16:
            data = data.astype(np.uint16)
        else:
            raise ValueError('Error casting from np.int32 to np.uint16, data out of range')

    if data.dtype not in (np.uint8, np.uint16, np.float32):
        raise ValueError(f'Cannot save data of type {data.dtype}')

    resolution = (1. / resolution,) * 2

    if not os.path.isdir(os.path.dirname(name)):
        os.makedirs(os.path.dirname(name))

    if data.ndim == 2:
        min, max = single_contrast(data, display_ranges)
        description = imagej_description_2D(min, max)
        imsave(name, data, photometric=photometric, description=description, resolution=resolution, compress=compress)
    else:
        nchannels = data.shape[-3]
        luts, display_ranges = infer_luts_display_ranges(data, luts, display_ranges)

        leading_shape = data.shape[:-2]
        if dimensions is None:
            dimensions = 'TZC'[::-1][:len(leading_shape)][::-1]

        if ('C' not in dimensions) or (nchannels == 1):
            contrast = single_contrast(data, display_ranges)
            description = imagej_description(leading_shape, dimensions, contrast=contrast)
            imsave(name, data, photometric=photometric, description=description, resolution=resolution, compress=compress)
        else:
            description = imagej_description(leading_shape, dimensions, display_mode=display_mode)
            tag_50838 = ij_tag_50838(nchannels)
            tag_50839 = ij_tag_50839(luts, display_ranges)

            imsave(name, data, photometric=photometric, description=description, resolution=resolution, compress=compress,
                   extratags=[(50838, 'I', len(tag_50838), tag_50838, True),
                              (50839, 'B', len(tag_50839), tag_50839, True)])


def infer_luts_display_ranges(data, luts, display_ranges):
    """Handles user input for LUTs (Look-Up Tables) and display ranges, ensuring they match the number of channels in the data.

    Args:
        data (np.ndarray): Image data array with shape including channel dimension.
        luts (list): List of LUTs for each channel. If None, default LUTs are used.
        display_ranges (list): List of (min, max) pairs for each channel. If None, they are inferred from the data.

    Returns:
        tuple: (luts, display_ranges) where both are lists of length equal to the number of channels.
    """
    nchannels = data.shape[-3]
    if luts is None:
        luts = DEFAULT_LUTS + (GRAY,) * (nchannels - len(DEFAULT_LUTS))

    if display_ranges is None:
        display_ranges = [None] * nchannels

    for i, dr in enumerate(display_ranges):
        if dr is None:
            x = data[..., i, :, :]
            display_ranges[i] = x.min(), x.max()
    
    if len(luts) < nchannels or len(display_ranges) < nchannels:
        error = 'Must provide at least {} luts and display ranges'
        raise IndexError(error.format(nchannels))
    else:
        luts = luts[:nchannels]
        display_ranges = display_ranges[:nchannels]

    return luts, display_ranges

def single_contrast(data, display_ranges):
    """Determines contrast (min and max values) for a single image or set of images based on display ranges.

    Args:
        data (np.ndarray): Image data array.
        display_ranges (list): List of (min, max) pairs for each channel.

    Returns:
        tuple: (min, max) contrast values.
    """
    try:
        min, max = np.array(display_ranges).flat[:2]
    except ValueError:
        min, max = data.min(), data.max()
    return min, max

def imagej_description_2D(min, max):
    """Generates a description string for 2D images in ImageJ format.

    Args:
        min (float): Minimum pixel value.
        max (float): Maximum pixel value.

    Returns:
        str: ImageJ description string.
    """
    return 'ImageJ=1.49v\nimages=1\nmin={min}\nmax={max}'.format(min=min, max=max)

def imagej_description(leading_shape, leading_axes, contrast=None, display_mode='composite'):
    """Generates a description string for multi-dimensional images in ImageJ format.

    Args:
        leading_shape (tuple): Shape of the leading dimensions.
        leading_axes (str): Axes labels for the dimensions.
        contrast (tuple, optional): (min, max) contrast values.
        display_mode (str): Display mode for the image (e.g., 'composite').

    Returns:
        str: ImageJ description string.
    """
    if len(leading_shape) != len(leading_axes):
        error = 'mismatched axes, shape is {} but axis labels are {}'
        raise ValueError(error.format(leading_shape, leading_axes))

    prefix = 'ImageJ=1.49v\n'
    suffix = 'hyperstack=true\nmode={mode}\n'.format(mode=display_mode)
    images = np.prod(leading_shape)
    sizes = {k: v for k,v in zip(leading_axes, leading_shape)}
    description = prefix + 'images={}\n'.format(images)
    if 'C' in sizes:
        description += 'channels={}\n'.format(sizes['C'])
    if 'Z' in sizes:
        description += 'slices={}\n'.format(sizes['Z'])
    if 'T' in sizes:
        description += 'frames={}\n'.format(sizes['T'])
    if contrast is not None:
        min, max = contrast
        description += 'min={0}\nmax={1}\n'.format(min, max)
    description += suffix

    return description

def ij_tag_50838(nchannels):
    """Creates a metadata tag for ImageJ indicating the size of metadata elements.

    Args:
        nchannels (int): Number of channels.

    Returns:
        tuple: Metadata size blocks.
    """
    info_block = (20,)  # Summary of metadata fields
    display_block = (16 * nchannels,)  # Display range block
    luts_block = (256 * 3,) * nchannels  # LUTs block
    return info_block + display_block + luts_block

def ij_tag_50839(luts, display_ranges):
    """Generates ImageJ metadata tag for display ranges and LUTs.

    Args:
        luts (tuple): List of LUTs for each channel.
        display_ranges (tuple): List of (min, max) pairs for each channel.

    Returns:
        tuple: ImageJ metadata tag.
    """
    d = struct.pack('<' + 'd' * len(display_ranges) * 2, *[y for x in display_ranges for y in x])
    tag = ''.join(['JIJI',
                   'gnar\x01\x00\x00\x00',
                   'stul%s\x00\x00\x00' % chr(len(luts)),
                   ]).encode('ascii') + d
    tag = struct.unpack('<' + 'B' * len(tag), tag)
    return tag + tuple(sum([list(x) for x in luts], []))

def load_stitching_offsets(filename):
    """Loads i,j coordinates from a text file saved by the Fiji Grid/Collection stitching plugin.

    Args:
        filename (str): Path to the text file.

    Returns:
        list: List of (i, j) coordinates.
    """
    from ast import literal_eval
    
    with open(filename, 'r') as fh:
        txt = fh.read()
    txt = txt.split('# Define the image coordinates')[1]
    lines = txt.split('\n')
    coordinates = []
    for line in lines:
        parts = line.split(';')
        if len(parts) == 3:
            coordinates += [parts[-1].strip()]
    
    return [(i, j) for j, i in map(literal_eval, coordinates)]
