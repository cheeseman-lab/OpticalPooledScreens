"""
Firesnake: Core Analysis Pipeline Module

This module serves as the central hub for the spatial transcriptomics analysis pipeline. It contains the Snake class, 
which encapsulates a wide range of methods for image processing, sequencing data analysis, and phenotype extraction. 
The module is designed to work with Snakemake, a workflow management system, to orchestrate complex computational workflows.

"""


import inspect
import functools
import os
import warnings

warnings.filterwarnings('ignore', message='numpy.dtype size changed')
warnings.filterwarnings('ignore', message='regionprops and image moments')
warnings.filterwarnings('ignore', message='non-tuple sequence for multi')
warnings.filterwarnings('ignore', message='precision loss when converting')

import numpy as np
import pandas as pd
import skimage
import ops.features
import ops.process
import ops.io
import ops.in_situ
import ops.io_hdf
from ops.process import Align
from scipy.stats import mode
from ops.constants import *
from itertools import combinations, permutations, product
import ops.cp_emulator

class Snake():
    """Container class for methods that act directly on data (names start with
    underscore) and methods that act on arguments from snakemake (e.g., filenames
    provided instead of image and table data). The snakemake methods (no underscore)
    are automatically loaded by `Snake.load_methods`.
    """

    # ALIGNMENT AND SEGMENTATION
    
    @staticmethod
    def _apply_illumination_correction(data, correction=None, zproject=False, rolling_ball=False, rolling_ball_kwargs={},
                                       n_jobs=1, backend='threading'):
        """
        Apply illumination correction to the given data.

        Parameters:
        data (numpy array): The input data to be corrected.
        correction (numpy array, optional): The correction factor to be applied. Default is None.
        zproject (bool, optional): If True, perform a maximum projection along the first axis. Default is False.
        rolling_ball (bool, optional): If True, apply a rolling ball background subtraction. Default is False.
        rolling_ball_kwargs (dict, optional): Additional arguments for the rolling ball background subtraction. Default is an empty dictionary.
        n_jobs (int, optional): The number of parallel jobs to run. Default is 1 (no parallelization).
        backend (str, optional): The parallel backend to use ('threading' or 'multiprocessing'). Default is 'threading'.

        Returns:
        numpy array: The corrected data.
        """

        # If zproject is True, perform a maximum projection along the first axis
        if zproject:
            data = data.max(axis=0)

        # If n_jobs is 1, process the data without parallelization
        if n_jobs == 1:
            # Apply the correction factor if provided
            if correction is not None:
                data = (data / correction).astype(np.uint16)

            # Apply rolling ball background subtraction if specified
            if rolling_ball:
                data = ops.process.subtract_background(data, **rolling_ball_kwargs).astype(np.uint16)

            return data

        else:
            # If n_jobs is greater than 1, apply illumination correction in parallel
            return ops.utils.applyIJ_parallel(Snake._apply_illumination_correction,
                                              arr=data,
                                              correction=correction,
                                              backend=backend,
                                              n_jobs=n_jobs)


    @staticmethod
    def _align_SBS(data, method='DAPI', upsample_factor=2, window=2, cutoff=1, q_norm=70,
                   align_within_cycle=True, cycle_files=None, keep_extras=False, n=1, remove_for_cycle_alignment=None):
        """
        Rigid alignment of sequencing cycles and channels.

        Parameters
        ----------
        data : np.ndarray or list of np.ndarrays
            Unaligned SBS image with dimensions (CYCLE, CHANNEL, I, J) or list of single cycle
            SBS images, each with dimensions (CHANNEL, I, J)

        method : {'DAPI','SBS_mean'}
            Method to use for alignment.

        upsample_factor : int, default 2
            Subpixel alignment is done if `upsample_factor` is greater than one (can be slow).

        window : int or float, default 2
            A centered subset of data is used if `window` is greater than one.

        cutoff : int or float, default 1
            Cutoff for normalized data to help deal with noise in images.

        q_norm : int, default 70
            Quantile for normalization to help deal with noise in images.

        align_within_cycle : bool, default True
            Align SBS channels within cycles.

        cycle_files : list of int or None, default None
            Used for parsing sets of images where individual channels are in separate files, which
            is more typically handled in a preprocessing step to combine images from the same cycle.

        keep_extras : bool, default False
            Retain channels that are not common across all cycles by propagating each 'extra' channel 
            to all cycles. Ignored if same number of channels exist for all cycles.

        n : int, default 1
            Determines the first SBS channel in `data`. This is after dealing with `keep_extras`, so 
            should only account for channels in common across all cycles if `keep_extras`=False.

        remove_for_cycle_alignment : None or int, default int
            Channel index to remove when finding cycle offsets. This is after dealing with `keep_extras`, 
            so should only account for channels in common across all cycles if `keep_extras`=False.

        Returns
        -------
        aligned : np.ndarray
            SBS image aligned across cycles.
        """

        # Handle case where cycle_files is provided
        if cycle_files is not None:
            arr = []
            current = 0
            # Iterate through cycle files to de-nest list of numpy arrays
            for cycle in cycle_files:
                if cycle == 1:
                    arr.append(data[current])
                else:
                    arr.append(np.array(data[current:current+cycle]))
                current += cycle
            data = arr
            print(data[0].shape)
            print(data[1].shape)

        # Check if the number of channels varies across cycles
        if ~all(x.shape == data[0].shape for x in data):
            # Keep only channels in common across all cycles
            channels = [x.shape[-3] if x.ndim > 2 else 1 for x in data]
            stacked = np.array([x[-min(channels):] for x in data])

            # Add back extra channels if requested
            if keep_extras:
                extras = np.array(channels) - min(channels)
                arr = []
                for cycle, extra in enumerate(extras):
                    if extra != 0:
                        arr.extend([data[cycle][extra_ch] for extra_ch in range(extra)])
                propagate = np.array(arr)
                stacked = np.concatenate((np.array([propagate] * stacked.shape[0]), stacked), axis=1)
            else:
                extras = [0] * stacked.shape[0]
        else:
            stacked = np.array(data)
            extras = [0] * stacked.shape[0]

        assert stacked.ndim == 4, 'Input data must have dimensions CYCLE, CHANNEL, I, J'

        # Align between SBS channels for each cycle
        aligned = stacked.copy()
        if align_within_cycle:
            align_it = lambda x: Align.align_within_cycle(x, window=window, upsample_factor=upsample_factor)
            aligned[:, n:] = np.array([align_it(x) for x in aligned[:, n:]])

        if method == 'DAPI':
            # Align cycles using the DAPI channel
            aligned = Align.align_between_cycles(aligned, channel_index=0,
                                                 window=window, upsample_factor=upsample_factor)
        elif method == 'SBS_mean':
            # Calculate cycle offsets using the average of SBS channels
            sbs_channels = list(range(n, aligned.shape[1]))
            if remove_for_cycle_alignment is not None:
                sbs_channels.remove(remove_for_cycle_alignment)
            target = Align.apply_window(aligned[:, sbs_channels], window=window).max(axis=1)
            normed = Align.normalize_by_percentile(target, q_norm=q_norm)
            normed[normed > cutoff] = cutoff
            offsets = Align.calculate_offsets(normed, upsample_factor=upsample_factor)
            # Apply cycle offsets to each channel
            for channel in range(aligned.shape[1]):
                if channel >= sum(extras):
                    aligned[:, channel] = Align.apply_offsets(aligned[:, channel], offsets)
                else:
                    # Don't apply offsets to extra channel in the cycle it was acquired
                    extra_idx = list(np.cumsum(extras) > channel).index(True)
                    extra_offsets = np.array([offsets[extra_idx]] * aligned.shape[0])
                    aligned[:, channel] = Align.apply_offsets(aligned[:, channel], extra_offsets)
        else:
            raise ValueError(f'method "{method}" not implemented')

        return aligned

    @staticmethod
    def _align_by_DAPI(data_1, data_2, channel_index=0, upsample_factor=2):
        """
        Align the second image to the first, using the channel at position `channel_index`.
        If `channel_index` is a tuple of length 2, specifies channels of [data_1, data_2] 
        to use for alignment. The first channel is usually DAPI.

        Parameters
        ----------
        data_1 : np.ndarray
            The reference image data.
        data_2 : np.ndarray
            The image data to be aligned.
        channel_index : int or tuple of int, default 0
            The index of the channel to use for alignment. If a tuple, specifies channels for
            data_1 and data_2 respectively.
        upsample_factor : int, default 2
            The factor by which to upsample the images for subpixel alignment.

        Returns
        -------
        aligned : np.ndarray
            The aligned version of `data_2`.
        """

        # Check if channel_index is a tuple
        if isinstance(channel_index, tuple):
            assert len(channel_index) == 2, 'channel_index must either be an integer or tuple of length 2'
            channel_index_1, channel_index_2 = channel_index
        else:
            channel_index_1, channel_index_2 = (channel_index,) * 2

        # Extract the channels to be used for alignment
        images = data_1[channel_index_1], data_2[channel_index_2]

        # Calculate the offsets needed to align data_2 to data_1
        _, offset = ops.process.Align.calculate_offsets(images, upsample_factor=upsample_factor)

        # Apply the calculated offsets to data_2
        offsets = [offset] * len(data_2)
        aligned = ops.process.Align.apply_offsets(data_2, offsets)

        return aligned
    
    @staticmethod
    def _align_phenotype_channels(data, target, source, riders=[], upsample_factor=2, window=2, remove=False):
        """
        Aligns phenotype channels in the data based on target and source channels.

        Args:
            data (np.ndarray): The input data containing the channels.
            target (int): The index of the channel you want to align other channels to.
            source (int): The index of the channel you want to align with the target.
            riders (list): Additional channel indices that should follow the same alignment as the source channel.
            upsample_factor (int): Factor to upsample the alignment process.
            window (int): Size of the window for alignment.
            remove (str or bool): Specifies whether to remove channels after alignment ('target', 'source', or False).

        Returns:
            np.ndarray: Aligned data with phenotype channels.
        """
        # Check if the input data has 4 dimensions (stacked data) or not.
        if data.ndim == 4:
            stack = True
            # Take maximum across the first axis if stacked data.
            data_ = data.max(axis=0)
        else:
            # Otherwise, copy the data.
            data_ = data.copy()
            stack = False
        
        # Apply windowing to the target and source channels.
        windowed = Align.apply_window(data_[[target, source]], window)
        
        # Calculate offsets based on the windowed data.
        offsets = Align.calculate_offsets(windowed, upsample_factor=upsample_factor)
        
        # Convert riders to a list if not already.
        if not isinstance(riders, list):
            riders = [riders]
        
        # Create full_offsets array to hold offsets for all channels.
        full_offsets = np.zeros((data_.shape[0], 2))
        full_offsets[[source] + riders] = offsets[1]
        
        # Align the data based on calculated offsets.
        if stack:
            aligned = np.array([Align.apply_offsets(slice_, full_offsets) for slice_ in data])
        else:
            aligned = Align.apply_offsets(data_, full_offsets)
        
        # Remove channels if specified.
        if remove == 'target':
            channel_order = list(range(data.shape[-3]))
            channel_order.remove(source)
            channel_order.insert(target + 1, source)
            aligned = aligned[..., channel_order, :, :]
            aligned = remove_channels(aligned, target)
        elif remove == 'source':
            aligned = remove_channels(aligned, source)

        return aligned

    @staticmethod
    def _stack_channels(data, flip_channels=False):
        """
        Stack channels from the given datasets into a single numpy array with the channel dimension as the third-to-last axis.

        Parameters
        ----------
        data : list of np.ndarrays
            A list of datasets, where each dataset can have different shapes.
            If a dataset has more than two dimensions, it is assumed to have a channel dimension.
        flip_channels : bool, optional
            If True, flip the channel order for datasets with more than 2 dimensions. Default is False.

        Returns
        -------
        np.ndarray
            A stacked array with the channel dimension as the third-to-last axis.
        """

        arr = []

        # Iterate through each dataset in the input data
        for dataset in data:
            # Check if the dataset has more than 2 dimensions (i.e., it has a channel dimension)
            if len(dataset.shape) > 2:
                # Flip the channels if requested
                if flip_channels:
                    dataset = np.flip(dataset, axis=0)
                # Extract each channel and append it to the arr list
                arr.extend([dataset[..., channel, :, :] for channel in range(dataset.shape[-3])])
            else:
                # If the dataset doesn't have a channel dimension, append it directly to arr
                arr.append(dataset)

        # Stack all the arrays along a new axis (third-to-last axis)
        return np.stack(arr, axis=-3)

        
    @staticmethod
    def _segment_nuclei(data, threshold, area_min, area_max, smooth=1.35, radius=15):
        """
        Find nuclei from DAPI channel.
        Uses local mean filtering to find cell foreground from aligned but unfiltered data,
        then filters identified regions by mean intensity threshold and area ranges.

        Parameters:
            data (numpy.ndarray or list): Image data.
                If numpy.ndarray, expected dimensions are (CHANNEL, I, J) with the DAPI channel in channel index 0.
                If list, the first element is assumed to be the DAPI channel.
                Can also be a single-channel DAPI image of dimensions (I, J).
            threshold (float): Foreground regions with mean DAPI intensity greater than `threshold` are labeled as nuclei.
            area_min (float): Minimum area for retaining nuclei after segmentation.
            area_max (float): Maximum area for retaining nuclei after segmentation.
            smooth (float, optional): Size of Gaussian kernel used to smooth the distance map to foreground prior to watershedding. Default is 1.35.
            radius (float, optional): Radius of disk used in local mean thresholding to identify foreground. Default is 15.

        Returns:
            nuclei (numpy.ndarray): Labeled segmentation mask of nuclei, dimensions are same as trailing two dimensions of `data`.
        """
        # Extract DAPI channel from the input data
        if isinstance(data, list):
            dapi = data[0]
        elif data.ndim == 3:
            dapi = data[0]
        else:
            dapi = data

        # Define keyword arguments for find_nuclei function
        kwargs = dict(threshold=lambda x: threshold, 
                      area_min=area_min, area_max=area_max,
                      smooth=smooth, radius=radius)

        # Suppress precision warning from skimage
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Use find_nuclei function to segment nuclei from DAPI channel
            nuclei = ops.process.find_nuclei(dapi, **kwargs)

        # Calculate the number of segmented nuclei (excluding background label)
        num_nuclei_segmented = len(np.unique(nuclei)) - 1
        print(f"Number of nuclei segmented: {num_nuclei_segmented}")

        # Convert nuclei array to uint16 dtype and return
        return nuclei.astype(np.uint16)

    @staticmethod
    def _segment_nuclei_stack(data, threshold, area_min, area_max, smooth=1.35, radius=15, n_jobs=1, backend='threading', tqdm=False):
        """
        Find nuclei from a nuclear stain (e.g., DAPI). Expects data to have shape (I, J) 
        (segments one image) or (N, I, J) (segments a series of nuclear stain images).

        Parameters
        ----------
        data : np.ndarray
            The input image data, either a single image (I, J) or a series of images (N, I, J).
        threshold : float
            The threshold value for nucleus detection.
        area_min : int
            The minimum area for detected nuclei.
        area_max : int
            The maximum area for detected nuclei.
        smooth : float, default 1.35
            The smoothing factor for the detection process.
        radius : int, default 15
            The radius for the rolling ball algorithm used in background subtraction.
        n_jobs : int, default 1
            The number of parallel jobs to run. If 1, no parallelization is used.
        backend : str, default 'threading'
            The parallel backend to use ('threading' or 'multiprocessing').
        tqdm : bool, default False
            Whether to use tqdm to show a progress bar during processing.

        Returns
        -------
        np.ndarray
            The segmented nuclei, with the same dimensions as the input data but with nuclei labeled by unique integers.
        """

        # Define keyword arguments for the find_nuclei function
        kwargs = dict(
            threshold=lambda x: threshold, 
            area_min=area_min, 
            area_max=area_max,
            smooth=smooth, 
            radius=radius
        )

        # Determine whether to use parallel processing
        if n_jobs == 1:
            # No parallel processing, use applyIJ function
            find_nuclei = ops.utils.applyIJ(ops.process.find_nuclei)
        else:
            # Use parallel processing, set additional kwargs for parallelization
            kwargs['n_jobs'] = n_jobs
            kwargs['tqdm'] = tqdm
            find_nuclei = functools.partial(ops.utils.applyIJ_parallel, ops.process.find_nuclei, backend=backend)

        # Suppress skimage precision warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Find nuclei in the data using the specified method
            nuclei = find_nuclei(data, **kwargs)

        # Return the nuclei data as unsigned 16-bit integers
        return nuclei.astype(np.uint16)


    @staticmethod
    def _segment_cells(data, nuclei, threshold, add_nuclei=True):
        """
        Segment cells from aligned data and match cell labels to nuclei labels.
        Note that labels can be skipped, for example if cells are touching the 
        image boundary.

        Parameters
        ----------
        data : np.ndarray
            The aligned image data. Can have 2, 3, or 4 dimensions.
        nuclei : np.ndarray
            The segmented nuclei data.
        threshold : float
            The threshold value for cell segmentation.
        add_nuclei : bool, default True
            Whether to add the nuclei shape to the cell mask to help with mapping 
            reads to cells at the edge of the field of view.

        Returns
        -------
        np.ndarray
            The segmented cells, with labels matched to nuclei.
        """

        # Determine the mask based on the number of dimensions in data
        if data.ndim == 4:
            # If data has 4 dimensions: no DAPI, min over cycles, mean over channels
            mask = data[:, 1:].min(axis=0).mean(axis=0)
        elif data.ndim == 3:
            # If data has 3 dimensions: median over the remaining channels
            mask = np.median(data[1:], axis=0)
        elif data.ndim == 2:
            # If data has 2 dimensions: use the data directly as the mask
            mask = data
        else:
            # Raise an error if data has an unsupported number of dimensions
            raise ValueError("Data must have 2, 3, or 4 dimensions")

        # Apply the threshold to the mask to create a binary mask
        mask = mask > threshold

        # Add the nuclei to the mask if add_nuclei is True
        if add_nuclei:
            mask += nuclei.astype(bool)

        try:
            # Suppress skimage precision warning
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Find cells in the mask
                cells = ops.process.find_cells(nuclei, mask)
        except ValueError:
            # Handle the case where no cells are found
            print('segment_cells error -- no cells')
            cells = nuclei

        # Calculate the number of segmented cells (excluding background label)
        num_cells_segmented = len(np.unique(cells)) - 1
        print(f"Number of cells segmented: {num_cells_segmented}")

        # Return the segmented cells
        return cells
    
    @staticmethod
    def _segment_cells_dilation(nuclei, radius=10, ring=True):
        """
        Segment cells by dilating the nuclei.

        Parameters
        ----------
        nuclei : np.ndarray
            The segmented nuclei data.
        radius : int, default 10
            The radius for the disk used in binary dilation.
        ring : bool, default True
            Whether to subtract the nuclei from the cells to create a ring around the nuclei.

        Returns
        -------
        np.ndarray
            The segmented cells, with optional ring around the nuclei.
        """

        # Perform binary dilation on the nuclei using a disk of the given radius
        mask = skimage.morphology.binary_dilation(nuclei > 0, skimage.morphology.disk(radius))

        try:
            # Suppress skimage precision warning
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Find cells in the mask
                cells = ops.process.find_cells(nuclei, mask)
        except ValueError:
            # Handle the case where no cells are found
            print('segment_cells error -- no cells')
            cells = nuclei

        if ring:
            # Subtract nuclei from cells to create a ring around the nuclei
            # WARNING: This can cause issues and integer wraparound for cells on the edge which are removed in find_cells()
            cells -= nuclei

        # Calculate the number of segmented cells (excluding background label)
        num_cells_segmented = len(np.unique(cells)) - 1
        print(f"Number of cells segmented: {num_cells_segmented}")           
        
        # Return the segmented cells
        return cells
    
    @staticmethod
    def _segment_cells_robust(data, channel, nuclei, background_offset, background_quantile=0.05, 
                              smooth=None, erosion=None, add_nuclei=True, mask_dilation=5):
        """
        Segment cells robustly from image data.

        Parameters:
            data (numpy.ndarray): Image data containing multiple channels.
            channel (int): Index of the channel to use for segmentation.
            nuclei (numpy.ndarray): Labeled segmentation mask of nuclei.
            background_offset (float): Offset value for background thresholding.
            background_quantile (float, optional): Quantile value for background thresholding. Default is 0.05.
            smooth (float, optional): Standard deviation of Gaussian kernel for image smoothing. Default is None.
            erosion (int, optional): Size of erosion for refining segmentation. Default is None.
            add_nuclei (bool, optional): Whether to add nuclei to the segmented mask. Default is True.
            mask_dilation (int, optional): Size of dilation for creating mask. Default is 5.

        Returns:
            numpy.ndarray: Labeled segmentation mask of cell boundaries.
        """
        # Find region where all channels are valid, e.g., after applying offsets to align channels
        mask = data.min(axis=0) > 0

        # Smooth the image if specified
        if smooth is not None:
            image = skimage.filters.gaussian(data[channel], smooth, preserve_range=True).astype(data.dtype)
        else:
            image = data[channel]

        # Calculate threshold for background segmentation
        threshold = np.quantile(image[mask], background_quantile) + background_offset

        # Perform background segmentation
        semantic = image > threshold

        # Add nuclei to the segmentation mask if specified
        if add_nuclei:
            semantic += nuclei.astype(bool)

        # Erode the segmentation mask if erosion is specified
        if erosion is not None:
            semantic[~mask] = True
            semantic = skimage.morphology.binary_erosion(semantic, skimage.morphology.disk(erosion/2))
            semantic[~mask] = False
            if add_nuclei:
                semantic += nuclei.astype(bool)

        # Find cells using nuclei as seeds
        labels = ops.process.find_cells(nuclei, semantic)

        # Function to remove cells near the image border
        def remove_border(labels, mask, dilate=mask_dilation):
            mask = skimage.morphology.binary_dilation(mask, np.ones((dilate, dilate)))
            remove = np.unique(labels[mask])
            labels = labels.copy()
            labels.flat[np.in1d(labels, remove)] = 0
            return labels

        # Remove cells near the image border and return the segmented mask
        cells = remove_border(labels, ~mask)
        
        # Calculate the number of segmented cells (excluding background label)
        num_cells_segmented = len(np.unique(cells)) - 1
        print(f"Number of cells segmented: {num_cells_segmented}")
        
        return cells

    @staticmethod
    def _segment_cell_2019(data, nuclei_threshold, nuclei_area_min,
                           nuclei_area_max, cell_threshold):
        """
        Combine morphological segmentation of nuclei and cells to have the same interface as _segment_cellpose.

        Parameters:
            data (numpy.ndarray): Image data for segmentation.
            nuclei_threshold (float): Threshold for nuclei segmentation.
            nuclei_area_min (float): Minimum area for retaining nuclei after segmentation.
            nuclei_area_max (float): Maximum area for retaining nuclei after segmentation.
            cell_threshold (float): Threshold used for cell boundary segmentation.

        Returns:
            tuple: A tuple containing:
                - nuclei (numpy.ndarray): Labeled segmentation mask of nuclei.
                - cells (numpy.ndarray): Labeled segmentation mask of cell boundaries.
        """
        # If SBS data, image will have 4 dimensions
        if data.ndim == 4:
            # Select first cycle
            nuclei_data = data[0]

        # Segment nuclei using the _segment_nuclei method
        nuclei = Snake._segment_nuclei(nuclei_data, nuclei_threshold, nuclei_area_min, nuclei_area_max)
        
        # Segment cells using the _segment_cells method
        cells = Snake._segment_cells(data, nuclei, cell_threshold)
        
        return nuclei, cells
    
    @staticmethod
    def _segment_cell_2022(data, nuclei_threshold, nuclei_area_min,
                           nuclei_area_max, channel, background_offset, 
                           cell_count_thresholds, background_quantile,
                           smooth=None, erosion=None, 
                           add_nuclei=True, mask_dilation=5):
        """
        Combine morphological segmentation of nuclei and cells to have the same interface as _segment_cellpose using segment_cell_robust.

        This method segments nuclei and cells from image data using specified thresholds and parameters. It first segments nuclei and then 
        segments cells based on the segmented nuclei.

        Parameters:
            data (numpy.ndarray): Image data for segmentation.
            nuclei_threshold (float): Threshold for nuclei segmentation.
            nuclei_area_min (float): Minimum area for retaining nuclei after segmentation.
            nuclei_area_max (float): Maximum area for retaining nuclei after segmentation.
            channel (int): Channel index used for background signal in cell segmentation.
            background_offset (float): Offset for background threshold in cell segmentation.
            cell_count_thresholds (tuple): Threshold values for cell count classification.
            background_quantile (dict): Dictionary containing quantile values for different cell count thresholds.
            smooth (float, optional): Smoothing factor for preprocessing. Defaults to None.
            erosion (float, optional): Erosion factor for preprocessing. Defaults to None.
            add_nuclei (bool, optional): Whether to add nuclei to the cell segmentation. Defaults to True.
            mask_dilation (int, optional): Dilation factor for masks. Defaults to 5.

        Returns:
            tuple: A tuple containing:
                - nuclei (numpy.ndarray): Labeled segmentation mask of nuclei.
                - cells (numpy.ndarray): Labeled segmentation mask of cell boundaries.
        """
        # Segment nuclei using the _segment_nuclei method
        nuclei = Snake._segment_nuclei(data[0], nuclei_threshold, nuclei_area_min, nuclei_area_max)

        # Determine quantile based on the maximum value in the segmented nuclei images
        if nuclei.max() < cell_count_thresholds[0]:
            quantile = background_quantile['low']
        elif nuclei.max() > cell_count_thresholds[1]:
            quantile = background_quantile['high']
        else:
            quantile = background_quantile['mid']

        # Segment cells using the _segment_cells_robust method
        cells = Snake._segment_cells_robust(data, channel, nuclei, background_offset, quantile, smooth, erosion, add_nuclei, mask_dilation)

        return nuclei, cells

    @staticmethod
    def _segment_cellpose(data, dapi_index, cyto_index, nuclei_diameter, cell_diameter,
                          cellpose_kwargs=dict(), cells=True, cyto_model='cyto', reconcile='consensus', 
                          logscale=True, return_counts=False):
        """
        Segment cells using Cellpose algorithm.
        Args:
            data (numpy.ndarray): Multichannel image data.
            dapi_index (int): Index of DAPI channel.
            cyto_index (int): Index of cytoplasmic channel.
            nuclei_diameter (int): Estimated diameter of nuclei.
            cell_diameter (int): Estimated diameter of cells.
            logscale (bool, optional): Whether to apply logarithmic transformation to image data.
            cellpose_kwargs (dict, optional): Additional keyword arguments for Cellpose.
            cells (bool, optional): Whether to segment both nuclei and cells or just nuclei.
            reconcile (str, optional): Method for reconciling nuclei and cells. Default is 'consensus'.
            return_counts (bool, optional): Whether to return counts of nuclei and cells. Default is False.
        Returns:
            tuple or numpy.ndarray: If 'cells' is True, returns tuple of nuclei and cell segmentation masks,
            otherwise returns only nuclei segmentation mask. If return_counts is True, includes a dictionary of counts.
        """
        # Prepare data for Cellpose by creating a merged RGB image
        log_kwargs = cellpose_kwargs.pop('log_kwargs', dict())  # Extract log_kwargs from cellpose_kwargs
        rgb = Snake._prepare_cellpose(data, dapi_index, cyto_index, logscale, log_kwargs=log_kwargs)

        counts = {}

        # Perform cell segmentation using Cellpose
        if cells:
            # Segment both nuclei and cells
            from ops.cellpose import segment_cellpose_rgb
            if return_counts:
                nuclei, cells, seg_counts = segment_cellpose_rgb(rgb, nuclei_diameter, cell_diameter, 
                                                                 reconcile=reconcile, return_counts=True, 
                                                                 **cellpose_kwargs)
                counts.update(seg_counts)

            else:
                nuclei, cells = segment_cellpose_rgb(rgb, nuclei_diameter, cell_diameter, 
                                                     reconcile=reconcile, **cellpose_kwargs)

            counts['final_nuclei'] = len(np.unique(nuclei)) - 1
            counts['final_cells'] = len(np.unique(cells)) - 1
            counts_df = pd.DataFrame([counts])
            print(f"Number of nuclei segmented: {counts['final_nuclei']}")
            print(f"Number of cells segmented: {counts['final_cells']}")

            if return_counts:
                return nuclei, cells, counts_df
            else:
                return nuclei, cells
        else:
            # Segment only nuclei
            from ops.cellpose import segment_cellpose_nuclei_rgb
            nuclei = segment_cellpose_nuclei_rgb(rgb, nuclei_diameter, **cellpose_kwargs)
            counts['final_nuclei'] = len(np.unique(nuclei)) - 1
            print(f"Number of nuclei segmented: {counts['final_nuclei']}")
            counts_df = pd.DataFrame([counts])

            if return_counts:
                return nuclei, counts_df
            else:
                return nuclei

    
    @staticmethod
    def _prepare_cellpose(data, dapi_index, cyto_index, logscale=True, log_kwargs=dict()):
        """
        Prepare a three-channel RGB image for use with the Cellpose GUI.

        Parameters:
            data (list or numpy.ndarray): List or array containing DAPI and cytoplasmic channel images.
            dapi_index (int): Index of the DAPI channel in the data.
            cyto_index (int): Index of the cytoplasmic channel in the data.
            logscale (bool, optional): Whether to apply log scaling to the cytoplasmic channel. Default is True.

        Returns:
            numpy.ndarray: Three-channel RGB image prepared for use with Cellpose GUI.
        """
        # Import necessary function from ops.cellpose module
        from ops.cellpose import image_log_scale

        # Import necessary function from skimage module
        from skimage import img_as_ubyte

        # Extract DAPI and cytoplasmic channel images from the data
        dapi = data[dapi_index]
        cyto = data[cyto_index]

        # Create a blank array with the same shape as the DAPI channel
        blank = np.zeros_like(dapi)

        # Apply log scaling to the cytoplasmic channel if specified
        if logscale:
            cyto = image_log_scale(cyto,**log_kwargs)
            cyto /= cyto.max()  # Normalize the image for uint8 conversion

        # Normalize the intensity of the DAPI channel and scale it to the range [0, 1]
        dapi_upper = np.percentile(dapi, 99.5)
        dapi = dapi / dapi_upper
        dapi[dapi > 1] = 1

        # Convert the channels to uint8 format for RGB image creation
        red, green, blue = img_as_ubyte(blank), img_as_ubyte(cyto), img_as_ubyte(dapi)

        # Stack the channels to create the RGB image and transpose the dimensions
        # return np.array([red, green, blue]).transpose([1, 2, 0])
        return np.array([red, green, blue])

    @staticmethod
    def _identify_cytoplasm(nuclei, cells):
        """
        Identifies and isolates the cytoplasm region in an image by subtracting the nuclei region from the cells region.

        Parameters:
        nuclei (ndarray): A 2D array representing the nuclei regions.
        cells (ndarray): A 2D array representing the cells regions.

        Returns:
        ndarray: A 2D array representing the cytoplasm regions.
        """
        # Subtract nuclei from cells to get an initial estimate of the cytoplasm
        cytoplasms = cells - nuclei

        # Extract the border elements of the cells array
        cut_1 = np.concatenate([cells[0, :], cells[-1, :], cells[:, 0], cells[:, -1]])

        # Get elements that are unique to cells but not in the initial cytoplasm estimate
        cut_2 = np.array(list(set(np.unique(cells)) - set(np.unique(cytoplasms))))

        # Get elements that are unique to nuclei but not in cells
        cut_3 = np.array(list(set(np.unique(nuclei)) - set(np.unique(cells))))

        # Combine all the cuts to form a comprehensive cut array
        cut = np.concatenate([cut_1, cut_2])

        # Set the elements in cells and nuclei arrays to 0 where the cut elements are present
        cells.flat[np.in1d(cells, np.unique(cut))] = 0
        nuclei.flat[np.in1d(nuclei, np.unique(cut))] = 0

        # Recalculate the cytoplasm after cleaning up the cells and nuclei arrays
        cytoplasms = cells - nuclei

        # Set any values in cytoplasm greater than the max value in cells to 0
        cytoplasms[cytoplasms > cells.max()] = 0
        
        # Calculate the number of identified cytoplasms (excluding background label)
        num_cytoplasm_segmented = len(np.unique(cytoplasms)) - 1
        print(f"Number of cytoplasms identified: {num_cytoplasm_segmented}")
        
        # Return the final cytoplasm array
        return cytoplasms

    @staticmethod
    def _identify_cytoplasm_cellpose(nuclei, cells):
        """
        Identifies and isolates the cytoplasm region in an image based on the provided nuclei and cells masks.

        Parameters:
        nuclei (ndarray): A 2D array representing the nuclei regions.
        cells (ndarray): A 2D array representing the cells regions.

        Returns:
        ndarray: A 2D array representing the cytoplasm regions.
        """
        # Check if the number of unique labels in nuclei and cells are the same
        if len(np.unique(nuclei)) != len(np.unique(cells)):
            return None  # Break out of the function if the masks are not compatible
        
        # Create an empty cytoplasmic mask with the same shape as cells
        cytoplasms = np.zeros(cells.shape)  
        
        # Iterate over each unique cell label
        for cell_label in np.unique(cells):
            # Skip if the cell label is 0 (background)
            if cell_label == 0:
                continue
            
            # Find the corresponding nucleus label for this cell
            nucleus_label = cell_label
            
            # Get the coordinates of the nucleus and cell regions
            nucleus_coords = np.argwhere(nuclei == nucleus_label)
            cell_coords = np.argwhere(cells == cell_label)
            
            # Update the cytoplasmic mask with the cell region
            cytoplasms[cell_coords[:, 0], cell_coords[:, 1]] = cell_label
            
            # Remove the nucleus region from the cytoplasmic mask
            cytoplasms[nucleus_coords[:, 0], nucleus_coords[:, 1]] = 0

        # Calculate the number of identified cytoplasms (excluding background label)
        num_cytoplasm_segmented = len(np.unique(cytoplasms)) - 1
        print(f"Number of cytoplasms identified: {num_cytoplasm_segmented}")
        
        # Return the final cytoplasm array
        return cytoplasms.astype(int)
    
    # IN SITU

    @staticmethod
    def _transform_log(data, sigma=1, skip_index=None):
        """
        Apply Laplacian-of-Gaussian filter from scipy.ndimage to the input data.

        Parameters:
            data (numpy.ndarray): Aligned SBS image data with expected dimensions of (CYCLE, CHANNEL, I, J).
            sigma (float, optional): Size of the Gaussian kernel used in the Laplacian-of-Gaussian filter. Default is 1.
            skip_index (None or int, optional): If an integer, skips transforming a specific channel (e.g., DAPI with skip_index=0).

        Returns:
            loged (numpy.ndarray): LoG-ed `data`.
        """
        # Convert input data to a numpy array
        data = np.array(data)
        
        # Apply Laplacian-of-Gaussian filter to the data using log_ndi function from ops.process module
        loged = ops.process.log_ndi(data, sigma=sigma)
        
        # If skip_index is specified, keep the original values for the corresponding channel
        if skip_index is not None:
            loged[..., skip_index, :, :] = data[..., skip_index, :, :]
        
        return loged

    @staticmethod
    def _compute_std(data, remove_index=None):
        """
        Use standard deviation over cycles, followed by mean across channels to estimate sequencing read locations.
        If only 1 cycle is present, takes standard deviation across channels.

        Parameters:
            data (numpy.ndarray): LoG-ed SBS image data with expected dimensions of (CYCLE, CHANNEL, I, J).
            remove_index (None or int, optional): Index of data to remove from subsequent analysis, generally any non-SBS channels (e.g., DAPI).

        Returns:
            consensus (numpy.ndarray): Standard deviation score for each pixel, dimensions of (I, J).
        """
        # Remove specified index channel if needed
        if remove_index is not None:
            data = remove_channels(data, remove_index)

        # If only one cycle present, add a new dimension
        if len(data.shape) == 3:
            data = data[:, None, ...]

        # Compute standard deviation across cycles and mean across channels
        consensus = np.std(data, axis=0).mean(axis=0)

        return consensus

    
    @staticmethod
    def _find_peaks(data, width=5, remove_index=None):
        """
        Find local maxima and label by difference to next-highest neighboring pixel.
        Conventionally used to estimate SBS read locations by inputting the standard deviation score.

        Parameters:
            data (numpy.ndarray): 2D image data.
            width (int, optional): Neighborhood size for finding local maxima. Default is 5.
            remove_index (None or int, optional): Index of data to remove from subsequent analysis, generally any non-SBS channels (e.g., DAPI).

        Returns:
            peaks (numpy.ndarray): Local maxima scores, dimensions same as data. At a maximum, the value is max - min in the defined neighborhood, elsewhere zero.
        """
        # Remove specified index channel if needed
        if remove_index is not None:
            data = remove_channels(data, remove_index)

        # If data is 2D, convert it to a list
        if data.ndim == 2:
            data = [data]

        # Find peaks in each image with a defined neighborhood size
        peaks = [ops.process.find_peaks(x, n=width) if x.max() > 0 else x for x in data]

        # Convert the list of peaks to a numpy array and squeeze it to remove singleton dimensions
        peaks = np.array(peaks).squeeze()
        
        return peaks

    @staticmethod
    def _max_filter(data, width, remove_index=None):
        """
        Apply a maximum filter in a window of `width`. 
        Conventionally operates on Laplacian-of-Gaussian filtered SBS data,
        dilating sequencing channels to compensate for single-pixel alignment error.

        Parameters:
            data (numpy.ndarray): Image data with expected dimensions of (..., I, J) with up to 4 total dimensions.
            width (int): Neighborhood size for max filtering.
            remove_index (None or int, optional): Index of data to remove from subsequent analysis, generally any non-SBS channels (e.g., DAPI).

        Returns:
            maxed (numpy.ndarray): Maxed `data` with preserved dimensions.
        """
        # Import necessary modules and functions
        import scipy.ndimage.filters
        
        # Ensure data has at least 3 dimensions
        if data.ndim == 2:
            data = data[None, None]
        elif data.ndim == 3:
            data = data[None]
        
        # Remove specified index channel if needed
        if remove_index is not None:
            data = remove_channels(data, remove_index)
        
        # Apply maximum filter to the data with specified window size
        maxed = scipy.ndimage.filters.maximum_filter(data, size=(1, 1, width, width))
        
        return maxed


    @staticmethod
    def _extract_bases(maxed, peaks, cells, threshold_peaks, wildcards, bases='GTAC'):
        """
        Find the signal intensity from `maxed` at each point in `peaks` above `threshold_peaks`.
        Output is labeled by `wildcards` (e.g., well and tile) and label at that position in integer
        mask `cells`.

        Parameters:
            maxed (numpy.ndarray): Base intensity at each point, output of Snake.max_filter(), expected dimensions of (CYCLE, CHANNEL, I, J).
            peaks (numpy.ndarray): Peaks/local maxima score for each pixel, output of Snake.find_peaks().
            cells (numpy.ndarray): Labeled segmentation mask of cell boundaries for labeling reads.
            threshold_reads (float): Threshold for `peaks` for identifying candidate sequencing reads.
            wildcards (dict): Metadata to include in output table, e.g., well, tile, etc. In Snakemake, use wildcards object.
            bases (str, optional): Order of bases corresponding to the order of acquired SBS channels in `maxed`. Default is 'GTAC'.

        Returns:
            pandas.DataFrame: Table of all candidate sequencing reads with intensity of each base for every cycle,
                (I,J) position of read, and metadata from `wildcards`.
        """
        if maxed.ndim == 3:
            maxed = maxed[None]

        if len(bases) != maxed.shape[1]:
            error = 'Sequencing {0} bases {1} but maxed data had shape {2}'
            raise ValueError(error.format(len(bases), bases, maxed.shape))

        # "cycle 0" is reserved for phenotyping
        cycles = list(range(1, maxed.shape[0] + 1))
        bases = list(bases)

        # Extract base intensity values, labels, and positions
        values, labels, positions = ops.in_situ.extract_base_intensity(maxed, peaks, cells, threshold_peaks)

        # Format base intensity data into DataFrame
        df_bases = ops.in_situ.format_bases(values, labels, positions, cycles, bases)

        # Add wildcard metadata to the DataFrame
        for k, v in sorted(wildcards.items()):
            df_bases[k] = v

        return df_bases

    @staticmethod
    def _analyze_single(data, alignment_ref, cells, peaks, threshold_peaks, wildcards, channel_ix=1):
        """
        Combine transform_log, max_filter, extract_bases into one function

        Args:
            data (numpy.ndarray): Raw sequencing data.
            alignment_ref (numpy.ndarray): Reference alignment data.
            cells (numpy.ndarray): Labeled segmentation mask defining cell objects.
            peaks (numpy.ndarray): Peaks data.
            threshold_peaks (float): Peak threshold value.
            wildcards (dict): Metadata to include in the output table.
            channel_ix (int): Index of the channel to analyze. Default is 1.

        Returns:
            pandas.DataFrame: Table of extracted bases for each cell.
        """
        # If alignment reference has 3 dimensions, extract the first slice
        if alignment_ref.ndim == 3:
            alignment_ref = alignment_ref[0]

        # Prepare data for alignment and analysis
        data = np.array([[alignment_ref, alignment_ref], data[[0, channel_ix]]])

        # Align data between cycles
        aligned = ops.process.Align.align_between_cycles(data, 0, window=2)

        # Transform aligned data using logarithm
        loged = Snake._transform_log(aligned[1, 1])

        # Apply max filter to the transformed data
        maxed = Snake._max_filter(loged, width=3)

        # Extract bases using maximum filtered data
        return Snake._extract_bases(maxed, peaks, cells, bases=['-'], threshold_peaks=threshold_peaks, wildcards=wildcards)
    
    @staticmethod
    def _call_reads(df_bases, peaks=None, correction_only_in_cells=True, normalize_bases=True):
        """
        Call reads by compensating for channel cross-talk and calling the base
        with the highest corrected intensity for each cycle. Median correction
        is performed independently for each tile.

        Parameters:
        -----------
        df_bases : pandas DataFrame
            Table of base intensity for all candidate reads, output of Snake.extract_bases().

        peaks : None or numpy array, default None
            Peaks/local maxima score for each pixel (output of Snake.find_peaks()) to be included
            in the df_reads table for downstream QC or other analysis. If None, does not include
            peaks scores in returned df_reads table.

        correction_only_in_cells : boolean, default True
            If True, restricts median correction/compensation step to account only for reads that
            are within a cell, as defined by the cell segmentation mask passed into
            Snake.extract_bases(). Often identified spots outside of cells are not true sequencing
            reads.

        normalize_bases : boolean, default True
            If True, normalizes the base intensities before performing median correction.

        Returns:
        --------
        df_reads : pandas DataFrame
            Table of all reads with base calls resulting from SBS compensation and related metadata.
        """

        if df_bases is None:
            return
        if correction_only_in_cells:
            if len(df_bases.query('cell > 0')) == 0:
                return

        cycles = len(set(df_bases['cycle']))
        channels = len(set(df_bases['channel']))

        if normalize_bases:
            # Clean up and normalize base intensities, then perform median calling
            df_reads = (df_bases
                        .pipe(ops.in_situ.clean_up_bases)
                        .pipe(ops.in_situ.normalize_bases)
                        .pipe(ops.in_situ.do_median_call, cycles, channels=channels,
                              correction_only_in_cells=correction_only_in_cells)
                        )
        else:
            # Clean up bases and perform median calling without normalization
            df_reads = (df_bases
                        .pipe(ops.in_situ.clean_up_bases)
                        .pipe(ops.in_situ.do_median_call, cycles, channels=channels,
                              correction_only_in_cells=correction_only_in_cells)
                        )

        # Include peaks scores if available
        if peaks is not None:
            i, j = df_reads[['i', 'j']].values.T
            df_reads['peak'] = peaks[i, j]

        return df_reads



    @staticmethod
    def _call_cells(df_reads, df_pool=None, q_min=0):
        """Perform median correction independently for each tile.

        Args:
            df_reads (DataFrame): DataFrame containing read information.
            df_pool (DataFrame, optional): DataFrame containing pool information. Default is None.
            q_min (int, optional): Minimum quality threshold. Default is 0.

        Returns:
            DataFrame: DataFrame containing corrected cells.
        """
        # Check if df_reads is None and return if so
        if df_reads is None:
            return

        # Check if df_pool is None
        if df_pool is None:
            # Filter reads by quality threshold and call cells
            return (
                df_reads
                .query('Q_min >= @q_min')
                .pipe(ops.in_situ.call_cells)
            )
        else:
            # Determine the experimental prefix length
            prefix_length = len(df_reads.iloc[0].barcode)

            # Add prefix to the pool DataFrame
            df_pool[PREFIX] = df_pool.apply(lambda x: x.sgRNA[:prefix_length], axis=1)

            # Filter reads by quality threshold and call cells mapping
            return (
                df_reads
                .query('Q_min >= @q_min')
                .pipe(ops.in_situ.call_cells_mapping, df_pool)
            )


    # ANNOTATE FUNCTIONS

    @staticmethod
    def _annotate_segment_on_sequencing_data(data, nuclei, cells):
        """
        Annotate outlines of nuclei and cells on sequencing data.

        This function overlays outlines of nuclei and cells on the provided sequencing data.

        Args:
            data (numpy.ndarray): Sequencing data with shape (cycles, channels, height, width).
            nuclei (numpy.ndarray): Array representing nuclei outlines.
            cells (numpy.ndarray): Array representing cells outlines.

        Returns:
            numpy.ndarray: Annotated sequencing data with outlines of nuclei and cells.

        Note:
            Assumes that the `ops.annotate.outline_mask()` function is available.
        """
        # Import necessary function from ops.annotate module
        from ops.annotate import outline_mask

        # Ensure data has at least 4 dimensions
        if data.ndim == 3:
            data = data[None]

        # Get dimensions of the sequencing data
        cycles, channels, height, width = data.shape

        # Create an array to store annotated data
        annotated = np.zeros((cycles, channels + 1, height, width), dtype=np.uint16)

        # Generate combined mask for nuclei and cells outlines
        mask = ((outline_mask(nuclei, direction='inner') > 0) +
                (outline_mask(cells, direction='inner') > 0))

        # Copy original data to annotated data
        annotated[:, :channels] = data

        # Add combined mask to the last channel
        annotated[:, channels] = mask

        return np.squeeze(annotated)

    @staticmethod
    def _annotate_on_phenotyping_data(data, nuclei, cells):
        """
        Annotate outlines of nuclei and cells on phenotyping data.

        This function overlays outlines of nuclei and cells on the provided phenotyping data.

        Args:
            data (numpy.ndarray): Phenotyping data with shape (channels, height, width).
            nuclei (numpy.ndarray): Array representing nuclei outlines.
            cells (numpy.ndarray): Array representing cells outlines.

        Returns:
            numpy.ndarray: Annotated phenotyping data with outlines of nuclei and cells.

        Note:
            Assumes that the `ops.annotate.outline_mask()` function is available.
        """
        # Import necessary function from ops.annotate module
        from ops.annotate import outline_mask

        # Ensure data has at least 3 dimensions
        if data.ndim == 2:
            data = data[None]

        # Get dimensions of the phenotyping data
        channels, height, width = data.shape

        # Create an array to store annotated data
        annotated = np.zeros((channels + 1, height, width), dtype=np.uint16)

        # Generate combined mask for nuclei and cells outlines
        mask = ((outline_mask(nuclei, direction='inner') > 0) +
                (outline_mask(cells, direction='inner') > 0))

        # Copy original data to annotated data
        annotated[:channels] = data

        # Add combined mask to the last channel
        annotated[channels] = mask

        return np.squeeze(annotated)
    
    @staticmethod
    def _annotate_bases_on_SBS_log(log, df_reads):
        """
        Annotate bases on a Single Base Sequencing (SBS) log.

        This function takes a log of SBS sequencing data and a DataFrame of reads, 
        then annotates the bases on the log according to the reads.

        Args:
            log (numpy.ndarray): The SBS sequencing log with shape (cycles, channels, height, width).
            df_reads (pandas.DataFrame): DataFrame containing reads information.

        Returns:
            numpy.ndarray: Annotated SBS log with bases.

        Note:
            Assumes that the `ops.annotate.annotate_bases()` function is available.

        """
        # Get dimensions of the SBS log
        cycles, channels, height, width = log.shape

        # Annotate bases on reads
        base_labels = ops.annotate.annotate_bases(df_reads, width=3, shape=(height, width))

        # Create an array to store annotated log
        annotated = np.zeros((cycles, channels + 1, height, width), dtype=np.uint16)

        # Copy original log to annotated log
        annotated[:, :channels] = log

        # Add annotated bases to the last channel
        annotated[:, channels] = base_labels

        return annotated

    @staticmethod
    def _annotate_bases_on_SBS_reads_peaks(log, peaks, df_reads, barcode_table, sbs_cycles, shape=(1024, 1024), return_channels="both"):
        """
        Annotate additional features on Single Base Sequencing (SBS) data.

        This function annotates additional features such as bases, quality scores, and peak intensity 
        around peaks on the provided SBS sequencing data.

        Args:
            log (numpy.ndarray): The SBS sequencing log with shape (cycles, channels, height, width).
            peaks (numpy.ndarray): Array representing peaks.
            df_reads (pandas.DataFrame): DataFrame containing reads information.
            barcode_table (pandas.DataFrame): DataFrame containing barcode information.
            sbs_cycles (list): List of SBS cycles.
            shape (tuple, optional): Shape of the sequencing data. Defaults to (1024, 1024).

        Returns:
            numpy.ndarray: Annotated SBS log with additional features.

        Note:
            Assumes that the `ops.annotate.annotate_bases()` and `ops.annotate.annotate_points()` functions are available.
        """
        # Define a lambda function to extract prefixes from barcodes
        barcode_to_prefix = lambda x: ''.join(x[c - 1] for c in sbs_cycles)
        # Extract prefixes from barcodes
        barcodes = [barcode_to_prefix(x) for x in barcode_table['sgRNA']]

        # Mark reads as mapped or unmapped based on barcodes
        df_reads['mapped'] = df_reads['barcode'].isin(barcodes)

        # Define structuring elements for morphological operations
        plus = [[0, 1, 0],
                [1, 1, 1],
                [0, 1, 0]]
        xcross = [[1, 0, 1],
                  [0, 1, 0],
                  [1, 0, 1]]
        notch = [[1, 1, 1],
                 [1, 1, 1],
                 [1, 1, 0]]
        notch2 = [[1, 1, 1],
                  [1, 1, 1],
                  [0, 1, 0]]
        top_right = [[0, 0, 0],
                     [0, 0, 0],
                     [1, 0, 0]]
        donut = [[1, 1, 1],
                 [1, 1, 1],
                 [1, 1, 1]]
        
        
        # Annotate bases for mapped and unmapped reads
        f = ops.annotate.annotate_bases
        base_labels = f(df_reads.query('mapped'), selem=plus, shape=shape)
        base_labels += f(df_reads.query('~mapped'), selem=xcross, shape=shape)

        # Annotate quality scores
        Q_min = ops.annotate.annotate_points(df_reads, 'Q_min', selem=top_right, shape=shape)
        Q_30 = (Q_min * 30).astype(int)

        # Create a "donut" around each peak indicating the peak intensity and other elements, maybe need to fix, review with Owen
        peaks_donut = skimage.morphology.dilation(peaks, selem=donut)
        peaks_donut[peaks > 0] = 0
        peaks_donut[base_labels.sum(axis=0) > 0] = 0
        peaks_donut[Q_30 > 0] = 0

        # Get dimensions of the SBS log
        cycles, channels, height, width = log.shape

        # Create an array to store annotated log with additional features
        annotated = np.zeros((cycles, channels + 2, height, width), dtype=np.uint16)

        # Copy original log to annotated log
        annotated[:, :channels] = log

        # Add annotated bases to the last channel
        annotated[:, channels] = base_labels

        # Add peaks donut to the second last channel
        annotated[:, channels + 1] = peaks_donut
               
        if return_channels=="both":
            return annotated
        elif return_channels=="reads":
            return annotated[:, [i for i in range(0, 6)], :, :]
        elif return_channels=="peaks":
            return annotated[:, [i for i in range(0, 5)] + [6], :, :]
    
    # PHENOTYPE FEATURE EXTRACTION

    @staticmethod
    def _extract_features(data, labels, wildcards, features=None, multichannel=False):
        """
        Extract features from the provided image data within labeled segmentation masks.

        Args:
            data (numpy.ndarray): Image data of dimensions (CHANNEL, I, J).
            labels (numpy.ndarray): Labeled segmentation mask defining objects to extract features from.
            wildcards (dict): Metadata to include in the output table, e.g., well, tile, etc.
            features (dict or None): Features to extract and their defining functions. Default is None.
            multichannel (bool): Flag indicating whether the data has multiple channels.

        Returns:
            pandas.DataFrame: Table of labeled regions in labels with corresponding feature measurements.
        """
        # Import necessary modules and feature functions
        from ops.features import features_basic
        features = features.copy() if features else dict()
        features.update(features_basic)

        # Choose appropriate feature table based on multichannel flag
        if multichannel:
            from ops.process import feature_table_multichannel as feature_table
        else:
            from ops.process import feature_table

        # Extract features using the feature table function
        df = feature_table(data, labels, features)

        # Add wildcard metadata to the DataFrame
        for k, v in sorted(wildcards.items()):
            df[k] = v

        return df

    @staticmethod
    def _extract_timelapse_features(data, labels, wildcards, features=None):
        """
        Extract features from timelapse data and combine with generic region features.

        Args:
            data (numpy.ndarray): Timelapse image data.
            labels (numpy.ndarray): Segmentation masks.
            wildcards (list): List of wildcards.
            features (list, optional): List of features to extract.

        Returns:
            pandas.DataFrame: DataFrame containing extracted features.
        """

        arr = []
        # Iterate over frames in the timelapse
        for i, (frame, labels_frame) in enumerate(zip(np.squeeze(data), np.squeeze(labels))):
            # Extract features for each frame and concatenate to the array
            arr += [(Snake._extract_features(frame, labels_frame, wildcards, features=features)
                     .assign(frame=i))]

        # Concatenate all extracted features and rename 'label' column to 'cell'
        return pd.concat(arr).rename(columns={'label': 'cell'})

    
    @staticmethod
    def _extract_features_bare(data, labels, features=None, wildcards=None, multichannel=False):
        """
        Extract features in dictionary and combine with generic region features.

        Args:
            data (numpy.ndarray): Image data of dimensions (CHANNEL, I, J).
            labels (numpy.ndarray): Labeled segmentation mask defining objects to extract features from.
            features (dict or None): Features to extract and their defining functions. Default is None.
            wildcards (dict or None): Metadata to include in the output table, e.g., well, tile, etc. Default is None.
            multichannel (bool): Flag indicating whether the data has multiple channels.

        Returns:
            pandas.DataFrame: Table of labeled regions in labels with corresponding feature measurements.
        """
        # Import necessary modules and feature functions
        from ops.process import feature_table
        features = features.copy() if features else dict()
        features.update({'label': lambda r: r.label})

        # Choose appropriate feature table based on multichannel flag
        if multichannel:
            from ops.process import feature_table_multichannel as feature_table
        else:
            from ops.process import feature_table

        # Extract features using the feature table function
        df = feature_table(data, labels, features)

        # Add wildcard metadata to the DataFrame if provided
        if wildcards is not None:
            for k, v in sorted(wildcards.items()):
                df[k] = v

        return df


    @staticmethod
    def _extract_named_features(data, labels, feature_names, wildcards):
        """
        Extract features specified by names and combine with generic region features.

        Args:
            data (numpy.ndarray): Image data of dimensions (CHANNEL, I, J).
            labels (numpy.ndarray): Labeled segmentation mask defining objects to extract features from.
            feature_names (list): List of feature names to extract.
            wildcards (dict): Metadata to include in the output table, e.g., well, tile, etc.

        Returns:
            pandas.DataFrame: Table of labeled regions in labels with corresponding feature measurements.
        """
        # Create a dictionary of features from the specified feature names
        features = ops.features.make_feature_dict(feature_names)

        # Extract features using the generic feature extraction function
        return Snake.extract_features(data, labels, wildcards, features)


    @staticmethod
    def _extract_named_cell_nucleus_features(
            data, cells, nuclei, cell_features, nucleus_features, wildcards,
            autoscale=True, join='inner'):
        """
        Extract named features for cell and nucleus labels and join the results.

        Args:
            data (numpy.ndarray): Image data of dimensions (CHANNEL, I, J).
            cells (numpy.ndarray): Labeled segmentation mask defining cell objects.
            nuclei (numpy.ndarray): Labeled segmentation mask defining nucleus objects.
            cell_features (list): List of feature names to extract for cells.
            nucleus_features (list): List of feature names to extract for nuclei.
            wildcards (dict): Metadata to include in the output table.
            autoscale (bool): Scale the cell and nuclei mask dimensions to match the data. Default is True.
            join (str): Type of join to perform when merging dataframes. Default is 'inner'.

        Returns:
            pandas.DataFrame: Table of labeled cell regions with corresponding feature measurements.
        """
        # Scale the cell and nuclei masks dimensions to match the data if autoscale is True
        if autoscale:
            cells = ops.utils.match_size(cells, data[0])
            nuclei = ops.utils.match_size(nuclei, data[0])

        # Check if 'label' is in both cell_features and nucleus_features
        assert 'label' in cell_features and 'label' in nucleus_features

        # Extract features for cells and nuclei
        df_phenotype = pd.concat([
            Snake.extract_named_features(data, cells, cell_features, {}).set_index('label')
            .rename(columns=lambda x: x + '_cell'),
            Snake.extract_named_features(data, nuclei, nucleus_features, {}).set_index('label')
            .rename(columns=lambda x: x + '_nucleus'),
        ], join=join, axis=1).reset_index().rename(columns={'label': 'cell'})

        # Add wildcard metadata to the DataFrame
        for k, v in sorted(wildcards.items()):
            df_phenotype[k] = v

        return df_phenotype

    @staticmethod
    def _extract_named_cell_nucleus_cytoplasm_features(
            data, cells, nuclei, cytoplasm, cell_features, nucleus_features, cytoplasm_features, wildcards, 
            autoscale=True, join='inner'):
        """
        Extract named features for cell, nucleus, and cytoplasm labels, and join the results.

        Parameters:
        data (ndarray): The primary data array.
        cells (ndarray): A 2D array representing the cells regions.
        nuclei (ndarray): A 2D array representing the nuclei regions.
        cytoplasm (ndarray): A 2D array representing the cytoplasm regions.
        cell_features (list): List of features to extract from cell regions.
        nucleus_features (list): List of features to extract from nucleus regions.
        cytoplasm_features (list): List of features to extract from cytoplasm regions.
        wildcards (dict): A dictionary of additional metadata to add to the output.
        autoscale (bool): If True, scale the cell and nuclei mask dimensions to match the data. Default is True.
        join (str): The type of join to use when merging features ('inner', 'outer', etc.). Default is 'inner'.

        Returns:
        DataFrame: A pandas DataFrame containing the extracted features, merged by the specified join type.
        """
        if autoscale:
            # Scale the cell, nuclei, and cytoplasm arrays to match the dimensions of the data array
            cells = ops.utils.match_size(cells, data[0])
            nuclei = ops.utils.match_size(nuclei, data[0])
            cytoplasm = ops.utils.match_size(cytoplasm, data[0])

        # Ensure that 'label' is a feature in the cell, nucleus, and cytoplasm feature lists
        assert 'label' in cell_features and 'label' in nucleus_features and 'label' in cytoplasm_features

        # Extract features for cells, nuclei, and cytoplasm, and rename the columns to distinguish them
        df_phenotype = pd.concat([
            Snake._extract_named_features(data, cells, cell_features, {})
                .set_index('label').rename(columns=lambda x: x + '_cell'),
            Snake._extract_named_features(data, nuclei, nucleus_features, {})
                .set_index('label').rename(columns=lambda x: x + '_nucleus'),
            Snake._extract_named_features(data, cytoplasm, cytoplasm_features, {})
                .set_index('label').rename(columns=lambda x: x + '_cytoplasm'),
        ], join=join, axis=1).reset_index().rename(columns={'label': 'cell'})

        # Add wildcard metadata to the DataFrame
        for k, v in sorted(wildcards.items()):
            df_phenotype[k] = v

        return df_phenotype
    
    
    @staticmethod
    def _extract_phenotype_nuclei_cells(data_phenotype, nuclei, cells, features_n, features_c, wildcards, columns=None,
                                         multichannel=False):
        """
        Extract phenotype features from nuclei and cell segmentation masks.

        Args:
            data_phenotype (pandas.DataFrame): Phenotype data.
            nuclei (numpy.ndarray): Nuclei segmentation masks.
            cells (numpy.ndarray): Cell segmentation masks.
            features_n (dict): Dictionary of nuclei features to extract.
            features_c (dict): Dictionary of cell features to extract.
            wildcards (dict): Dictionary of wildcards.
            columns (dict, optional): Custom column mapping for features.
            multichannel (bool, optional): Whether the data is multichannel.

        Returns:
            pandas.DataFrame: DataFrame containing extracted phenotype features.
        """

        # Check if there are no nuclei or cells
        if (nuclei.max() == 0) or (cells.max() == 0):
            return

        import ops.features

        # Use default column mapping if not provided
        if columns is None:
            columns = {c: c for c in set(features_n.keys()) | set(features_c.keys())}

        # Extract nuclei features
        df_n = (
            Snake._extract_features(data_phenotype, nuclei, wildcards=wildcards, features=features_n,
                                    multichannel=multichannel)
            .rename(columns=columns)
            .set_index(['label'] + list(wildcards.keys()))
            .add_prefix('nucleus_')
            .reset_index(level=list(range(1, len(wildcards) + 1)))
        )

        # Extract cell features
        df_c = (
            Snake._extract_features(data_phenotype, cells, features=features_c, wildcards=wildcards,
                                    multichannel=multichannel)
            .drop(columns=wildcards.keys())
            .rename(columns=columns)
            .set_index('label')
            .add_prefix('cell_')
        )

        # Concatenate nuclei and cell features, joining on 'label'
        # Inner join discards nuclei without corresponding cells
        df = (pd.concat([df_n, df_c], axis=1, join='inner')
              .reset_index())

        return (df
                .rename(columns={'label': 'cell'}))

    
    @staticmethod
    def _extract_phenotype_FR(data_phenotype, nuclei, wildcards):
        """
        Extract features for frameshift reporter phenotyped in DAPI, HA channels.

        Args:
            data_phenotype (numpy.ndarray): Phenotype data with dimensions (CHANNEL, I, J).
            nuclei (numpy.ndarray): Labeled segmentation mask defining nucleus objects.
            wildcards (dict): Metadata to include in the output table.

        Returns:
            pandas.DataFrame: Table of labeled cell regions with corresponding feature measurements.
        """
        # Import necessary modules and feature functions
        from ops.features import features_frameshift

        # Extract features using frameshift reporter feature functions
        df = Snake.extract_features(data_phenotype, nuclei, wildcards, features_frameshift)

        # Rename 'label' column to 'cell'
        df = df.rename(columns={'label': 'cell'})

        return df


    @staticmethod
    def _extract_phenotype_FR_myc(data_phenotype, nuclei, wildcards):
        """
        Extract features for frameshift reporter phenotyped in DAPI, HA, myc channels.

        Args:
            data_phenotype (numpy.ndarray): Phenotype data with dimensions (CHANNEL, I, J).
            nuclei (numpy.ndarray): Labeled segmentation mask defining nucleus objects.
            wildcards (dict): Metadata to include in the output table.

        Returns:
            pandas.DataFrame: Table of labeled cell regions with corresponding feature measurements.
        """
        # Import necessary modules and feature functions
        from ops.features import features_frameshift_myc

        # Extract features using frameshift reporter with myc feature functions
        df = Snake.extract_features(data_phenotype, nuclei, wildcards, features_frameshift_myc)

        # Rename 'label' column to 'cell'
        df = df.rename(columns={'label': 'cell'})

        return df


    @staticmethod
    def _extract_phenotype_translocation(data_phenotype, nuclei, cells, wildcards):
        """
        Extract features for translocation phenotype.

        Args:
            data_phenotype (numpy.ndarray): Phenotype data with dimensions (CHANNEL, I, J).
            nuclei (numpy.ndarray): Labeled segmentation mask defining nucleus objects.
            cells (numpy.ndarray): Labeled segmentation mask defining cell objects.
            wildcards (dict): Metadata to include in the output table.

        Returns:
            pandas.DataFrame or None: Table of labeled cell regions with corresponding feature measurements if nuclei and cells are found, else None.
        """
        # Check if both nuclei and cells exist
        if (nuclei.max() == 0) or (cells.max() == 0):
            return None

        # Import necessary modules and feature functions
        import ops.features

        # Define features for nuclear and cell compartments
        features_n = ops.features.features_translocation_nuclear
        features_c = ops.features.features_translocation_cell

        # Rename feature keys to differentiate between nuclear and cell features
        features_n = {k + '_nuclear': v for k, v in features_n.items()}
        features_c = {k + '_cell': v for k, v in features_c.items()}

        # Extract features for nuclear compartment
        df_n = (Snake.extract_features(data_phenotype, nuclei, wildcards, features_n)
                .rename(columns={'area': 'area_nuclear'}))

        # Extract features for cell compartment
        df_c = (Snake.extract_features(data_phenotype, cells, wildcards, features_c)
                .drop(['i', 'j'], axis=1)
                .rename(columns={'area': 'area_cell'}))

        # Inner join to discard nuclei without corresponding cells
        df = (pd.concat([df_n.set_index('label'), df_c.set_index('label')], axis=1, join='inner')
              .reset_index()
              .rename(columns={'label': 'cell'}))

        return df



    @staticmethod
    def _extract_phenotype_translocation_live(data, nuclei, wildcards):
        """
        Extract features for translocation phenotype from live data.

        Args:
            data (numpy.ndarray): Live data frames.
            nuclei (numpy.ndarray): Labeled segmentation mask defining nucleus objects.
            wildcards (dict): Metadata to include in the output table.

        Returns:
            pandas.DataFrame: Table of labeled cell regions with corresponding feature measurements for each frame.
        """
        def _extract_phenotype_translocation_simple(data, nuclei, wildcards):
            """
            Extract features for translocation phenotype in a single frame.

            Args:
                data (numpy.ndarray): Image data of dimensions (CHANNEL, I, J).
                nuclei (numpy.ndarray): Labeled segmentation mask defining nucleus objects.
                wildcards (dict): Metadata to include in the output table.

            Returns:
                pandas.DataFrame: Table of labeled cell regions with corresponding feature measurements.
            """
            # Import necessary modules and feature functions
            import ops.features
            features = ops.features.features_translocation_nuclear_simple

            # Extract features for translocation phenotype
            return (Snake._extract_features(data, nuclei, wildcards, features)
                    .rename(columns={'label': 'cell'}))

        # Define function to extract features for each frame
        extract = _extract_phenotype_translocation_simple
        arr = []

        # Iterate over each frame and extract features
        for i, (frame, nuclei_frame) in enumerate(zip(data, nuclei)):
            arr += [extract(frame, nuclei_frame, wildcards).assign(frame=i)]

        # Concatenate feature dataframes for each frame
        return pd.concat(arr)


    @staticmethod
    def _extract_phenotype_translocation_ring(data_phenotype, nuclei, wildcards, width=3):
        """
        Extract features for translocation phenotype in the ring surrounding nuclei.

        Args:
            data_phenotype (numpy.ndarray): Phenotype data with dimensions (CHANNEL, I, J).
            nuclei (numpy.ndarray): Labeled segmentation mask defining nucleus objects.
            wildcards (dict): Metadata to include in the output table.
            width (int): Width of the ring. Default is 3.

        Returns:
            pandas.DataFrame: Table of labeled cell regions with corresponding feature measurements for the ring area.
        """
        # Define a structuring element for morphological operations
        selem = np.ones((width, width))

        # Create a perimeter around the nuclei
        perimeter = skimage.morphology.dilation(nuclei, selem)
        perimeter[nuclei > 0] = 0

        # Create an inner area inside the perimeter
        inside = skimage.morphology.erosion(nuclei, selem)
        inner_ring = nuclei.copy()
        inner_ring[inside > 0] = 0

        # Extract features for the ring area
        return (Snake.extract_phenotype_translocation(data_phenotype, inner_ring, perimeter, wildcards)
                .rename(columns={'label': 'cell'}))


    @staticmethod
    def _extract_phenotype_minimal(data_phenotype, nuclei, wildcards):
        """
        Extracts minimal phenotype features from the provided phenotype data.

        Parameters:
        - data_phenotype (pandas DataFrame): DataFrame containing phenotype data.
        - nuclei (numpy array): Array containing nuclei information.
        - wildcards (dict): Metadata to include in output table.

        Returns:
        - pandas DataFrame: Extracted minimal phenotype features with cell labels.
        """
        # Call _extract_features method to extract features using provided phenotype data and nuclei information
        return (Snake._extract_features(data_phenotype, nuclei, wildcards, dict())
                # Rename the column containing labels to 'cell'
                .rename(columns={'label': 'cell'}))


    @staticmethod
    def _extract_phenotype_geom(labels, wildcards):
        """
        Extract geometric features for labeled regions.

        Args:
            labels (numpy.ndarray): Labeled segmentation mask defining regions.
            wildcards (dict): Metadata to include in the output table.

        Returns:
            pandas.DataFrame: Table of labeled regions with corresponding geometric feature measurements.
        """
        # Import necessary modules and feature functions
        from ops.features import features_geom

        # Extract geometric features
        return Snake._extract_features(labels, labels, wildcards, features_geom)
    
    @staticmethod
    def _extract_simple_nuclear_morphology(data_phenotype, nuclei, wildcards):
        """
        Extract simple nuclear morphology features.

        Args:
            data_phenotype (pandas.DataFrame): Phenotype data.
            nuclei (numpy.ndarray): Nuclei segmentation masks.
            wildcards (list): List of wildcards.

        Returns:
            pandas.DataFrame: DataFrame containing extracted nuclear morphology features.
        """

        import ops.morphology_features

        # Extract nuclear morphology features using specified features
        df = (Snake._extract_features(data_phenotype, nuclei, wildcards, ops.morphology_features.features_nuclear)
              .rename(columns={'label': 'cell'})
              )

        return df


    @staticmethod
    def _extract_phenotype_cp_old(data_phenotype, nuclei, cells, wildcards, nucleus_channels='all', cell_channels='all', channel_names=['dapi','tubulin','gh2ax','phalloidin']):
        """
        Extract phenotype features from cellprofiler-like data.

        Parameters:
        - data_phenotype (numpy.ndarray): Phenotype data array of shape (..., CHANNELS, I, J).
        - nuclei (numpy.ndarray): Nuclei segmentation data.
        - cells (numpy.ndarray): Cell segmentation data.
        - wildcards (dict): Dictionary containing wildcards.
        - nucleus_channels (str or list): List of nucleus channel indices to consider or 'all'.
        - cell_channels (str or list): List of cell channel indices to consider or 'all'.
        - channel_names (list): List of channel names.

        Returns:
        - pandas.DataFrame: DataFrame containing extracted phenotype features.
        """
        if nucleus_channels == 'all':
            try:
                nucleus_channels = list(range(data_phenotype.shape[-3]))
            except:
                nucleus_channels = [0]

        if cell_channels == 'all':
            try:
                cell_channels = list(range(data_phenotype.shape[-3]))
            except:
                cell_channels = [0]

        dfs = []

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data_phenotype = data_phenotype.astype(np.uint16)

        # Extract nucleus shape features
        dfs.append(Snake._extract_features(nuclei, nuclei, wildcards, ops.cp_emulator.shape_features)
                   .rename(columns=ops.cp_emulator.shape_columns)
                   .set_index('label')
                   .rename(columns=lambda x: 'nucleus_' + x if x not in wildcards.keys() else x)
                   )

        # Extract cell shape features
        dfs.append(Snake._extract_features_bare(cells, cells, ops.cp_emulator.shape_features)
                   .rename(columns=ops.cp_emulator.shape_columns)
                   .set_index('label')
                   .add_prefix('cell_')
                   )

        # Extract nucleus grayscale channel features
        dfs.extend([(Snake._extract_features_bare(data_phenotype[..., channel, :, :], nuclei, ops.cp_emulator.grayscale_features)
                     .rename(columns=ops.cp_emulator.grayscale_columns)
                     .set_index('label')
                     .add_prefix(f'nucleus_{channel_names[channel]}_')
                     )
                    for channel in nucleus_channels]
                   )

        # Extract cell grayscale channel features
        dfs.extend([(Snake._extract_features_bare(data_phenotype[..., channel, :, :], cells, ops.cp_emulator.grayscale_features)
                     .rename(columns=ops.cp_emulator.grayscale_columns)
                     .set_index('label')
                     .add_prefix(f'cell_{channel_names[channel]}_')
                     )
                    for channel in cell_channels]
                   )

        # Generate correlation column names for nucleus and cell
        nucleus_correlation_columns = {
            'colocalization_{}'.format(inner_num + outer_num * len(ops.cp_emulator.colocalization_columns)): col.format(
                first=channel_names[first], second=channel_names[second])
            for outer_num, (first, second) in enumerate(combinations(nucleus_channels, 2))
            for inner_num, col in enumerate(ops.cp_emulator.colocalization_columns)
        }

        nucleus_correlation_columns.update({
            'correlation_{}'.format(num): ops.cp_emulator.correlation_columns[0].format(
                first=channel_names[first], second=channel_names[second])
            for num, (first, second) in enumerate(combinations(nucleus_channels, 2))
        })

        nucleus_correlation_columns.update({
            'lstsq_slope_{}'.format(num): ops.cp_emulator.correlation_columns[1].format(
                first=channel_names[first], second=channel_names[second])
            for num, (first, second) in enumerate(combinations(nucleus_channels, 2))
        })

        cell_correlation_columns = {
            'colocalization_{}'.format(inner_num + outer_num * len(ops.cp_emulator.colocalization_columns)): col.format(
                first=channel_names[first], second=channel_names[second])
            for outer_num, (first, second) in enumerate(combinations(cell_channels, 2))
            for inner_num, col in enumerate(ops.cp_emulator.colocalization_columns)
        }

        cell_correlation_columns.update({
            'correlation_{}'.format(num): ops.cp_emulator.correlation_columns[0].format(
                first=channel_names[first], second=channel_names[second])
            for num, (first, second) in enumerate(combinations(cell_channels, 2))
        })

        cell_correlation_columns.update({
            'lstsq_slope_{}'.format(num): ops.cp_emulator.correlation_columns[1].format(
                first=channel_names[first], second=channel_names[second])
            for num, (first, second) in enumerate(combinations(cell_channels, 2))
        })

        # Extract nucleus channel correlations
        dfs.append(Snake._extract_features_bare(data_phenotype[..., nucleus_channels, :, :], nuclei,
                                                ops.cp_emulator.correlation_features)
                   .rename(columns=nucleus_correlation_columns)
                   .set_index('label')
                   .add_prefix('nucleus_')
                   )

        # Extract cell channel correlations
        dfs.append(Snake._extract_features_bare(data_phenotype[..., cell_channels, :, :], cells,
                                                ops.cp_emulator.correlation_features)
                   .rename(columns=cell_correlation_columns)
                   .set_index('label')
                   .add_prefix('cell_')
                   )

        # Extract nucleus and cell neighbors
        dfs.append(ops.cp_emulator.neighbor_measurements(nuclei, distances=[1])
                   .set_index('label')
                   .add_prefix('nucleus_')
                   )

        dfs.append(ops.cp_emulator.neighbor_measurements(cells, distances=[1])
                   .set_index('label')
                   .add_prefix('cell_')
                   )

        # Concatenate data frames and reset index
        return pd.concat(dfs, axis=1, join='outer', sort=True).reset_index()

    @staticmethod
    def _extract_phenotype_cp_ch(data_phenotype, nuclei, cells, wildcards, nucleus_channels='all', cell_channels='all', channel_names=['dapi','tubulin','gh2ax','phalloidin']):
        """
        Extract phenotype features from cellprofiler-like data with channel-specific functions.

        Parameters:
        - data_phenotype (numpy.ndarray): Phenotype data array of shape (..., CHANNELS, I, J).
        - nuclei (numpy.ndarray): Nuclei segmentation data.
        - cells (numpy.ndarray): Cell segmentation data.
        - wildcards (dict): Dictionary containing wildcards.
        - nucleus_channels (str or list): List of nucleus channel indices to consider or 'all'.
        - cell_channels (str or list): List of cell channel indices to consider or 'all'.
        - channel_names (list): List of channel names.

        Returns:
        - pandas.DataFrame: DataFrame containing extracted phenotype features.
        """
        from functools import partial

        # Check if all channels should be used
        if nucleus_channels == 'all':
            try:
                nucleus_channels = list(range(data_phenotype.shape[-3]))
            except:
                nucleus_channels = [0]

        if cell_channels == 'all':
            try:
                cell_channels = list(range(data_phenotype.shape[-3]))
            except:
                cell_channels = [0]

        dfs = []

        # Define dictionary to store nucleus and cell features and columns
        nucleus_features = {}
        nucleus_columns = {}
        cell_features = {}
        cell_columns = {}

        # Loop through nucleus channels to define features and columns
        for ch in nucleus_channels:
            # Grayscale features and columns
            nucleus_columns.update({k.format(channel=channel_names[ch]): v.format(channel=channel_names[ch])
                                    for k, v in ops.cp_emulator.grayscale_columns_ch.items()})
            for name, func in ops.cp_emulator.grayscale_features_ch.items():
                nucleus_features[f'{channel_names[ch]}_{name}'] = partial(func, ch=ch)
            # Colocalization and correlation features and columns
            for first, second in combinations(list(range(len(nucleus_channels))), 2):
                nucleus_columns.update({k.format(first=channel_names[first], second=channel_names[second]):
                                        v.format(first=channel_names[first], second=channel_names[second])
                                        for k, v in ops.cp_emulator.colocalization_columns_ch.items()})
                for name, func in ops.cp_emulator.correlation_features_ch.items():
                    nucleus_features[f'{name}_{channel_names[first]}_{channel_names[second]}'] = partial(func, ch1=first, ch2=second)

        nucleus_features.update(ops.cp_emulator.shape_features)
        nucleus_columns.update(ops.cp_emulator.shape_columns)

        # Loop through cell channels to define features and columns
        for ch in cell_channels:
            # Grayscale features and columns
            cell_columns.update({k.format(channel=channel_names[ch]): v.format(channel=channel_names[ch])
                                 for k, v in ops.cp_emulator.grayscale_columns_ch.items()})
            for name, func in ops.cp_emulator.grayscale_features_ch.items():
                cell_features[f'{channel_names[ch]}_{name}'] = partial(func, ch=ch)
            # Colocalization and correlation features and columns
            for first, second in combinations(list(range(len(cell_channels))), 2):
                cell_columns.update({k.format(first=channel_names[first], second=channel_names[second]):
                                     v.format(first=channel_names[first], second=channel_names[second])
                                     for k, v in ops.cp_emulator.colocalization_columns_ch.items()})
                for name, func in ops.cp_emulator.correlation_features_ch.items():
                    cell_features[f'{name}_{channel_names[first]}_{channel_names[second]}'] = partial(func, ch1=first, ch2=second)

        cell_features.update(ops.cp_emulator.shape_features)
        cell_columns.update(ops.cp_emulator.shape_columns)

        # Extract nucleus features
        dfs.append(Snake._extract_features(data_phenotype[..., nucleus_channels, :, :], nuclei, wildcards, nucleus_features)
                   .rename(columns=nucleus_columns)
                   .set_index('label')
                   .rename(columns=lambda x: 'nucleus_'+x if x not in wildcards.keys() else x)
                   )

        # Extract cell features
        dfs.append(Snake._extract_features_bare(data_phenotype[..., cell_channels, :, :], cells, cell_features)
                   .rename(columns=cell_columns)
                   .set_index('label')
                   .add_prefix('cell_')
                   )

        # Extract nucleus and cell neighbors
        dfs.append(ops.cp_emulator.neighbor_measurements(nuclei, distances=[1])
                   .set_index('label')
                   .add_prefix('nucleus_')
                   )

        dfs.append(ops.cp_emulator.neighbor_measurements(cells, distances=[1])
                   .set_index('label')
                   .add_prefix('cell_')
                   )

        # Concatenate data frames and reset index
        return pd.concat(dfs, axis=1, join='outer', sort=True).reset_index()

    @staticmethod
    def _extract_phenotype_cp_multichannel(data_phenotype, nuclei, cells, wildcards, cytoplasms=None,  
                                           nucleus_channels='all', cell_channels='all', cytoplasm_channels='all', 
                                           foci_channel=None,
                                           channel_names=['dapi','tubulin','gh2ax','phalloidin']):
        """
        Extract phenotype features from CellProfiler-like data with multi-channel functionality.

        Parameters:
        - data_phenotype (numpy.ndarray): Phenotype data array of shape (..., CHANNELS, I, J).
        - nuclei (numpy.ndarray): Nuclei segmentation data.
        - cells (numpy.ndarray): Cell segmentation data.
        - cytoplasms (numpy.ndarray, optional): Cytoplasmic segmentation data.
        - wildcards (dict): Dictionary containing wildcards.
        - nucleus_channels (str or list): List of nucleus channel indices to consider or 'all'.
        - cell_channels (str or list): List of cell channel indices to consider or 'all'.
        - foci_channel (int): Index of the channel containing foci information.
        - channel_names (list): List of channel names.

        Returns:
        - pandas.DataFrame: DataFrame containing extracted phenotype features.
        """
        # Check if all channels should be used
        if nucleus_channels == 'all':
            try:
                nucleus_channels = list(range(data_phenotype.shape[-3]))
            except:
                nucleus_channels = [0]

        if cell_channels == 'all':
            try:
                cell_channels = list(range(data_phenotype.shape[-3]))
            except:
                cell_channels = [0]
                
        if cytoplasm_channels == 'all':
            try:
                cytoplasm_channels = list(range(data_phenotype.shape[-3]))
            except:
                cytoplasm_channels = [0]

        dfs = []

        # Define features
        features = ops.cp_emulator.grayscale_features_multichannel
        features.update(ops.cp_emulator.correlation_features_multichannel)
        features.update(ops.cp_emulator.shape_features)

        # Define function to create column map
        def make_column_map(channels):
            columns = {}
            # Create columns for grayscale features
            for feat, out in ops.cp_emulator.grayscale_columns_multichannel.items():
                columns.update({f'{feat}_{n}': f'{channel_names[ch]}_{renamed}' 
                                for n, (renamed, ch) in enumerate(product(out, channels))})
            # Create columns for correlation features
            for feat, out in ops.cp_emulator.correlation_columns_multichannel.items():
                if feat == 'lstsq_slope':
                    iterator = permutations
                else:
                    iterator = combinations
                columns.update({f'{feat}_{n}': renamed.format(first=channel_names[first], second=channel_names[second])
                                for n, (renamed, (first, second)) in enumerate(product(out, iterator(channels, 2)))})
            # Add shape columns
            columns.update(ops.cp_emulator.shape_columns)
            return columns

        # Create column maps for nucleus and cell
        nucleus_columns = make_column_map(nucleus_channels)
        cell_columns = make_column_map(cell_channels)

        # Extract nucleus features
        dfs.append(Snake._extract_features(data_phenotype[..., nucleus_channels, :, :], nuclei, wildcards, features, multichannel=True)
                   .rename(columns=nucleus_columns)
                   .set_index('label')
                   .rename(columns=lambda x: 'nucleus_'+x if x not in wildcards.keys() else x)
                   )

        # Extract cell features
        dfs.append(Snake._extract_features(data_phenotype[..., cell_channels, :, :], cells, dict(), features, multichannel=True)
                   .rename(columns=cell_columns)
                   .set_index('label')
                   .add_prefix('cell_')
                   )

        # Extract cytoplasmic features if cytoplasms are provided
        if cytoplasms is not None:
            cytoplasmic_columns = make_column_map(cytoplasm_channels)
            dfs.append(Snake._extract_features(data_phenotype[..., cytoplasm_channels, :, :], cytoplasms, dict(), features, multichannel=True)
                       .rename(columns=cytoplasmic_columns)
                       .set_index('label')
                       .add_prefix('cytoplasm_')
                       )

        # Extract foci features if foci channel is provided
        if foci_channel is not None:
            foci = ops.process.find_foci(data_phenotype[..., foci_channel, :, :], remove_border_foci=True)
            dfs.append(Snake._extract_features_bare(foci, cells, features=ops.features.foci)
                       .set_index('label')
                       .add_prefix(f'cell_{channel_names[foci_channel]}_')
                       )

        # Extract nucleus and cell neighbors
        dfs.append(ops.cp_emulator.neighbor_measurements(nuclei, distances=[1])
                   .set_index('label')
                   .add_prefix('nucleus_')
                   )

        dfs.append(ops.cp_emulator.neighbor_measurements(cells, distances=[1])
                   .set_index('label')
                   .add_prefix('cell_')
                   )
        if cytoplasms is not None:
            dfs.append(ops.cp_emulator.neighbor_measurements(cytoplasms, distances=[1])
                       .set_index('label')
                       .add_prefix('cytoplasm_')
                      )

        # Concatenate data frames and reset index
        return pd.concat(dfs, axis=1, join='outer', sort=True).reset_index()

    # HASH
    
    @staticmethod
    def _merge_sbs_phenotype(sbs_tables, phenotype_tables, barcode_table, sbs_cycles, join='outer'):
        """
        Combine sequencing and phenotype tables with one row per cell, using key (well, tile, cell).
        This was used when merging was between images at the same magnification.

        Args:
            sbs_tables (list of pandas.DataFrame): List of sequencing tables.
            phenotype_tables (list of pandas.DataFrame): List of phenotype tables.
            barcode_table (pandas.DataFrame): Barcode table.
            sbs_cycles (list): List of cycle indices.
            join (str): Method of joining the tables. Default is 'outer'.

        Returns:
            pandas.DataFrame: Combined table with one row per cell.
            
        Note:
            The cell column labels must be the same in both tables (e.g., both 
            tables generated from the same cell or nuclei segmentation). The default method of joining
            (outer) preserves cells present in only the sequencing table or phenotype table.        
            The barcode table is then joined using its `barcode` column to the most abundant 
            (`cell_barcode_0`) and second-most abundant (`cell_barcode_1`) barcodes for each cell. 

        """
        # Ensure sbs_tables and phenotype_tables are lists
        if isinstance(sbs_tables, pd.DataFrame):
            sbs_tables = [sbs_tables]
        if isinstance(phenotype_tables, pd.DataFrame):
            phenotype_tables = [phenotype_tables]

        # Set columns for indexing
        cols = ['well', 'tile', 'cell']

        # Concatenate sequencing and phenotype tables
        df_sbs = pd.concat(sbs_tables).set_index(cols)
        df_phenotype = pd.concat(phenotype_tables).set_index(cols)
        df_combined = pd.concat([df_sbs, df_phenotype], join=join, axis=1).reset_index()

        # Generate prefixes from barcodes for joining
        barcode_to_prefix = lambda x: ''.join(x[c - 1] for c in sbs_cycles)
        df_barcodes = (barcode_table
                       .assign(prefix=lambda x: x['barcode'].apply(barcode_to_prefix))
                       .assign(duplicate_prefix=lambda x: x['prefix'].duplicated(keep=False))
                       )

        # Drop 'barcode' column if 'sgRNA' column is present
        if 'barcode' in df_barcodes and 'sgRNA' in df_barcodes:
            df_barcodes = df_barcodes.drop('barcode', axis=1)

        # Set barcode information for joining
        barcode_info = df_barcodes.set_index('prefix')

        # Join barcode information to combined table
        return (df_combined
                .join(barcode_info, on='cell_barcode_0')
                .join(barcode_info.rename(columns=lambda x: x + '_1'), on='cell_barcode_1')
                )

    
    @staticmethod
    def _merge_triangle_hash(df_0, df_1, alignment, threshold=2):
        """
        Merge two dataframes using triangle hashing. This is done after images at different 
        magnifications are hashed together.

        Args:
            df_0 (pandas.DataFrame): First dataframe.
            df_1 (pandas.DataFrame): Second dataframe.
            alignment (dict): Alignment parameters containing rotation and translation.
            threshold (int): Threshold value. Default is 2.

        Returns:
            pandas.DataFrame: Merged dataframe.
        """
        # Import necessary modules
        import ops.triangle_hash as th

        # Rename 'tile' column to 'site' in df_1
        df_1 = df_1.rename(columns={'tile': 'site'})

        # Build linear model
        model = th.build_linear_model(alignment['rotation'], alignment['translation'])

        # Merge dataframes using triangle hashing
        return th.merge_sbs_phenotype(df_0, df_1, model, threshold=threshold)


    # PARAMSEARCH    
    
    @staticmethod
    def _summarize_paramsearch_segmentation(data, segmentations):
        """
        Summarize parameter search results for segmentation.

        Args:
            data (numpy.ndarray): Array containing segmentation data.
            segmentations (list of numpy.ndarray): List of segmentation data arrays.

        Returns:
            numpy.ndarray: Summary array.
        """
        # Stack data and compute median
        summary = np.stack([data[0], np.median(data[1:], axis=0)] + segmentations)

        return summary


    @staticmethod
    def _summarize_paramsearch_reads(barcode_table, reads_tables, cells, sbs_cycles, figure_output):
        """
        Summarize parameter search results for read mapping.

        Args:
            barcode_table (pandas.DataFrame): Barcode table.
            reads_tables (list of pandas.DataFrame): List of read tables.
            cells (list of numpy.ndarray): List of cell segmentation masks.
            sbs_cycles (list): List of cycle indices.
            figure_output (str): File path for the output figure.

        Returns:
            pandas.DataFrame: Summary dataframe.
        """
        # Import necessary modules
        import matplotlib
        import seaborn as sns

        # Set matplotlib backend to 'Agg'
        matplotlib.use('Agg')

        # Define function to generate barcode prefixes
        barcode_to_prefix = lambda x: ''.join(x[c - 1] for c in sbs_cycles)

        # Extract unique barcodes from the barcode table
        barcodes = (barcode_table
                    .assign(prefix=lambda x: x['barcode'].apply(barcode_to_prefix))
                    ['prefix']
                    .pipe(set)
                    )

        # Compute the number of cells per frame
        n_cells = [(len(np.unique(labels))-1) for labels in cells]

        # Concatenate read tables and add total cells count to each frame's data
        df_reads = pd.concat(reads_tables)
        df_reads = pd.concat([df.assign(total_cells=cell_count) 
                              for cell_count, (_, df) in zip(n_cells, df_reads.groupby(['well', 'tile'], sort=False))
                             ])

        # Flag reads as mapped or unmapped based on barcodes
        df_reads['mapped'] = df_reads['barcode'].isin(barcodes)

        # Define function to compute summary statistics
        def summarize(df):
            return pd.Series({
                'mapped_reads': df['mapped'].value_counts()[True],
                'mapped_reads_within_cells': df.query('cell != 0')['mapped'].value_counts()[True],
                'mapping_rate': df['mapped'].value_counts(normalize=True)[True],
                'mapping_rate_within_cells': df.query('cell != 0')['mapped'].value_counts(normalize=True)[True],
                'average_reads_per_cell': len(df.query('cell != 0')) / df.iloc[0]['total_cells'],
                'average_mapped_reads_per_cell': len(df.query('(cell != 0) & (mapped)')) / df.iloc[0]['total_cells'],
                'cells_with_reads': df.query('(cell != 0)')['cell'].nunique(),
                'cells_with_mapped_reads': df.query('(cell != 0) & (mapped)')['cell'].nunique()
            })

        # Group data by parameters and apply summary function
        df_summary = df_reads.groupby(['well', 'tile', 'THRESHOLD_READS']).apply(summarize).reset_index()

        # Plotting
        fig, axes = matplotlib.pyplot.subplots(2, 1, figsize=(7, 10), sharex=True)
        axes_right = [ax.twinx() for ax in axes]

        sns.lineplot(data=df_summary, x='THRESHOLD_READS', y='mapping_rate', color='steelblue', ax=axes[0])
        sns.lineplot(data=df_summary, x='THRESHOLD_READS', y='mapped_reads', color='coral', ax=axes_right[0])
        sns.lineplot(data=df_summary, x='THRESHOLD_READS', y='mapping_rate_within_cells', color='steelblue', ax=axes[0])
        sns.lineplot(data=df_summary, x='THRESHOLD_READS', y='mapped_reads_within_cells', color='coral', ax=axes_right[0])

        axes[0].set_ylabel('Mapping rate', fontsize=16)
        axes_right[0].set_ylabel('Number of mapped reads', fontsize=16)
        axes[0].set_title('Read mapping', fontsize=18)

        sns.lineplot(data=df_summary, x='THRESHOLD_READS', y='average_reads_per_cell', color='steelblue', ax=axes[1])
        sns.lineplot(data=df_summary, x='THRESHOLD_READS', y='average_mapped_reads_per_cell', color='steelblue', ax=axes[1])
        sns.lineplot(data=df_summary, x='THRESHOLD_READS', y='cells_with_reads', color='coral', ax=axes_right[1])
        sns.lineplot(data=df_summary, x='THRESHOLD_READS', y='cells_with_mapped_reads', color='coral', ax=axes_right[1])

        axes[1].set_ylabel('Mean reads per cell', fontsize=16)
        axes_right[1].set_ylabel('Number of cells', fontsize=16)
        axes[1].set_title('Read mapping per cell', fontsize=18)

        [ax.get_lines()[1].set_linestyle('--') for ax in list(axes)+list(axes_right)]

        axes[0].legend(handles=axes[0].get_lines()+axes_right[0].get_lines(),
                       labels=['Mapping rate, all reads', 'Mapping rate, within cells', 'All mapped reads', 'Mapped reads within cells'], loc=7)
        axes[1].legend(handles=axes[1].get_lines()+axes_right[1].get_lines(),
                       labels=['Mean reads per cell', 'Mean mapped reads per cell', 'Cells with reads', 'Cells with mapped reads'], loc=1)

        axes[1].set_xlabel('THRESHOLD_READS', fontsize=16)
        axes[1].set_xticks(df_summary['THRESHOLD_READS'].unique()[::2])

        [ax.tick_params(axis='y', colors='steelblue') for ax in axes]
        [ax.tick_params(axis='y', colors='coral') for ax in axes_right]

        matplotlib.pyplot.savefig(figure_output, dpi=300, bbox_inches='tight')

        return df_summary
    
    # TIMELAPSE
    
    @staticmethod
    def _align_stage_drift(data, frames=10):
        """
        Correct minor stage drift across first frames of timelapse.

        Args:
            data (numpy.ndarray): Timelapse image data.
            frames (int, optional): Number of frames to use for drift correction.

        Returns:
            list: List of offsets for each frame.
        """

        offsets = []
        data = np.squeeze(data)  # Remove singleton dimensions
        for source, target in zip(data[:(frames - 1)], data[1:frames]):
            # Apply windowed alignment between consecutive frames
            windowed = Align.apply_window(np.array([target, source]), window=2)
            # Calculate and store the offset between frames
            offsets.append(windowed[1].translation - windowed[0].translation)

        return offsets


    @staticmethod
    def _track_live_nuclei(nuclei, tolerance_per_frame=5):
        """
        Track nuclei across frames in a live imaging dataset.

        Args:
            nuclei (numpy.ndarray): Labeled segmentation masks defining nuclei objects across frames.
            tolerance_per_frame (int): Maximum allowed motion between consecutive frames for tracking. Default is 5.

        Returns:
            numpy.ndarray: Labeled segmentation masks with nuclei objects tracked across frames.
        """
        # Check if there are any nuclei detected in each frame
        count = nuclei.max(axis=(-2, -1))
        if (count == 0).any():
            frames_with_no_nuclei = np.where(count == 0)
            error = 'No nuclei detected in frames: {}'.format(frames_with_no_nuclei)
            print(error)
            return np.zeros_like(nuclei)

        # Import necessary modules
        import ops.timelapse

        # Extract nuclei coordinates and initialize DataFrame
        arr = []
        for i, nuclei_frame in enumerate(nuclei):
            extract = Snake._extract_phenotype_minimal
            arr += [extract(nuclei_frame, nuclei_frame, {'frame': i})]
        df_nuclei = pd.concat(arr)

        # Track nuclei motion
        motion_threshold = len(nuclei) * tolerance_per_frame
        G = (df_nuclei
             .rename(columns={'cell': 'label'})
             .pipe(ops.timelapse.initialize_graph))

        cost, path = ops.timelapse.analyze_graph(G)
        relabel = ops.timelapse.filter_paths(cost, path, threshold=motion_threshold)
        nuclei_tracked = ops.timelapse.relabel_nuclei(nuclei, relabel)

        return nuclei_tracked

    @staticmethod
    def _relabel_trackmate(nuclei, df_trackmate, df_nuclei_coords):
        """
        Relabel nuclei segmentation masks based on TrackMate tracking data.

        Args:
            nuclei (numpy.ndarray): Nuclei segmentation masks.
            df_trackmate (pandas.DataFrame): DataFrame containing TrackMate tracking data.
            df_nuclei_coords (pandas.DataFrame): DataFrame containing nuclei coordinates.

        Returns:
            tuple: DataFrame containing relabeled nuclei data and relabeled nuclei segmentation masks.
        """

        import ops.timelapse

        # Merge nuclei coordinates with TrackMate tracking data
        df = (df_nuclei_coords
              .merge(df_trackmate[['id', 'track_id', 'cell', 'frame', 'parent_ids']], how='left', on=['frame', 'cell'])
              .fillna({'track_id': -1,
                       'parent_ids': '[]',
                       })
              )

        # Assign unique IDs to untracked nuclei
        missing = sorted(set(range(len(df))) - set(df['id']))
        df.loc[df['id'].isna(), 'id'] = missing

        # Format TrackMate data for relabeling
        df_relabel = ops.timelapse.format_trackmate(df[['id', 'cell', 'frame', 'parent_ids']])

        # Merge relabeling data with nuclei DataFrame
        df_relabel = (df
                      .merge(df_relabel[['id', 'relabel', 'parent_cell_0', 'parent_cell_1']], how='left', on='id')
                      .drop(columns=['id', 'parent_ids'])
                      )

        def relabel_frame(nuclei_frame, df_relabel_frame):
            """
            Relabel nuclei segmentation mask for a single frame.

            Args:
                nuclei_frame (numpy.ndarray): Nuclei segmentation mask for a single frame.
                df_relabel_frame (pandas.DataFrame): DataFrame containing relabeling data for the frame.

            Returns:
                numpy.ndarray: Relabeled nuclei segmentation mask for the frame.
            """
            nuclei_frame_ = nuclei_frame.copy()
            max_label = nuclei_frame.max() + 1
            labels = df_relabel_frame['cell'].tolist()
            relabels = df_relabel_frame['relabel'].tolist()
            table = np.zeros(nuclei_frame.max() + 1)
            table[labels] = relabels
            nuclei_frame_ = table[nuclei_frame_]
            return nuclei_frame_

        # Relabel nuclei segmentation masks for each frame
        relabeled = np.array([relabel_frame(nuclei_frame, df_relabel_frame)
                              for nuclei_frame, (_, df_relabel_frame) in zip(nuclei, df_relabel.groupby('frame'))])

        # Return relabeled nuclei data and segmentation masks
        return (df_relabel.drop(columns=['cell']).rename(columns={'relabel': 'cell'}), relabeled.astype(np.uint16))


    # SNAKEMAKE

    @staticmethod
    def add_method(class_, name, f):
        """
        Adds a static method to a class dynamically.

        Args:
            class_ (type): The class to which the method will be added.
            name (str): The name of the method.
            f (function): The function to be added as a static method.
        """
        # Convert the function to a static method
        f = staticmethod(f)

        # Dynamicaly add the method to the class
        exec('%s.%s = f' % (class_, name))


    @staticmethod
    def load_methods():
        """
        Dynamically loads methods to Snake class from its static methods.

        Uses reflection to get all static methods from the Snake class and adds them as regular methods to the class.
        """
        # Get all methods of the Snake class
        methods = inspect.getmembers(Snake)

        # Iterate over methods
        for name, f in methods:
            # Check if the method name is not a special method or a private method
            if name not in ('__doc__', '__module__') and name.startswith('_'):
                # Add the method to the Snake class
                Snake.add_method('Snake', name[1:], Snake.call_from_snakemake(f))


    @staticmethod
    def call_from_snakemake(f):
        """
        Wrap a function to accept and return filenames for image and table data, with additional arguments.

        Args:
            f (function): The original function.

        Returns:
            function: Wrapped function.
        """
        def g(**kwargs):

            # split keyword arguments into input (needed for function)
            # and output (needed to save result)
            input_kwargs, output_kwargs = restrict_kwargs(kwargs, f)

            load_kwargs = {}
            if 'maxworkers' in output_kwargs:
                load_kwargs['maxworkers'] = output_kwargs.pop('maxworkers')

            # load arguments provided as filenames
            input_kwargs = {k: load_arg(v,**load_kwargs) for k,v in input_kwargs.items()}

            results = f(**input_kwargs)

            if 'output' in output_kwargs:
                outputs = output_kwargs['output']
                
                if len(outputs) == 1:
                    results = [results]

                if len(outputs) != len(results):
                    error = '{0} output filenames provided for {1} results'
                    raise ValueError(error.format(len(outputs), len(results)))

                for output, result in zip(outputs, results):
                    save_output(output, result, **output_kwargs)

            else:
                return results 

        return functools.update_wrapper(g, f)



# 
    
Snake.load_methods()

# IO

def load_arg(x):
    """
    Try loading data from `x` if it is a filename or list of filenames.
    Otherwise just return `x`.

    Args:
        x (str or list): File name or list of file names.

    Returns:
        object: Loaded data if successful, otherwise returns the original argument.
    """
    # Define functions for loading one file and multiple files
    one_file = load_file
    many_files = lambda x: [load_file(f) for f in x]

    # Try loading from one file or multiple files
    for f in (one_file, many_files):
        try:
            return f(x)
        except (pd.errors.EmptyDataError, TypeError, IOError) as e:
            if isinstance(e, (TypeError, IOError)):
                # If not a file, probably a string argument
                pass
            elif isinstance(e, pd.errors.EmptyDataError):
                # If failed to load file, return None
                return None
            pass
    else:
        return x

def save_output(filename, data, **kwargs):
    """
    Saves `data` to `filename`.

    Guesses the save function based on the file extension. Saving as .tif passes on kwargs (luts, ...) from input.

    Args:
        filename (str): Name of the file to save.
        data: Data to be saved.
        **kwargs: Additional keyword arguments passed to the save function.

    Returns:
        None
    """
    filename = str(filename)

    # If data is None, save a dummy output to satisfy Snakemake
    if data is None:
        with open(filename, 'w') as fh:
            pass
        return

    # Determine the save function based on the file extension
    if filename.endswith('.tif'):
        return save_tif(filename, data, **kwargs)
    elif filename.endswith('.pkl'):
        return save_pkl(filename, data)
    elif filename.endswith('.csv'):
        return save_csv(filename, data)
    elif filename.endswith('.png'):
        return save_png(filename, data)
    elif filename.endswith('.hdf'):
        return save_hdf(filename, data)
    else:
        raise ValueError('Not a recognized filetype: ' + filename)


def load_csv(filename):
    """
    Load data from a CSV file using pandas.

    Args:
        filename (str): Name of the CSV file to load.

    Returns:
        pandas.DataFrame or None: Loaded DataFrame if data exists, otherwise None.
    """
    df = pd.read_csv(filename)
    if len(df) == 0:
        return None
    return df

def load_pkl(filename):
    """
    Load data from a pickle file using pandas.

    Args:
        filename (str): Name of the pickle file to load.

    Returns:
        pandas.DataFrame or None: Loaded DataFrame if data exists, otherwise None.
    """
    df = pd.read_pickle(filename)
    if len(df) == 0:
        return None

def load_tif(filename):
    """
    Load image stack from a TIFF file using ops.

    Args:
        filename (str): Name of the TIFF file to load.

    Returns:
        numpy.ndarray: Loaded image stack.
    """
    return ops.io.read_stack(filename)

def load_hdf(filename):
    """
    Load image from an HDF file using ops.

    Args:
        filename (str): Name of the HDF file to load.

    Returns:
        numpy.ndarray: Loaded image.
    """
    return ops.io_hdf.read_hdf_image(filename)

def save_csv(filename, df):
    """
    Save DataFrame to a CSV file using pandas.

    Args:
        filename (str): Name of the CSV file to save.
        df (pandas.DataFrame): DataFrame to be saved.

    Returns:
        None
    """
    df.to_csv(filename, index=None)

def save_pkl(filename, df):
    """
    Save DataFrame to a pickle file using pandas.

    Args:
        filename (str): Name of the pickle file to save.
        df (pandas.DataFrame): DataFrame to be saved.

    Returns:
        None
    """
    df.to_pickle(filename)

def save_tif(filename, data_, **kwargs):
    """
    Save image data to a TIFF file using ops.

    Args:
        filename (str): Name of the TIFF file to save.
        data_ (numpy.ndarray): Image data to be saved.
        **kwargs: Additional keyword arguments passed to ops.io.save_stack.

    Returns:
        None
    """
    kwargs, _ = restrict_kwargs(kwargs, ops.io.save_stack)
    kwargs['data'] = data_
    ops.io.save_stack(filename, **kwargs)

def save_hdf(filename, data_):
    """
    Save image data to an HDF file using ops.

    Args:
        filename (str): Name of the HDF file to save.
        data_ (numpy.ndarray): Image data to be saved.

    Returns:
        None
    """
    ops.io_hdf.save_hdf_image(filename, data_)

def save_png(filename, data_):
    """
    Save image data to a PNG file using skimage.

    Args:
        filename (str): Name of the PNG file to save.
        data_ (numpy.ndarray): Image data to be saved.

    Returns:
        None
    """
    skimage.io.imsave(filename, data_)

def restrict_kwargs(kwargs, f):
    """
    Partition kwargs into two dictionaries based on overlap with default arguments of function f.

    Args:
        kwargs (dict): Keyword arguments.
        f (function): Function.

    Returns:
        dict: Dictionary containing keyword arguments that overlap with function f's default arguments.
        dict: Dictionary containing keyword arguments that do not overlap with function f's default arguments.
    """
    f_kwargs = set(get_kwarg_defaults(f).keys()) | set(get_arg_names(f))
    keep, discard = {}, {}
    for key in kwargs.keys():
        if key in f_kwargs:
            keep[key] = kwargs[key]
        else:
            discard[key] = kwargs[key]
    return keep, discard

def load_file(filename):
    """
    Attempt to load a file.

    Args:
        filename (str): Path to the file.

    Returns:
        Loaded file object.
        
    Raises:
        TypeError: If filename is not a string.
        IOError: If file is not found or the file extension is not recognized.
    """
    if not isinstance(filename, str):
        raise TypeError("Filename must be a string.")
    if not os.path.isfile(filename):
        raise IOError(2, 'Not a file: {0}'.format(filename))
    if filename.endswith('.tif'):
        return load_tif(filename)
    elif filename.endswith('.pkl'):
        return load_pkl(filename)
    elif filename.endswith('.csv'):
        return load_csv(filename)
    else:
        raise IOError(filename)

def get_arg_names(f):
    """
    Get a list of regular and keyword argument names from function definition.

    Args:
        f (function): Function.

    Returns:
        list: List of argument names.
    """
    argspec = inspect.getargspec(f)
    if argspec.defaults is None:
        return argspec.args
    n = len(argspec.defaults)
    return argspec.args[:-n]

def get_kwarg_defaults(f):
    """
    Get the keyword argument defaults as a dictionary.

    Args:
        f (function): Function.

    Returns:
        dict: Dictionary containing keyword arguments and their defaults.
    """
    argspec = inspect.getargspec(f)
    if argspec.defaults is None:
        defaults = {}
    else:
        defaults = {k: v for k,v in zip(argspec.args[::-1], argspec.defaults[::-1])}
    return defaults

def load_well_tile_list(filename, include='all'):
    """Read and format a table of acquired wells and tiles for Snakemake.

    Parameters
    ----------
    filename : str, path object, or file-like object
        File path to table of acquired wells and tiles.

    include : str or list of lists, default "all"
        If "all", keeps all wells and tiles defined in the supplied table. If any
        other str, this is used as a query of the well-tile table to restrict
        which sites are analyzed. If a list of [well,tile] pair lists, restricts
        analysis to this defined set of fields-of-view.

    Returns
    -------
    wells : np.ndarray
        Array of included wells, should be zipped with `tiles`.

    tiles : np.ndarray
        Array of included tiles, should be zipped with `wells`.
    """
    # Read the file based on the extension
    if filename.endswith('pkl'):
        df_wells_tiles = pd.read_pickle(filename)
    elif filename.endswith('csv'):
        df_wells_tiles = pd.read_csv(filename)

    # Handle different cases for including wells and tiles
    if include == 'all':
        wells, tiles = df_wells_tiles[['well', 'tile']].values.T
    elif isinstance(include, list):
        df_wells_tiles['well_tile'] = df_wells_tiles['well'] + df_wells_tiles['tile'].astype(str)
        include_wells_tiles = [''.join(map(str, well_tile)) for well_tile in include]
        included_df = df_wells_tiles[df_wells_tiles['well_tile'].isin(include_wells_tiles)]
        wells, tiles = included_df[['well', 'tile']].values.T
    else:
        included_df = df_wells_tiles.query(include)
        wells, tiles = included_df[['well', 'tile']].values.T

    return wells, tiles


def remove_channels(data, remove_index):
    """
    Remove channel or list of channels from array of shape (..., CHANNELS, I, J).

    Parameters:
    - data (numpy array): Input array of shape (..., CHANNELS, I, J).
    - remove_index (int or list of ints): Index or indices of the channels to remove.

    Returns:
    - numpy array: Array with specified channels removed.
    """
    # Create a boolean mask for all channels
    channels_mask = np.ones(data.shape[-3], dtype=bool)
    # Set the values corresponding to channels in remove_index to False
    channels_mask[remove_index] = False
    # Apply the mask along the channel axis to remove specified channels
    data = data[..., channels_mask, :, :]
    return data


def processed_file(suffix, directory='process', magnification='10X', temp_tags=tuple()):
    """
    Format output file pattern.

    Parameters:
    - suffix (str): File extension or suffix for the output file.
    - directory (str): Directory where the output file will be stored. Default is 'process'.
    - magnification (str): Magnification level of the image. Default is '10X'.
    - temp_tags (tuple): Tuple of temporary tags used for identifying temporary files.

    Returns:
    - str: Formatted file pattern for the output file.
    """
    # Construct the file pattern using placeholders for well and tile
    file_pattern = f'{directory}/{magnification}_{{well}}_Tile-{{tile}}.{suffix}'
    # Check if the suffix is a temporary tag
    if suffix in temp_tags:
        # Import the temp function from snakemake.io and apply it to the file pattern
        from snakemake.io import temp
        file_pattern = temp(file_pattern)
    return file_pattern

def input_files(suffix, cycles, directory='input', magnification='10X'):
    """
    Generate input file paths.

    Parameters:
    - suffix (str): File extension or suffix for the input files.
    - cycles (list): List of cycle numbers indicating different time points or iterations.
    - directory (str): Directory where the input files are located. Default is 'input'.
    - magnification (str): Magnification level of the image. Default is '10X'.

    Returns:
    - list: Input file paths generated based on the provided parameters.
    """
    # Construct the file pattern using placeholders for cycle, well, and tile
    pattern = (f'{directory}/{magnification}_{{cycle}}/'
               f'{magnification}_{{cycle}}_{{{{well}}}}_Tile-{{{{tile}}}}.{suffix}')
    # Use the expand function to generate input file paths for each cycle
    return expand(pattern, cycle=cycles)

def initialize_paramsearch(config):
    """
    Initialize parameter search configurations based on the given settings.

    Parameters:
    - config (dict): Dictionary containing various configuration settings.

    Returns:
    - tuple: Updated configuration dictionary along with Paramspace objects.
    """
    from snakemake.utils import Paramspace
    from itertools import product

    if config['MODE'] == 'paramsearch_segmentation':
        # Segmentaton Mode: Setting up parameter spaces for nuclei segmentation and cell segmentation
        if isinstance(config['THRESHOLD_DAPI'], list):
            thresholds_dapi = config['THRESHOLD_DAPI']
        else:
            thresholds_dapi = np.arange(config['THRESHOLD_DAPI'] - 200, config['THRESHOLD_DAPI'] + 300, 100, dtype=int)

        if isinstance(config['NUCLEUS_AREA'][0], list):
            nucleus_areas = config['NUCLEUS_AREA']
        else:
            nucleus_areas = [config['NUCLEUS_AREA']]

        if isinstance(config['THRESHOLD_CELL'], list):
            thresholds_cell = config['THRESHOLD_CELL']
        else:
            thresholds_cell = np.arange(config['THRESHOLD_CELL'] - 200, config['THRESHOLD_CELL'] + 300, 100, dtype=int)

        # Creating data frames for nuclei segmentation and cell segmentation parameters
        df_nuclei_segmentation = pd.DataFrame([{'THRESHOLD_DAPI': t_dapi, 'NUCLEUS_AREA_MIN': n_area_min,
                                                'NUCLEUS_AREA_MAX': n_area_max}
                                               for t_dapi, (n_area_min, n_area_max) in
                                               product(thresholds_dapi, nucleus_areas)])

        df_cell_segmentation = pd.DataFrame(thresholds_cell, columns=['THRESHOLD_CELL'])

        # Creating Paramspace objects to manage parameter space and filename pattern generation
        nuclei_segmentation_paramspace = Paramspace(df_nuclei_segmentation,
                                                    filename_params=['THRESHOLD_DAPI', 'NUCLEUS_AREA_MIN',
                                                                      'NUCLEUS_AREA_MAX'])

        cell_segmentation_paramspace = Paramspace(df_cell_segmentation,
                                                  filename_params=['THRESHOLD_CELL'])

        # Setting up file patterns and tags for segmentation summary
        config['REQUESTED_FILES'] = []
        config['REQUESTED_TAGS'] = [f'segmentation_summary.{nuclei_segmentation_instance}.'
                                    f'{"_".join(cell_segmentation_paramspace.instance_patterns)}.tif'
                                    for nuclei_segmentation_instance in nuclei_segmentation_paramspace.instance_patterns]
        config['TEMP_TAGS'] = [f'nuclei.{nuclei_segmentation_paramspace.wildcard_pattern}.tif',
                               f'cells.{nuclei_segmentation_paramspace.wildcard_pattern}.'
                               f'{cell_segmentation_paramspace.wildcard_pattern}.tif']

        return config, nuclei_segmentation_paramspace, cell_segmentation_paramspace

    elif config['MODE'] == 'paramsearch_read-calling':
        # Read Calling Mode: Setting up parameter space for read calling
        if isinstance(config['THRESHOLD_READS'], list):
            thresholds_reads = config['THRESHOLD_READS']
        else:
            thresholds_reads = np.concatenate([np.array([10]), np.arange(50, 1050, 50, dtype=int)])

        # Creating data frame for read threshold parameters
        df_read_thresholds = pd.DataFrame(thresholds_reads, columns=['THRESHOLD_READS'])

        # Creating Paramspace object for read calling to manage parameter space and filename pattern generation
        read_calling_paramspace = Paramspace(df_read_thresholds, filename_params=['THRESHOLD_READS'])

        # Setting up file patterns and tags for read calling
        config['REQUESTED_FILES'] = ['paramsearch_read-calling.summary.csv', 'paramsearch_read-calling.summary.pdf']
        config['REQUESTED_TAGS'] = []
        config['TEMP_TAGS'] = [f'bases.{read_calling_paramspace.wildcard_pattern}.csv',
                               f'reads.{read_calling_paramspace.wildcard_pattern}.csv']

        return config, read_calling_paramspace

