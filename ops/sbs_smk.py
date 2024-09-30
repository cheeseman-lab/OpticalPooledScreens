import warnings
import inspect
import functools
import os

import numpy as np
import pandas as pd
import skimage

from ops.process import Align
import ops.process
import ops.in_situ
import ops.utils
from ops.constants import PREFIX


class Snake_sbs:

    # ALIGNMENT AND SEGMENTATION

    @staticmethod
    def _apply_illumination_correction(
        data,
        correction=None,
        zproject=False,
        rolling_ball=False,
        rolling_ball_kwargs={},
        n_jobs=1,
        backend="threading",
    ):
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
                data = ops.process.subtract_background(
                    data, **rolling_ball_kwargs
                ).astype(np.uint16)

            return data

        else:
            # If n_jobs is greater than 1, apply illumination correction in parallel
            return ops.utils.applyIJ_parallel(
                Snake_sbs._apply_illumination_correction,
                arr=data,
                correction=correction,
                backend=backend,
                n_jobs=n_jobs,
            )

    @staticmethod
    def _align_SBS(
        data,
        method="DAPI",
        upsample_factor=2,
        window=2,
        cutoff=1,
        q_norm=70,
        align_within_cycle=True,
        cycle_files=None,
        keep_extras=False,
        n=1,
        remove_for_cycle_alignment=None,
    ):
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
                    arr.append(np.array(data[current : current + cycle]))
                current += cycle
            data = arr
            print(data[0].shape)
            print(data[1].shape)

        # Check if the number of channels varies across cycles
        if not all(x.shape == data[0].shape for x in data):
            # Keep only channels in common across all cycles
            channels = [x.shape[-3] if x.ndim > 2 else 1 for x in data]
            stacked = np.array([x[-min(channels) :] for x in data])

            # Add back extra channels if requested
            if keep_extras:
                extras = np.array(channels) - min(channels)
                arr = []
                for cycle, extra in enumerate(extras):
                    if extra != 0:
                        arr.extend([data[cycle][extra_ch] for extra_ch in range(extra)])
                propagate = np.array(arr)
                stacked = np.concatenate(
                    (np.array([propagate] * stacked.shape[0]), stacked), axis=1
                )
            else:
                extras = [0] * stacked.shape[0]
        else:
            stacked = np.array(data)
            extras = [0] * stacked.shape[0]

        assert stacked.ndim == 4, "Input data must have dimensions CYCLE, CHANNEL, I, J"

        # Align between SBS channels for each cycle
        aligned = stacked.copy()
        if align_within_cycle:
            align_it = lambda x: Align.align_within_cycle(
                x, window=window, upsample_factor=upsample_factor
            )
            aligned[:, n:] = np.array([align_it(x) for x in aligned[:, n:]])

        if method == "DAPI":
            # Align cycles using the DAPI channel
            aligned = Align.align_between_cycles(
                aligned, channel_index=0, window=window, upsample_factor=upsample_factor
            )
        elif method == "SBS_mean":
            # Calculate cycle offsets using the average of SBS channels
            sbs_channels = list(range(n, aligned.shape[1]))
            if remove_for_cycle_alignment is not None:
                sbs_channels.remove(remove_for_cycle_alignment)
            target = Align.apply_window(aligned[:, sbs_channels], window=window).max(
                axis=1
            )
            normed = Align.normalize_by_percentile(target, q_norm=q_norm)
            normed[normed > cutoff] = cutoff
            offsets = Align.calculate_offsets(normed, upsample_factor=upsample_factor)
            # Apply cycle offsets to each channel
            for channel in range(aligned.shape[1]):
                if channel >= sum(extras):
                    aligned[:, channel] = Align.apply_offsets(
                        aligned[:, channel], offsets
                    )
                else:
                    # Don't apply offsets to extra channel in the cycle it was acquired
                    extra_idx = list(np.cumsum(extras) > channel).index(True)
                    extra_offsets = np.array([offsets[extra_idx]] * aligned.shape[0])
                    aligned[:, channel] = Align.apply_offsets(
                        aligned[:, channel], extra_offsets
                    )
        else:
            raise ValueError(f'method "{method}" not implemented')

        return aligned

    @staticmethod
    def _prepare_cellpose(
        data, dapi_index, cyto_index, logscale=True, log_kwargs=dict()
    ):
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
            cyto = image_log_scale(cyto, **log_kwargs)
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
        kwargs = dict(
            threshold=lambda x: threshold,
            area_min=area_min,
            area_max=area_max,
            smooth=smooth,
            radius=radius,
        )

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
            print("segment_cells error -- no cells")
            cells = nuclei

        # Calculate the number of segmented cells (excluding background label)
        num_cells_segmented = len(np.unique(cells)) - 1
        print(f"Number of cells segmented: {num_cells_segmented}")

        # Return the segmented cells
        return cells

    @staticmethod
    def _segment_cell_2019(
        data, nuclei_threshold, nuclei_area_min, nuclei_area_max, cell_threshold
    ):
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
        nuclei = Snake_sbs._segment_nuclei(
            nuclei_data, nuclei_threshold, nuclei_area_min, nuclei_area_max
        )

        # Segment cells using the _segment_cells method
        cells = Snake_sbs._segment_cells(data, nuclei, cell_threshold)

        return nuclei, cells

    @staticmethod
    def _segment_cellpose(
        data,
        dapi_index,
        cyto_index,
        nuclei_diameter,
        cell_diameter,
        cellpose_kwargs=dict(),
        cells=True,
        cyto_model="cyto",
        reconcile="consensus",
        logscale=True,
        return_counts=False,
    ):
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
        log_kwargs = cellpose_kwargs.pop(
            "log_kwargs", dict()
        )  # Extract log_kwargs from cellpose_kwargs
        rgb = Snake_sbs._prepare_cellpose(
            data, dapi_index, cyto_index, logscale, log_kwargs=log_kwargs
        )

        counts = {}

        # Perform cell segmentation using Cellpose
        if cells:
            # Segment both nuclei and cells
            from ops.cellpose import segment_cellpose_rgb

            if return_counts:
                nuclei, cells, seg_counts = segment_cellpose_rgb(
                    rgb,
                    nuclei_diameter,
                    cell_diameter,
                    reconcile=reconcile,
                    return_counts=True,
                    **cellpose_kwargs,
                )
                counts.update(seg_counts)

            else:
                nuclei, cells = segment_cellpose_rgb(
                    rgb,
                    nuclei_diameter,
                    cell_diameter,
                    reconcile=reconcile,
                    **cellpose_kwargs,
                )

            counts["final_nuclei"] = len(np.unique(nuclei)) - 1
            counts["final_cells"] = len(np.unique(cells)) - 1
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

            nuclei = segment_cellpose_nuclei_rgb(
                rgb, nuclei_diameter, **cellpose_kwargs
            )
            counts["final_nuclei"] = len(np.unique(nuclei)) - 1
            print(f"Number of nuclei segmented: {counts['final_nuclei']}")
            counts_df = pd.DataFrame([counts])

            if return_counts:
                return nuclei, counts_df
            else:
                return nuclei

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

        # Apply Laplacian-of-Gaussian filter to the data using ops.process.log_ndi function from ops.process module
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
    def _extract_bases(maxed, peaks, cells, threshold_peaks, wildcards, bases="GTAC"):
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
            error = "Sequencing {0} bases {1} but maxed data had shape {2}"
            raise ValueError(error.format(len(bases), bases, maxed.shape))

        # "cycle 0" is reserved for phenotyping
        cycles = list(range(1, maxed.shape[0] + 1))
        bases = list(bases)

        # Extract base intensity values, labels, and positions
        values, labels, positions = ops.in_situ.extract_base_intensity(
            maxed, peaks, cells, threshold_peaks
        )

        # Format base intensity data into DataFrame
        df_bases = ops.in_situ.format_bases(values, labels, positions, cycles, bases)

        # Add wildcard metadata to the DataFrame
        for k, v in sorted(wildcards.items()):
            df_bases[k] = v

        return df_bases

    @staticmethod
    def _call_reads(
        df_bases, peaks=None, correction_only_in_cells=True, normalize_bases=True
    ):
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
            if len(df_bases.query("cell > 0")) == 0:
                return

        cycles = len(set(df_bases["cycle"]))
        channels = len(set(df_bases["channel"]))

        if normalize_bases:
            # Clean up and normalize base intensities, then perform median calling
            df_reads = (
                df_bases.pipe(ops.in_situ.clean_up_bases)
                .pipe(ops.in_situ.normalize_bases)
                .pipe(
                    ops.in_situ.do_median_call,
                    cycles,
                    channels=channels,
                    correction_only_in_cells=correction_only_in_cells,
                )
            )
        else:
            # Clean up bases and perform median calling without normalization
            df_reads = df_bases.pipe(ops.in_situ.clean_up_bases).pipe(
                ops.in_situ.do_median_call,
                cycles,
                channels=channels,
                correction_only_in_cells=correction_only_in_cells,
            )

        # Include peaks scores if available
        if peaks is not None:
            i, j = df_reads[["i", "j"]].values.T
            df_reads["peak"] = peaks[i, j]

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
            return df_reads.query("Q_min >= @q_min").pipe(ops.in_situ.call_cells)
        else:
            # Determine the experimental prefix length
            prefix_length = len(df_reads.iloc[0].barcode)

            # Add prefix to the pool DataFrame
            df_pool[PREFIX] = df_pool.apply(lambda x: x.sgRNA[:prefix_length], axis=1)

            # Filter reads by quality threshold and call cells mapping
            return df_reads.query("Q_min >= @q_min").pipe(
                ops.in_situ.call_cells_mapping, df_pool
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
        mask = (outline_mask(nuclei, direction="inner") > 0) + (
            outline_mask(cells, direction="inner") > 0
        )

        # Copy original data to annotated data
        annotated[:, :channels] = data

        # Add combined mask to the last channel
        annotated[:, channels] = mask

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
        base_labels = ops.annotate.annotate_bases(
            df_reads, width=3, shape=(height, width)
        )

        # Create an array to store annotated log
        annotated = np.zeros((cycles, channels + 1, height, width), dtype=np.uint16)

        # Copy original log to annotated log
        annotated[:, :channels] = log

        # Add annotated bases to the last channel
        annotated[:, channels] = base_labels

        return annotated

    @staticmethod
    def _annotate_bases_on_SBS_reads_peaks(
        log,
        peaks,
        df_reads,
        barcode_table,
        sbs_cycles,
        shape=(1024, 1024),
        return_channels="both",
    ):
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
        barcode_to_prefix = lambda x: "".join(x[c - 1] for c in sbs_cycles)
        # Extract prefixes from barcodes
        barcodes = [barcode_to_prefix(x) for x in barcode_table["sgRNA"]]

        # Mark reads as mapped or unmapped based on barcodes
        df_reads["mapped"] = df_reads["barcode"].isin(barcodes)

        # Define structuring elements for morphological operations
        plus = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
        xcross = [[1, 0, 1], [0, 1, 0], [1, 0, 1]]
        notch = [[1, 1, 1], [1, 1, 1], [1, 1, 0]]
        notch2 = [[1, 1, 1], [1, 1, 1], [0, 1, 0]]
        top_right = [[0, 0, 0], [0, 0, 0], [1, 0, 0]]
        donut = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]

        # Annotate bases for mapped and unmapped reads
        f = ops.annotate.annotate_bases
        base_labels = f(df_reads.query("mapped"), selem=plus, shape=shape)
        base_labels += f(df_reads.query("~mapped"), selem=xcross, shape=shape)

        # Annotate quality scores
        Q_min = ops.annotate.annotate_points(
            df_reads, "Q_min", selem=top_right, shape=shape
        )
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

        if return_channels == "both":
            return annotated
        elif return_channels == "reads":
            return annotated[:, [i for i in range(0, 6)], :, :]
        elif return_channels == "peaks":
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
        return (
            Snake_sbs._extract_features(data_phenotype, nuclei, wildcards, dict())
            # Rename the column containing labels to 'cell'
            .rename(columns={"label": "cell"})
        )

    # SNAKEMAKE WRAPPER FUNCTIONS

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

        # Dynamically add the method to the class
        exec("%s.%s = f" % (class_, name))

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
            if "maxworkers" in output_kwargs:
                load_kwargs["maxworkers"] = output_kwargs.pop("maxworkers")

            # load arguments provided as filenames
            input_kwargs = {
                k: load_arg(v, **load_kwargs) for k, v in input_kwargs.items()
            }

            results = f(**input_kwargs)

            if "output" in output_kwargs:
                outputs = output_kwargs["output"]

                if len(outputs) == 1:
                    results = [results]

                if len(outputs) != len(results):
                    error = "{0} output filenames provided for {1} results"
                    raise ValueError(error.format(len(outputs), len(results)))

                for output, result in zip(outputs, results):
                    save_output(output, result, **output_kwargs)

            else:
                return results

        return functools.update_wrapper(g, f)

    @staticmethod
    def load_methods():
        """
        Dynamically loads methods to Snake_preprocessing class from its static methods.

        Uses reflection to get all static methods from the Snake_preprocessing class and adds them as regular methods to the class.
        """
        # Get all methods of the Snake class
        methods = inspect.getmembers(Snake_sbs)

        # Iterate over methods
        for name, f in methods:
            # Check if the method name is not a special method or a private method
            if name not in ("__doc__", "__module__") and name.startswith("_"):
                # Add the method to the Snake class
                Snake_sbs.add_method(
                    "Snake_sbs", name[1:], Snake_sbs.call_from_snakemake(f)
                )


# call load methods to make class methods accessible from Snakemake
Snake_sbs.load_methods()


# HELPER FUNCTIONS


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


# SNAKEMAKE FUNCTIONS


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
        with open(filename, "w") as fh:
            pass
        return

    # Determine the save function based on the file extension
    if filename.endswith(".tif"):
        return save_tif(filename, data, **kwargs)
    elif filename.endswith(".pkl"):
        return save_pkl(filename, data)
    elif filename.endswith(".csv"):
        return save_csv(filename, data)
    elif filename.endswith(".png"):
        return save_png(filename, data)
    elif filename.endswith(".hdf"):
        return save_hdf(filename, data)
    else:
        raise ValueError("Not a recognized filetype: " + filename)


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
    kwargs["data"] = data_
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
        raise IOError(2, "Not a file: {0}".format(filename))
    if filename.endswith(".tif"):
        return load_tif(filename)
    elif filename.endswith(".pkl"):
        return load_pkl(filename)
    elif filename.endswith(".csv"):
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
    argspec = inspect.getfullargspec(f)
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
    argspec = inspect.getfullargspec(f)
    if argspec.defaults is None:
        defaults = {}
    else:
        defaults = {k: v for k, v in zip(argspec.args[::-1], argspec.defaults[::-1])}
    return defaults
