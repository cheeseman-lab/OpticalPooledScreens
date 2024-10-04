from itertools import combinations, permutations, product

import numpy as np
import pandas as pd

import ops.process
import ops.in_situ
import ops.utils
from ops.constants import PREFIX
import ops.cp_emulator


class Snake_ph:

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
                Snake_ph._apply_illumination_correction,
                arr=data,
                correction=correction,
                backend=backend,
                n_jobs=n_jobs,
            )

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
        rgb = Snake_ph._prepare_cellpose(
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
        mask = (outline_mask(nuclei, direction="inner") > 0) + (
            outline_mask(cells, direction="inner") > 0
        )

        # Copy original data to annotated data
        annotated[:channels] = data

        # Add combined mask to the last channel
        annotated[channels] = mask

        return np.squeeze(annotated)

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
    def _extract_features_bare(
        data, labels, features=None, wildcards=None, multichannel=False
    ):
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
        features.update({"label": lambda r: r.label})

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
            Snake_ph._extract_features(data_phenotype, nuclei, wildcards, dict())
            # Rename the column containing labels to 'cell'
            .rename(columns={"label": "cell"})
        )

    @staticmethod
    def _extract_phenotype_cp_multichannel(
        data_phenotype,
        nuclei,
        cells,
        wildcards,
        cytoplasms=None,
        nucleus_channels="all",
        cell_channels="all",
        cytoplasm_channels="all",
        foci_channel=None,
        channel_names=["dapi", "tubulin", "gh2ax", "phalloidin"],
    ):
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
        if nucleus_channels == "all":
            try:
                nucleus_channels = list(range(data_phenotype.shape[-3]))
            except:
                nucleus_channels = [0]

        if cell_channels == "all":
            try:
                cell_channels = list(range(data_phenotype.shape[-3]))
            except:
                cell_channels = [0]

        if cytoplasm_channels == "all":
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
                columns.update(
                    {
                        f"{feat}_{n}": f"{channel_names[ch]}_{renamed}"
                        for n, (renamed, ch) in enumerate(product(out, channels))
                    }
                )
            # Create columns for correlation features
            for feat, out in ops.cp_emulator.correlation_columns_multichannel.items():
                if feat == "lstsq_slope":
                    iterator = permutations
                else:
                    iterator = combinations
                columns.update(
                    {
                        f"{feat}_{n}": renamed.format(
                            first=channel_names[first], second=channel_names[second]
                        )
                        for n, (renamed, (first, second)) in enumerate(
                            product(out, iterator(channels, 2))
                        )
                    }
                )
            # Add shape columns
            columns.update(ops.cp_emulator.shape_columns)
            return columns

        # Create column maps for nucleus and cell
        nucleus_columns = make_column_map(nucleus_channels)
        cell_columns = make_column_map(cell_channels)

        # Extract nucleus features
        dfs.append(
            Snake_ph._extract_features(
                data_phenotype[..., nucleus_channels, :, :],
                nuclei,
                wildcards,
                features,
                multichannel=True,
            )
            .rename(columns=nucleus_columns)
            .set_index("label")
            .rename(
                columns=lambda x: "nucleus_" + x if x not in wildcards.keys() else x
            )
        )

        # Extract cell features
        dfs.append(
            Snake_ph._extract_features(
                data_phenotype[..., cell_channels, :, :],
                cells,
                dict(),
                features,
                multichannel=True,
            )
            .rename(columns=cell_columns)
            .set_index("label")
            .add_prefix("cell_")
        )

        # Extract cytoplasmic features if cytoplasms are provided
        if cytoplasms is not None:
            cytoplasmic_columns = make_column_map(cytoplasm_channels)
            dfs.append(
                Snake_ph._extract_features(
                    data_phenotype[..., cytoplasm_channels, :, :],
                    cytoplasms,
                    dict(),
                    features,
                    multichannel=True,
                )
                .rename(columns=cytoplasmic_columns)
                .set_index("label")
                .add_prefix("cytoplasm_")
            )

        # Extract foci features if foci channel is provided
        if foci_channel is not None:
            foci = ops.process.find_foci(
                data_phenotype[..., foci_channel, :, :], remove_border_foci=True
            )
            dfs.append(
                Snake_ph._extract_features_bare(foci, cells, features=ops.features.foci)
                .set_index("label")
                .add_prefix(f"cell_{channel_names[foci_channel]}_")
            )

        # Extract nucleus and cell neighbors
        dfs.append(
            ops.cp_emulator.neighbor_measurements(nuclei, distances=[1])
            .set_index("label")
            .add_prefix("nucleus_")
        )

        dfs.append(
            ops.cp_emulator.neighbor_measurements(cells, distances=[1])
            .set_index("label")
            .add_prefix("cell_")
        )
        if cytoplasms is not None:
            dfs.append(
                ops.cp_emulator.neighbor_measurements(cytoplasms, distances=[1])
                .set_index("label")
                .add_prefix("cytoplasm_")
            )

        # Concatenate data frames and reset index
        return pd.concat(dfs, axis=1, join="outer", sort=True).reset_index()
