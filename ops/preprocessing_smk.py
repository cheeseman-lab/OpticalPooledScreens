import inspect
import functools
import os
import re

import pandas as pd
import numpy as np
from nd2reader import ND2Reader

import ops.filenames

class Snake_preprocessing:
    @staticmethod
    def _extract_metadata_well(file, parse_function_home=None, parse_function_dataset=None, parse_function_tiles=False):
        """
        Extracts metadata from a single ND2 file.

        Args:
            file (str): Path to the ND2 file.
            parse_function_home (str): Absolute path to the screen directory.
            parse_function_dataset (str): Dataset name within the screen directory.
            parse_function_tiles (bool): Whether to include tile information in the parsing function.

        Returns:
            pandas.DataFrame: Extracted metadata.
        """
        # Extract info from the file name
        info = parse_file(file, home=parse_function_home, dataset=parse_function_dataset, tiles=parse_function_tiles)
        
        # Create file description dictionary
        file_description = {k: v for k, v in sorted(info.items())}
        
        with ND2Reader(file) as images:
            # Extract metadata
            raw_metadata = images.parser._raw_metadata
            
            data = {
                'x_data': raw_metadata.x_data,
                'y_data': raw_metadata.y_data,
                'z_data': images.metadata['z_coordinates'],
                'pfs_offset': raw_metadata.pfs_offset,
                'filename': file,
            }
            
            df = pd.DataFrame(data)
            
            # If z_levels metadata exists, select every 4th row
            if 'z_levels' in images.metadata and set(images.metadata['z_levels']) == set(range(0, 4)):
                df = df.iloc[::4, :]
                
            df['field_of_view'] = images.metadata['fields_of_view']
            
            return df.reset_index(drop=True)

    @staticmethod
    def _extract_metadata_tile(files, parse_function_home=None, parse_function_dataset=None, parse_function_tiles=True):
        """
        Extracts metadata from a list of ND2 files.

        Args:
            files (list): List of paths to ND2 files.
            parse_function_home (str): Absolute path to the screen directory.
            parse_function_dataset (str): Dataset name within the screen directory.
            parse_function_tiles (bool): Whether to include tile information in the parsing function.

        Returns:
            pandas.DataFrame: Combined extracted metadata from all provided ND2 files.
        """
        all_metadata = []
        
        # Iterate through all provided files
        for file_path in files:
            if file_path.endswith('.nd2'):
                info = parse_file(file_path, home=parse_function_home, dataset=parse_function_dataset, tiles=parse_function_tiles)
                
                with ND2Reader(file_path) as images:
                    raw_metadata = images.parser._raw_metadata
                    
                    data = {
                        'x_data': raw_metadata.x_data,
                        'y_data': raw_metadata.y_data,
                        'z_data': images.metadata['z_coordinates'],
                        'pfs_offset': raw_metadata.pfs_offset,
                        'field_of_view': info.get('tile'),
                        'filename': file_path,
                    }
                    
                    df = pd.DataFrame(data)
                    
                    if 'z_levels' in images.metadata and set(images.metadata['z_levels']) == set(range(0, 4)):
                        df = df.iloc[::4, :]
                    all_metadata.append(df)
        
        # Combine all metadata
        if all_metadata:
            combined_metadata = pd.concat(all_metadata, ignore_index=True)
            
            # Convert 'field_of_view' to numeric, coercing any non-numeric values to NaN
            combined_metadata['field_of_view'] = pd.to_numeric(combined_metadata['field_of_view'], errors='coerce')
            # Sort by 'field_of_view' in ascending order
            combined_metadata = combined_metadata.sort_values('field_of_view').reset_index(drop=True)
            return combined_metadata
        else:
            print(f"No valid ND2 files found in the provided list.")
            return pd.DataFrame()  # Return an empty DataFrame if no files were processed

    @staticmethod
    def _convert_to_tif_tile(file, channel_order_flip=False, parse_function_home=None, parse_function_dataset=None, parse_function_tiles=True, parse_function_channels=False, zstacks=1):
        """
        Converts a single ND2 file with one field of view and multiple channels to a multidimensional numpy array.

        Args:
            file (str): Path to the ND2 file.
            channel_order_flip (bool): If True, reverses the order of channels. Defaults to False.
            parse_function_home (str): Absolute path to the screen directory for file parsing.
            parse_function_dataset (str): Dataset name within the screen directory for file parsing.
            parse_function_tiles (bool): Whether to include tile parsing. Defaults to True.
            zstacks (int): Number of z-stacks to keep. If 1, performs max projection. Defaults to 1.

        Returns:
            tuple: A tuple containing the numpy array of the image and the file description dictionary.
        """
        # Extract file description using the provided parse_function
        file_description = parse_file(file, home=parse_function_home, dataset=parse_function_dataset, tiles=parse_function_tiles)

        with ND2Reader(file) as images:
            print(f"Available channels: {images.metadata['channels']}")
            print(f"Image axes: {images.axes}")

            # Determine the axes order (always include 'c' for channels)
            axes = 'cyx'
            if 'z' in images.axes:
                print("There is a z dimension")
                axes = 'zcyx'

            images.bundle_axes = axes
            
            # Get the single image (all channels)
            image = images[0]

            # Handle z-stacks: max project if 'z' is in axes
            if 'z' in axes:
                image = np.max(image, axis=0)  # Max projection along z-axis

            # Flip channel order if specified
            if channel_order_flip:
                image = np.flip(image, axis=0)  # Flip along first axis (channels)

            # Ensure the image is uint16
            image_array = np.array(image, dtype=np.uint16)

            print(f"Final image shape: {image_array.shape}")

        return image_array, file_description
    
    @staticmethod
    def _convert_to_tif_well(file, channel_order_flip=False, parse_function_home=None, parse_function_dataset=None, parse_function_tiles=False, parse_function_channels=True, separate_dapi=True):
        """
        Converts an ND2 image file to TIFF format, handling different image types (multichannel SBS, single-channel SBS, and single-channel multi-z PH)
        across multiple fields of view.

        Args:
            file (str): Path to the ND2 image file to be converted.
            channel_order_flip (bool): Optional; if True, reverses the order of channels in the output images. Defaults to False.
            parse_function_home (str): Required; the absolute path to the screen directory for file parsing.
            parse_function_dataset (str): Required; the dataset name within the screen directory for file parsing.
            parse_function_tiles (bool): Optional; specifies whether to include tile parsing. Defaults to True.
            parse_function_channels (bool): Optional; specifies whether to parse channels. Defaults to True.
            separate_dapi (bool): Optional; if True, returns DAPI and multichannel images separately for multichannel SBS. Defaults to True.

        Returns:
                dict: A dictionary where keys are field of view indices and values are tuples containing:
                      - For multichannel SBS: 
                        If separate_dapi is True: (full_image_array, dapi_image_array, file_description)
                        If separate_dapi is False: (full_image_array, file_description)
                      - For multichannel z-stack PH: (dapi_image, gfp_image, file_description)
                      - For single-channel SBS: (image_array, file_description)
                      - For single-channel multi-z PH: (image_array, file_description)
        """
        file_description = parse_file(file, home=parse_function_home, dataset=parse_function_dataset, tiles=parse_function_tiles, channels=parse_function_channels)
        print(f"File description: {file_description}")

        results = {}

        with ND2Reader(file) as images:
            print(f"Available channels: {images.metadata['channels']}")
            print(f"Image axes: {images.axes}")
            print(f"Number of fields of view: {images.sizes.get('v', 1)}")

            # Determine image type
            is_multichannel = len(images.metadata['channels']) > 1
            is_z_stack = 'z' in images.axes

            # Set bundle axes
            axes = 'zcyx' if (is_multichannel and is_z_stack) else ('cyx' if is_multichannel else ('zyx' if is_z_stack else 'yx'))
            images.bundle_axes = axes
            images.iter_axes = 'v'  # Iterate over fields of view

            for fov_index, image in enumerate(images):
                fov_description = file_description.copy()
                fov_description['tile'] = str(fov_index)

                if is_multichannel and is_z_stack:  # Multichannel z-stack PH case
                    image_max_projected = np.max(image, axis=0)  # Now shape is (channels, y, x)
                    image_max_projected = np.array(image_max_projected, dtype=np.uint16)
                    if channel_order_flip:
                        image_max_projected = np.flip(image_max_projected, axis=0)
                    print(f"FOV {fov_index}: Multichannel z-stack PH image shape after processing: {image_max_projected.shape}")
                    if separate_dapi:
                        dapi_image_array = image_max_projected[0]
                        gfp_image = image_max_projected[1]
                        results[fov_index] = (dapi_image, gfp_image, fov_description)
                    else:
                        results[fov_index] = (image_max_projected, fov_description)

                elif is_multichannel: # Multi-channel SBS
                    if channel_order_flip:
                        image = np.flip(image, axis=0)
                    full_image_array = np.array(image, dtype=np.uint16)
                    print(f"FOV {fov_index}: Multichannel SBS image shape: {full_image_array.shape}")
                    if separate_dapi:
                        dapi_image_array = full_image_array[0]
                        results[fov_index] = (full_image_array, dapi_image_array, fov_description)
                    else:
                        results[fov_index] = (full_image_array, fov_description)

                elif is_z_stack:  # Single-channel multi-z PH
                    image_array = np.max(image, axis=0)  # Max projection
                    image_array = np.array(image_array, dtype=np.uint16)
                    print(f"FOV {fov_index}: Single-channel PH image shape (after max projection): {image_array.shape}")
                    results[fov_index] = (image_array, fov_description)

                else:  # Single-channel SBS
                    image_array = np.array(image, dtype=np.uint16)
                    print(f"FOV {fov_index}: Single-channel SBS image shape: {image_array.shape}")
                    results[fov_index] = (image_array, fov_description)

        return results
        
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
        exec('%s.%s = f' % (class_, name))


    @staticmethod
    def load_methods():
        """
        Dynamically loads methods to Snake_preprocessing class from its static methods.

        Uses reflection to get all static methods from the Snake_preprocessing class and adds them as regular methods to the class.
        """
        # Get all methods of the Snake class
        methods = inspect.getmembers(Snake_preprocessing)

        # Iterate over methods
        for name, f in methods:
            # Check if the method name is not a special method or a private method
            if name not in ('__doc__', '__module__') and name.startswith('_'):
                # Add the method to the Snake class
                Snake_preprocessing.add_method('Snake_preprocessing', name[1:], Snake_preprocessing.call_from_snakemake(f))


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
    
Snake_preprocessing.load_methods()

def extract_cycle(filename):
    """
    Extracts the cycle number from a filename assuming that the file
    has a subdirectory /c*/ in its name.
    Args:
        filename (str): The filename containing the cycle information.
    Returns:
        str: The extracted cycle number in the format 'c{number}-SBS-{number}'.
    """
    # Use regex to find the cycle number
    match = re.search(r'/c(\d+)/', filename)
    if match:
        cycle = match.group(1)
        cycle_str = f'c{cycle}-SBS-{cycle}'
        return cycle_str
    else:
        raise ValueError(f"Could not extract cycle number from filename: {filename}")

def extract_well(full_filename):
    """
    Extracts the Well from the filename assuming that the file has Wells-XX_ or WellXX_ in its name.

    Args:
        full_filename (str): The full filename containing the well information.

    Returns:
        str: The extracted well information.
    """
    # Extract the filename from the full path
    short_fname = full_filename.split('/')[-1]
    
    # Check for 'Wells-' or 'Well' in the filename
    if 'Wells-' in short_fname:
        well_loc = short_fname.find('Wells-')
        well_prefix_length = 6  # Length of 'Wells-'
    else:
        well_loc = short_fname.find('Well')
        well_prefix_length = 4  # Length of 'Well'
    
    if well_loc == -1:
        raise ValueError("No 'Well' or 'Wells' found in the filename.")
    
    # Extract the well information
    well = str(short_fname[well_loc + well_prefix_length : short_fname.find('_', well_loc)])
    return well

def extract_tile(full_filename):
    """
    For files in which the ND2 is split up by FoV, extracts the tile by searching for Points-#### in the filename.

    Args:
        full_filename (str): The full filename containing the tile information.

    Returns:
        str: The extracted tile information.
    """
    # Extract the filename from the full path
    short_fname = full_filename.split('/')[-1]
    
    # Find the location of 'Points-' in the filename
    seq_loc = short_fname.find('Points-')
    
    if seq_loc == -1:
        raise ValueError("No 'Points-' found in the filename.")
    
    # Find the end of the sequence (next underscore after 'Points-')
    seq_end = short_fname.find('_', seq_loc)
    
    if seq_end == -1:
        raise ValueError("No underscore found after 'Points-' in the filename.")
    
    # Extract the sequence and remove leading zeros
    seq = short_fname[seq_loc + 7: seq_end].lstrip('0')
    
    # If all zeros were removed, return '0'
    if seq == '':
        seq = '0'
    
    return seq

def extract_plate(full_filename):
    """
    Extracts the plate information from the filename.

    Args:
        full_filename (str): The full filename containing the plate information.

    Returns:
        str: The extracted plate information.
    """
    # Extract the plate information (assumed to be the 6th element when split by '/')
    plate = full_filename.split('/')[5]
    return plate

def extract_channel(file_name):
    """
    Extracts the channel information from the filename.
    Args:
        file_name (str): The filename containing the channel information.
    Returns:
        str: The extracted channel information.
    """
    # Use regex to find the channel information
    match = re.search(r'Channel([\w\-\.]+)_', file_name)
    if match:
        return match.group(1)
    else:
        return 'Unknown'

def parse_file(filename, home, dataset, tiles=False, channels=False):
    """
    Extracts relevant information from file

    Args:
        filename (str): The filename to parse.
        home (str): The home directory path.
        dataset (str): The dataset name.

    Returns:
        dict: A dictionary containing parsed information from the filename.
    """
    file_description = {}
    file_description['home'] = home
    file_description['dataset'] = dataset
    file_description['ext'] = 'tif'
    file_description['well'] = extract_well(filename)

    if tiles:
        file_description['tile'] = extract_tile(filename)
    if channels:
        file_description['channel'] = extract_channel(filename)

    if 'input_ph' in filename.split('/'):
        file_description['mag'] = '20X'
        file_description['tag'] = 'phenotype'
        file_description['subdir'] = 'input_ph_tif'
    elif 'input_sbs' in filename.split('/'):
        file_description['mag'] = '10X'
        file_description['tag'] = 'sbs'
        file_description['cycle'] = extract_cycle(filename)
        file_description['subdir'] = 'input_sbs_tif'

    return file_description

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
        defaults = {k: v for k,v in zip(argspec.args[::-1], argspec.defaults[::-1])}
    return defaults