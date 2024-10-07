"""
Utility Functions for Image Processing and Data Analysis

This module provides a collection of utility functions for image processing, data manipulation, 
and analysis, primarily using NumPy and Pandas. It includes functions for:

1. DataFrame operations: sorting, grouping, applying functions to groups, and data transformation.
2. Image processing: tiling, montaging, trimming, and manipulating image stacks.
3. Parallel processing: applying functions to groups of data in parallel.
4. File I/O: reading and combining CSV files, handling image files.
5. Memoization: caching function results for improved performance.
6. Specialized image analysis: region properties, resizing, and multichannel image handling.

"""

import functools
import multiprocessing
from joblib import Parallel, delayed

import re
import os
import string
from itertools import product
from glob import glob
from collections.abc import Iterable

import decorator
from natsort import natsorted
import numpy as np
import pandas as pd

def combine_tables(tag, output_filetype='hdf', subdir_read='process', n_jobs=1, usecols=None, subdir_write=None):
    """
    Combines CSV files with a specific tag into a single output file.

    Args:
        tag (str): Tag to identify the CSV files to be combined.
        output_filetype (str, optional): Output file type ('hdf' or 'csv'). Defaults to 'hdf'.
        subdir_read (str, optional): Subdirectory to read input files from. Defaults to 'process'.
        n_jobs (int, optional): Number of parallel jobs to run. Defaults to 1.
        usecols (list, optional): List of columns to use from the CSV files. Defaults to None (all columns).
        subdir_write (str, optional): Subdirectory to write output file to. Defaults to None (same as subdir_read).

    Returns:
        None: Writes the combined data to a file.
    """
    from tqdm.notebook import tqdm

    # Get list of files matching the tag
    files = glob(os.path.join(subdir_read, f'*.{tag}.csv'))

    def get_file(f, usecols):
        try:
            return pd.read_csv(f, usecols=usecols)
        except pd.errors.EmptyDataError:
            pass

    # Read files in parallel or sequentially based on n_jobs
    if n_jobs != 1:
        from joblib import Parallel, delayed
        arr = Parallel(n_jobs=n_jobs)(delayed(get_file)(file, usecols) for file in tqdm(files))
    else:
        arr = [get_file(file, usecols) for file in files]

    # Combine all dataframes
    df = pd.concat(arr)
    
    # Set output directory
    if subdir_write is None:
        subdir_write = subdir_read
    output_path = os.path.join(subdir_write, f"{tag}.{output_filetype}")

    # Save combined data to file
    if output_filetype == 'csv':
        df.to_csv(output_path)
    else:
        df.to_hdf(output_path, tag, mode='w')

        
def format_input(input_table, n_jobs=1, **kwargs):
    """
    Processes an input table to format and combine image data.

    Args:
        input_table (str): Path to the input Excel file.
        n_jobs (int, optional): Number of parallel jobs to run. Defaults to 1.
        **kwargs: Additional keyword arguments for parallel processing.

    Returns:
        None: Processes and saves the formatted image data.
    """
    df = pd.read_excel(input_table)
    
    def process_site(output_file, df_input):
        # Stack images from different channels
        stacked = np.array([read(input_file) for input_file in df_input.sort_values('channel')['original filename']])
        save(output_file, stacked)
        
    # Process sites in parallel or sequentially based on n_jobs
    if n_jobs != 1:
        from joblib import Parallel, delayed
        Parallel(n_jobs=n_jobs, **kwargs)(delayed(process_site)(output_file, df_input) 
                                          for output_file, df_input in df.groupby('snakemake filename'))
    else:
        for output_file, df_input in df.groupby('snakemake filename'):
            process_site(output_file, df_input)
            
            
def memoize(active=True, copy_numpy=True):
    """
    Decorator for memoizing function results.

    Args:
        active (bool, optional): Whether memoization is active. Defaults to True.
        copy_numpy (bool, optional): Whether to copy numpy arrays in the cache. Defaults to True.

    Returns:
        function: Decorated function with memoization capabilities.
    """
    def inner(f):
        f_ = decorator.decorate(f, _memoize)

        keys = dict(active=active, copy_numpy=copy_numpy)
        f.keys = keys
        f_.keys = keys

        def reset():
            cache = {}
            f.cache = cache
            f_.cache = cache
        
        reset()
        f_.reset = reset

        return f_
    return inner


def _memoize(f, *args, **kwargs):
    """
    Internal function for memoization logic.

    Args:
        f (function): The function to be memoized.
        *args: Positional arguments to the function.
        **kwargs: Keyword arguments to the function.

    Returns:
        The memoized result of the function call.
    """
    if not f.keys['active']:
        return f(*args, **kwargs)

    key = str(args) + str(kwargs)
    if key not in f.cache:
        f.cache[key] = f(*args, **kwargs)

    # Copy numpy arrays unless disabled
    if isinstance(f.cache[key], np.ndarray):
        if f.keys['copy_numpy']:
            return f.cache[key].copy()
        else:
            return f.cache[key]

    return f.cache[key]
    

# PANDAS UTILS

def natsort_values(df, cols, ascending=True):
    """
    Sort DataFrame using natural sorting order.

    Args:
        df (pd.DataFrame): Input DataFrame.
        cols (str or list): Column(s) to sort by.
        ascending (bool, optional): Sort ascending vs. descending. Defaults to True.

    Returns:
        pd.DataFrame: Sorted DataFrame.
    """
    from natsort import index_natsorted
    if not isinstance(cols, list):
        cols = [cols]
    values = np.array([np.argsort(index_natsorted(df[c])) for c in cols]).T
    ix = (pd.DataFrame(values, columns=cols)
          .sort_values(cols, ascending=ascending)
          .index)
    return df.iloc[list(ix)].copy()

def bin_join(xs, symbol):
    """
    Join strings with a symbol, wrapping each in parentheses.

    Args:
        xs (iterable): Strings to join.
        symbol (str): Symbol to join with.

    Returns:
        str: Joined string.
    """
    symbol = ' ' + symbol + ' ' 
    return symbol.join('(%s)' % x for x in xs)

or_join  = functools.partial(bin_join, symbol='|')
and_join = functools.partial(bin_join, symbol='&')

def groupby_reduce_concat(gb, *args, **kwargs):
    """
    Apply multiple reduction operations to a grouped DataFrame.

    Args:
        gb (pd.DataFrameGroupBy): Grouped DataFrame.
        *args: Names of reduction operations to apply.
        **kwargs: Custom reduction operations.

    Returns:
        pd.DataFrame: DataFrame with results of reduction operations.
    """
    for arg in args:
        kwargs[arg] = arg
    reductions = {'mean': lambda x: x.mean(),
                  'min': lambda x: x.min(),
                  'max': lambda x: x.max(),
                  'median': lambda x: x.median(),
                  'std': lambda x: x.std(),
                  'sem': lambda x: x.sem(),
                  'size': lambda x: x.size(),
                  'count': lambda x: x.size(),
                  'sum': lambda x: x.sum(),
                  'sum_int': lambda x: x.sum().astype(int),
                  'first': lambda x: x.nth(0),
                  'second': lambda x: x.nth(1)}
    
    for arg in args:
        if arg in reductions:
            kwargs[arg] = arg

    arr = []
    for name, f in kwargs.items():
        if callable(f):
            arr += [gb.apply(f).rename(name)]
        else:
            arr += [reductions[f](gb).rename(name)]

    return pd.concat(arr, axis=1).reset_index()

def groupby_histogram(df, index, column, bins, cumulative=False, normalize=False):
    """
    Create a histogram for grouped data.

    Args:
        df (pd.DataFrame): Input DataFrame.
        index (str or list): Column(s) to group by.
        column (str): Column to create histogram for.
        bins (array-like): Bin edges for histogram.
        cumulative (bool, optional): If True, calculate cumulative histogram. Defaults to False.
        normalize (bool, optional): If True, normalize histogram. Defaults to False.

    Returns:
        pd.DataFrame: Histogram data.
    """
    maybe_cumsum = lambda x: x.cumsum(axis=1) if cumulative else x
    maybe_normalize = lambda x: x.div(x.sum(axis=1), axis=0) if normalize else x
    column_bin = column + '_bin'
    if cumulative and normalize:
        new_col = 'csum_fraction'
    elif cumulative and not normalize:
        new_col = 'csum'
    elif not cumulative and normalize:
        new_col = 'fraction'
    else:
        new_col = 'count'

    column_value = column + ('_csum' if cumulative else '_count')
    bins = np.array(bins)
    return (df
        .assign(dummy=1)
        .assign(bin=bins[np.digitize(df[column], bins) - 1])
        .pivot_table(index=index, columns='bin', values='dummy', 
                     aggfunc='sum')
        .reindex(labels=list(bins), axis=1)
        .fillna(0).astype(int)
        .pipe(maybe_cumsum)
        .pipe(maybe_normalize)
        .stack().rename(new_col)
        .reset_index()
           )

def groupby_apply2(df_1, df_2, cols, f, tqdm=True):
    """
    Apply a function to paired groups from two DataFrames.

    Args:
        df_1 (pd.DataFrame): First DataFrame.
        df_2 (pd.DataFrame): Second DataFrame.
        cols (list): Columns to group by.
        f (callable): Function to apply to each pair of groups.
        tqdm (bool, optional): If True, show progress bar. Defaults to True.

    Returns:
        pd.DataFrame: Concatenated results of applying f to each pair of groups.
    """
    d_1 = {k: v for k,v in df_1.groupby(cols)}
    d_2 = {k: v for k,v in df_2.groupby(cols)}

    if tqdm:
        from tqdm.auto import tqdm
        progress = tqdm
    else:
        progress = lambda x: x

    arr = []
    for k in progress(d_1):
        arr.append(f(d_1[k], d_2[k]))
    
    return pd.concat(arr)    

def groupby_apply_norepeat(gb, f, *args, **kwargs):
    """
    Apply a function to each group in a GroupBy object without repeating on the first group.

    Args:
        gb (pd.DataFrameGroupBy): GroupBy object.
        f (callable): Function to apply to each group.
        *args: Positional arguments to pass to f.
        **kwargs: Keyword arguments to pass to f.

    Returns:
        pd.DataFrame: Concatenated results of applying f to each group.
    """
    arr = []
    for _, df in gb:
        arr += [f(df, *args, **kwargs)]
    return pd.concat(arr)

def ndarray_to_dataframe(values, index):
    """
    Convert a numpy array to a DataFrame with MultiIndex columns.

    Args:
        values (np.ndarray): Input array.
        index (list of tuples): List of (name, levels) tuples for MultiIndex.

    Returns:
        pd.DataFrame: Resulting DataFrame.
    """
    names, levels  = zip(*index)
    columns = pd.MultiIndex.from_product(levels, names=names)
    df = pd.DataFrame(values.reshape(values.shape[0], -1), columns=columns)
    return df

def uncategorize(df, as_codes=False):
    """
    Convert categorical columns to non-categorical types.

    Args:
        df (pd.DataFrame): Input DataFrame.
        as_codes (bool, optional): If True, convert to category codes. Defaults to False.

    Returns:
        pd.DataFrame: DataFrame with uncategorized columns.
    """
    for col in df.select_dtypes(include=['category']).columns:
        if as_codes:
            df[col] = df[col].cat.codes
        else:
            df[col] = np.asarray(df[col])
    return df

def rank_by_order(df, groupby_columns):
    """
    Rank rows within groups based on their order.

    Args:
        df (pd.DataFrame): Input DataFrame.
        groupby_columns (str or list): Column(s) to group by.

    Returns:
        list: List of ranks (1-based).
    """
    return (df
        .groupby(groupby_columns).cumcount()
        .pipe(lambda x: list(x + 1))
        )

def flatten_cols(df, f='underscore'):
    """
    Flatten MultiIndex columns of a DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame with MultiIndex columns.
        f (str or callable, optional): Function to join column levels. Defaults to 'underscore'.

    Returns:
        pd.DataFrame: DataFrame with flattened column names.
    """
    if f == 'underscore':
        f = lambda x: '_'.join(str(y) for y in x if y != '')
    df = df.copy()
    df.columns = [f(x) for x in df.columns]
    return df

def vpipe(df, f, *args, **kwargs):
    """
    Apply a function to the values of a DataFrame and return a new DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.
        f (callable): Function to apply to DataFrame values.
        *args: Positional arguments to pass to f.
        **kwargs: Keyword arguments to pass to f.

    Returns:
        pd.DataFrame: Resulting DataFrame after applying f to values.
    """
    return pd.DataFrame(f(df.values, *args, **kwargs), 
                 columns=df.columns, index=df.index)

def cast_cols(df, int_cols=tuple(), float_cols=tuple(), str_cols=tuple()):
    """
    Cast columns of a DataFrame to specified types.

    Args:
        df (pd.DataFrame): Input DataFrame.
        int_cols (tuple, optional): Columns to cast to int. Defaults to tuple().
        float_cols (tuple, optional): Columns to cast to float. Defaults to tuple().
        str_cols (tuple, optional): Columns to cast to str. Defaults to tuple().

    Returns:
        pd.DataFrame: DataFrame with casted columns.
    """
    return (df
           .assign(**{c: df[c].astype(int) for c in int_cols})
           .assign(**{c: df[c].astype(float) for c in float_cols})
           .assign(**{c: df[c].astype(str) for c in str_cols})
           )

def replace_cols(df, **kwargs):
    """
    Apply functions to update specified columns in a DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.
        **kwargs: Column names and functions to apply.

    Returns:
        pd.DataFrame: DataFrame with updated columns.
    """
    d = {}
    for k, v in kwargs.items():
        def f(x, k=k, v=v):
            return x[k].apply(v)
        d[k] = f
    return df.assign(**d)

def expand_sep(df, col, sep=','):
    """
    Expand a column with separated values into multiple rows.

    Args:
        df (pd.DataFrame): Input DataFrame.
        col (str): Column to expand.
        sep (str, optional): Separator used in the column. Defaults to ','.

    Returns:
        pd.DataFrame: Expanded DataFrame.
    """
    index, values = [], []
    for i, x in enumerate(df[col]):
        entries = [y.strip() for y in x.split(sep)]
        index += [i] * len(entries)
        values += entries
        
    return (pd.DataFrame(df.values[index], columns=df.columns)
     .assign(**{col: values}))

def csv_frame(files_or_search, progress=lambda x: x, add_file=None, file_pat=None, sort=True, 
              include_cols=None, exclude_cols=None, **kwargs):
    """
    Read multiple CSV files into a single DataFrame.

    Args:
        files_or_search (str or list): Glob pattern or list of file paths.
        progress (callable, optional): Progress bar function. Defaults to lambda x: x.
        add_file (str, optional): Column name to add file path. Defaults to None.
        file_pat (str, optional): Regex pattern to extract info from file names. Defaults to None.
        sort (bool, optional): Whether to sort the resulting DataFrame. Defaults to True.
        include_cols (list, optional): Columns to include. Defaults to None.
        exclude_cols (list, optional): Columns to exclude. Defaults to None.
        **kwargs: Additional arguments for pd.read_csv.

    Returns:
        pd.DataFrame: Combined DataFrame from all CSV files.
    """
    def read_csv(f):
        try:
            df = pd.read_csv(f, **kwargs)
        except pd.errors.EmptyDataError:
            return None
        if add_file is not None:
            df[add_file] = f
        if include_cols is not None:
            include_pat = include_cols if isinstance(include_cols, str) else '|'.join(include_cols)
            keep = [x for x in df.columns if re.match(include_pat, x)]
            df = df[keep]
        if exclude_cols is not None:
            exclude_pat = exclude_cols if isinstance(exclude_cols, str) else '|'.join(exclude_cols)
            keep = [x for x in df.columns if not re.match(exclude_pat, x)]
            df = df[keep]
        if file_pat is not None:
            match = re.match(f'.*?{file_pat}.*', f)
            if match is None:
                raise ValueError(f'{file_pat} failed to match {f}')
            if match.groupdict():
                for k,v in match.groupdict().items():
                    df[k] = v
            else:
                if add_file is None:
                    raise ValueError(f'must provide `add_file` or named groups in {file_pat}')
                first = match.groups()[0]
                df[add_file] = first
        return df
    
    if isinstance(files_or_search, str):
        files = natsorted(glob(files_or_search))
    else:
        files = files_or_search

    return pd.concat([read_csv(f) for f in progress(files)], sort=sort)

def gb_apply_parallel(df, cols, func, n_jobs=None, tqdm=True, backend='loky'):
    """
    Apply a function to groups of a DataFrame in parallel.

    Args:
        df (pd.DataFrame): Input DataFrame.
        cols (str or list): Column(s) to group by.
        func (callable): Function to apply to each group.
        n_jobs (int, optional): Number of parallel jobs. If None, uses (CPU count - 1). Defaults to None.
        tqdm (bool, optional): Whether to show progress bar. Defaults to True.
        backend (str, optional): Joblib parallel backend. Defaults to 'loky'.

    Returns:
        pd.DataFrame or pd.Series: Results of applying func to each group, combined into a single DataFrame or Series.
    """
    # Ensure cols is a list
    if isinstance(cols, str):
        cols = [cols]

    from joblib import Parallel, delayed

    # Set number of jobs if not specified
    if n_jobs is None:
        import multiprocessing
        n_jobs = multiprocessing.cpu_count() - 1

    # Group the DataFrame
    grouped = df.groupby(cols)
    names, work = zip(*grouped)

    # Add progress bar if requested
    if tqdm:
        from tqdm import tqdm_notebook 
        work = tqdm_notebook(work, str(cols))

    # Apply function in parallel
    results = Parallel(n_jobs=n_jobs, backend=backend)(delayed(func)(w) for w in work)

    # Process results based on their type
    if isinstance(results[0], pd.DataFrame):
        # For DataFrame results
        arr = []
        for labels, df in zip(names, results):
            if not isinstance(labels, Iterable):
                labels = [labels]
            if df is not None:
                (df.assign(**{c: l for c, l in zip(cols, labels)})
                    .pipe(arr.append))
        results = pd.concat(arr)
    elif isinstance(results[0], pd.Series):
        # For Series results
        if len(cols) == 1:
            results = (pd.concat(results, axis=1).T
                .assign(**{cols[0]: names}))
        else:
            labels = zip(*names)
            results = (pd.concat(results, axis=1).T
                .assign(**{c: l for c, l in zip(cols, labels)}))
    elif isinstance(results[0], dict):
        # For dict results
        results = pd.DataFrame(results, index=pd.Index(names, name=cols)).reset_index()

    return results

def add_fstrings(df, **format_strings):
    """
    Add new columns to a DataFrame using f-string-like formatting.

    Args:
        df (pd.DataFrame): Input DataFrame.
        **format_strings: Keyword arguments where keys are new column names and 
                          values are format strings using existing column names.

    Returns:
        pd.DataFrame: DataFrame with new formatted string columns added.

    Example:
        df.pipe(add_fstrings, well_tile='{well}_{tile}')
    """
    format_strings = list(format_strings.items())
    results = {}
    for name, fmt in format_strings:
        # Extract column names from the format string
        cols = [x[1] for x in string.Formatter().parse(fmt) if x[1] is not None]
        # Convert relevant columns to a list of dictionaries
        rows = df[cols].to_dict('records')
        # Apply the format string to each row
        results[name] = [fmt.format(**row) for row in rows]
    # Add new columns to the DataFrame
    return df.assign(**results)

# NUMPY UTILS

def pile(arr):
    """
    Concatenate stacks of same dimensionality along leading dimension.

    Args:
        arr (list of np.ndarray): List of arrays to be concatenated.

    Returns:
        np.ndarray: Concatenated array with background filled with zeros.

    Notes:
        - Values are filled from top left of matrix.
        - Arrays are padded with zeros to match the largest dimensions.
    """
    shape = [max(s) for s in zip(*[x.shape for x in arr])]
    # Handle numpy limitations
    arr_out = []
    for x in arr:
        y = np.zeros(shape, x.dtype)
        slicer = tuple(slice(None, s) for s in x.shape)
        y[slicer] = x
        arr_out += [y[None, ...]]

    return np.concatenate(arr_out, axis=0)

def montage(arr, shape=None):
    """
    Tile ND arrays in last two dimensions to create a montage.

    Args:
        arr (list of np.ndarray): List of arrays to be tiled.
        shape (tuple, optional): Desired shape of the montage (rows, columns).

    Returns:
        np.ndarray: Tiled montage of input arrays.

    Notes:
        - First N-2 dimensions must match across all input arrays.
        - Tiles are expanded to max height and width and padded with zeros.
        - If shape is not provided, defaults to square, clipping last row if empty.
        - If shape contains -1, that dimension is inferred.
        - If rows or columns is 1, does not pad zeros in width or height respectively.
    """
    sz = list(zip(*[img.shape for img in arr]))
    h, w, n = max(sz[-2]), max(sz[-1]), len(arr)

    # Determine the shape of the montage
    if not shape:
        nr = nc = int(np.ceil(np.sqrt(n)))
        if (nr - 1) * nc >= n:
            nr -= 1
    elif -1 in shape:
        assert shape[0] != shape[1], 'cannot infer both rows and columns, use shape=None for square montage'
        shape = np.array(shape)
        infer, given = int(np.argwhere(shape==-1)),int(np.argwhere(shape!=-1))
        shape[infer] = int(np.ceil(n/shape[given]))
        if (shape[infer]-1)*shape[given] >= n:
            shape[infer] -= 1
        nr, nc = shape
    else:
        nr, nc = shape

    # Handle special case where one dimension is 1
    if 1 in (nr,nc):
        assert nr != nc, 'no need to montage a single image'
        shape = np.array((nr,nc))
        single_axis,other_axis = int(np.argwhere(shape==1)),int(np.argwhere(shape!=1))
        arr_padded = []
        for img in arr:
            sub_size = (h,img.shape[-2])[single_axis], (w,img.shape[-1])[other_axis]
            sub = np.zeros(img.shape[:-2] + (sub_size[0],) + (sub_size[1],), dtype=arr[0].dtype)
            s = [[None] for _ in img.shape]
            s[-2] = (0, img.shape[-2])
            s[-1] = (0, img.shape[-1])
            sub[tuple(slice(*x) for x in s)] = img
            arr_padded.append(sub)
        M = np.concatenate(arr_padded,axis=(-2+other_axis))
    else:
        M = np.zeros(arr[0].shape[:-2] + (nr * h, nc * w), dtype=arr[0].dtype)
        for (r, c), img in zip(product(range(nr), range(nc)), arr):
            s = [[None] for _ in img.shape]
            s[-2] = (r * h, r * h + img.shape[-2])
            s[-1] = (c * w, c * w + img.shape[-1])
            M[tuple(slice(*x) for x in s)] = img

    return M

def make_tiles(arr, m, n, pad=None):
    """
    Divide a stack of images into tiles.

    Args:
        arr (np.ndarray): Input array of images.
        m (int or float): Tile height or fraction of input height.
        n (int or float): Tile width or fraction of input width.
        pad (scalar, optional): Value to use for padding. If None, tiles may not be equally sized.

    Returns:
        list: List of tiled arrays.

    Notes:
        - If m or n is between 0 and 1, it specifies a fraction of the input size.
        - If pad is specified, it's used to fill in edges.
    """
    assert arr.ndim > 1
    h, w = arr.shape[-2:]
    # Convert to number of tiles
    m_ = h / m if m >= 1 else int(np.round(1 / m))
    n_ = w / n if n >= 1 else int(np.round(1 / n))

    if pad is not None:
        pad_width = (arr.ndim - 2) * ((0, 0),) + ((0, -h % m), (0, -w % n))
        arr = np.pad(arr, pad_width, 'constant', constant_values=pad)

    h_ = int(int(h / m) * m)
    w_ = int(int(w / n) * n)

    tiled = []
    for x in np.array_split(arr[:h_, :w_], m_, axis=-2):
        for y in np.array_split(x, n_, axis=-1):
            tiled.append(y)
    
    return tiled

def trim(arr, return_slice=False):
    """
    Remove i,j area that overlaps a zero value in any leading dimension.

    Args:
        arr (np.ndarray): Input array to be trimmed.
        return_slice (bool, optional): If True, return the slice object instead of the trimmed array.

    Returns:
        np.ndarray or tuple: Trimmed array or slice object if return_slice is True.

    Notes:
        - Trims stitched and piled images.
        - Removes areas where any leading dimension has a zero value.
    """
    def coords_to_slice(i_0, i_1, j_0, j_1):
        return slice(i_0, i_1), slice(j_0, j_1)

    leading_dims = tuple(range(arr.ndim)[:-2])
    mask = (arr == 0).any(axis=leading_dims)
    coords = inscribe(mask)
    sl = (Ellipsis,) + coords_to_slice(*coords)
    if return_slice:
        return sl
    return arr[sl]


@decorator.decorator
def applyIJ(f, arr, *args, **kwargs):   
    """
    Decorator to apply a function that expects 2D input to the trailing two dimensions of an array.

    Parameters:
        f (function): The function to be decorated.
        arr (numpy.ndarray): The input array to apply the function to.
        *args: Additional positional arguments to be passed to the function.
        **kwargs: Additional keyword arguments to be passed to the function.

    Returns:
        numpy.ndarray: Output array after applying the function.
    """
    # Get the height and width of the trailing two dimensions of the input array
    h, w = arr.shape[-2:]
    
    # Reshape the input array to a 3D array with shape (-1, h, w), where -1 indicates the product of all other dimensions
    reshaped = arr.reshape((-1, h, w))

    # Apply the function f to each frame in the reshaped array, along with additional arguments and keyword arguments
    # Note: kwargs are not actually getting passed in directly; this may need adjustment
    arr_ = [f(frame, *args, **kwargs) for frame in reshaped]

    # Determine the output shape based on the input array shape and the shape of the output from the function f
    output_shape = arr.shape[:-2] + arr_[0].shape
    
    # Reshape the resulting list of arrays to the determined output shape
    return np.array(arr_).reshape(output_shape)

def applyIJ_parallel(f, arr, n_jobs=-2, backend='threading', tqdm=False, *args, **kwargs):
    """
    Decorator to apply a function that expects 2D input to the trailing two dimensions of an array,
    parallelizing computation across 2D frames.

    Parameters:
        f (function): The function to be decorated and applied in parallel.
        arr (numpy.ndarray): The input array to apply the function to.
        n_jobs (int): The number of jobs to run in parallel. Default is -2.
        backend (str): The parallelization backend to use. Default is 'threading'.
        tqdm (bool): Whether to use tqdm for progress tracking. Default is False.
        *args: Additional positional arguments to be passed to the function.
        **kwargs: Additional keyword arguments to be passed to the function.

    Returns:
        numpy.ndarray: Output array after applying the function in parallel.
    """
    from joblib import Parallel, delayed

    h, w = arr.shape[-2:]
    reshaped = arr.reshape((-1, h, w))

    if tqdm:
        from tqdm import tqdm_notebook as tqdn
        work = tqdn(reshaped,'frame')
    else:
        work = reshaped

    arr_ = Parallel(n_jobs=n_jobs,backend=backend)(delayed(f)(frame, *args, **kwargs) for frame in work)

    output_shape = arr.shape[:-2] + arr_[0].shape
    return np.array(arr_).reshape(output_shape)

def inscribe(mask):
    """
    Guess the largest axis-aligned rectangle inside a binary mask.

    Args:
        mask (np.ndarray): 2D binary mask where 0 indicates background.

    Returns:
        list: Coordinates of the largest inscribed rectangle [i_0, i_1, j_0, j_1].

    Notes:
        - Rectangle must exclude zero values.
        - Assumes zeros are at the edges and there are no holes.
        - Iteratively shrinks the rectangle's most problematic edge.
    """
    h, w = mask.shape
    i_0, i_1 = 0, h - 1
    j_0, j_1 = 0, w - 1
    
    def edge_costs(i_0, i_1, j_0, j_1):
        """Calculate the cost (mean value) of each edge of the rectangle."""
        a = mask[i_0, j_0:j_1 + 1].mean() # top
        b = mask[i_1, j_0:j_1 + 1].mean() # bottom
        c = mask[i_0:i_1 + 1, j_0].mean() # left
        d = mask[i_0:i_1 + 1, j_1].mean() # right  
        return a, b, c, d
    
    def area(i_0, i_1, j_0, j_1):
        """Calculate the area of the rectangle."""
        return (i_1 - i_0) * (j_1 - j_0)
    
    coords = [i_0, i_1, j_0, j_1]
    while area(*coords) > 0:
        costs = edge_costs(*coords)
        if sum(costs) == 0:
            return coords
        worst = costs.index(max(costs))
        coords[worst] += 1 if worst in (0, 2) else -1
    return coords

def subimage(stack, bbox, pad=0):
    """
    Extract a rectangular region from a stack of images with optional padding.

    Args:
        stack (np.ndarray): Input stack of images [...xYxX].
        bbox (np.ndarray or list): Bounding box coordinates (min_row, min_col, max_row, max_col).
        pad (int, optional): Padding width. Defaults to 0.

    Returns:
        np.ndarray: Extracted subimage.

    Notes:
        - If boundary lies outside stack, raises error.
        - If padded rectangle extends outside stack, fills with zeros.
    """
    i0, j0, i1, j1 = bbox + np.array([-pad, -pad, pad, pad])

    sub = np.zeros(stack.shape[:-2]+(i1-i0, j1-j0), dtype=stack.dtype)

    i0_, j0_ = max(i0, 0), max(j0, 0)
    i1_, j1_ = min(i1, stack.shape[-2]), min(j1, stack.shape[-1])
    s = (Ellipsis, 
         slice(i0_-i0, (i0_-i0) + i1_-i0_),
         slice(j0_-j0, (j0_-j0) + j1_-j0_))

    sub[s] = stack[..., i0_:i1_, j0_:j1_]
    return sub

def offset(stack, offsets):
    """
    Apply offsets to a stack of images, filling new areas with zeros.

    Args:
        stack (np.ndarray): Input stack of images.
        offsets (list or np.ndarray): Offsets to apply for each dimension.

    Returns:
        np.ndarray: Offset stack of images.

    Notes:
        - Only applies integer offsets.
        - If len(offsets) == 2 and stack.ndim > 2, applies offsets to last two dimensions.
    """
    if len(offsets) != stack.ndim:
        if len(offsets) == 2 and stack.ndim > 2:
            offsets = [0] * (stack.ndim - 2) + list(offsets)
        else:
            raise IndexError("number of offsets must equal stack dimensions, or 2 (trailing dimensions)")

    offsets = np.array(offsets).astype(int)

    n = stack.ndim
    ns = (slice(None),)
    for d, offset in enumerate(offsets):
        stack = np.roll(stack, offset, axis=d)
        if offset < 0:
            index = ns * d + (slice(offset, None),) + ns * (n - d - 1)
            stack[index] = 0
        if offset > 0:
            index = ns * d + (slice(None, offset),) + ns * (n - d - 1)
            stack[index] = 0

    return stack    

def join_stacks(*args):
    """
    Join multiple array stacks along specified dimensions.

    Parameters:
        *args: Variable number of arrays or tuples of (array, code).
               If a tuple, the code specifies how to join the array:
               'a' for append, 'r' for repeat, '.' for normal axis.

    Returns:
        numpy.ndarray: Joined array with dimensions determined by input arrays and codes.
    """
    def with_default(arg):
        """Convert single array argument to (array, code) tuple with empty code."""
        try:
            arr, code = arg
            return arr, code
        except ValueError:
            return arg, ''

    def expand_dims(arr, n):
        """Expand array dimensions to match the target number of dimensions."""
        if arr.ndim < n:
            return expand_dims(arr[None], n)
        return arr

    def expand_code(arr, code):
        """Extend code with dots to match array dimensions."""
        return code + '.' * (arr.ndim - len(code))

    def validate_code(arr, code):
        """Check if the code is valid for the given array."""
        if code.count('a') > 1:
            raise ValueError('cannot append same array along multiple dimensions')
        if len(code) > arr.ndim:
            raise ValueError('length of code greater than number of dimensions')

    def mark_all_appends(codes):
        """Ensure consistency in append operations across all arrays."""
        arr = []
        for pos in zip(*codes):
            if 'a' in pos:
                if 'r' in pos:
                    raise ValueError('cannot repeat and append along the same axis')
                pos = 'a' * len(pos)
            arr += [pos]
        return [''.join(code) for code in zip(*arr)]

    def special_case_no_ops(args):
        if all([c == '.' for _, code in args for c in code]):
            return [(arr[None], 'a' + code) for arr, code in args]
        return args
    
    # insert default code (only dots)
    args = [with_default(arg) for arg in args]
    # expand the dimensions of the input arrays
    output_ndim = max(arr.ndim for arr, _ in args)
    args = [(expand_dims(arr, output_ndim), code) for arr, code in args]
    # add trailing dots to codes
    args = [(arr, expand_code(arr, code)) for arr, code in args]
    # if no codes are provided, interpret as appending along a new dimension
    args = special_case_no_ops(args)
    # recalculate due to special case
    output_ndim = max(arr.ndim for arr, _ in args)
    
    [validate_code(arr, code) for arr, code in args]
    # if any array is appended along an axis, every array must be
    # input codes are converted from dot to append for those axes
    codes = mark_all_appends([code for _, code in args])
    args = [(arr, code) for (arr, _), code in zip(args, codes)]

    # calculate shape for output array
    # uses numpy addition rule to determine output dtype
    output_dtype = sum([arr.flat[:1] for arr, _ in args]).dtype
    output_shape = [0] * output_ndim
    for arr, code in args:
        for i, c in enumerate(code):
            s = arr.shape[i]
            if c == '.':
                if output_shape[i] == 0 or output_shape[i] == s:
                    output_shape[i] = s
                else:
                    error = 'inconsistent shapes {0}, {1} at axis {2}'
                    raise ValueError(error.format(output_shape[i], s, i))

    for arg, code in args:
        for i, c in enumerate(code):
            s = arg.shape[i]
            if c == 'a':
                output_shape[i] += s
    
    output = np.zeros(output_shape, dtype=output_dtype)
    
    # assign from input arrays to output 
    # (values automatically coerced to most general numeric type)
    slices_so_far = [0] * output_ndim
    for arr, code in args:
        slices = []
        for i, c in enumerate(code):
            if c in 'r.':
                slices += [slice(None)]
            if c == 'a':
                s = slices_so_far[i]
                slices += [slice(s, s + arr.shape[i])]
                slices_so_far[i] += arr.shape[i]

        output[tuple(slices)] = arr
        
    return output

def max_project_zstack(stack, slices=5):
    """
    Condense z-stack into a single slice using maximum projection through all slices for each channel.

    Parameters:
        stack (numpy.ndarray): Input z-stack array.
        slices (int or list): Number of slices for each channel. If int, same for all channels.

    Returns:
        numpy.ndarray: Maximum projected array with dimensions (channels, height, width).
    """
    if isinstance(slices,list):
        channels = len(slices)

        maxed = []
        end_ch_slice = 0
        for ch in range(len(slices)):
            end_ch_slice += slices[ch]
            ch_slices = stack[(end_ch_slice-slices[ch]):(end_ch_slice)]
            ch_maxed = np.amax(ch_slices,axis=0)
            maxed.append(ch_maxed)

    else:
        channels = int(stack.shape[-3]/slices)
        assert len(stack) == int(slices)*channels, 'Input data must have leading dimension length slices*channels'

        maxed = []
        for ch in range(channels):
            ch_slices = stack[(ch*slices):((ch+1)*slices)]
            ch_maxed = np.amax(ch_slices,axis=0)
            maxed.append(ch_maxed)

    maxed = np.array(maxed)

    return maxed

# SCIKIT-IMAGE

def regionprops(labeled, intensity_image):
    """
    Supplement skimage.measure.regionprops with additional field `intensity_image_full` 
    containing multi-dimensional intensity image.

    Args:
        labeled (np.ndarray): Labeled segmentation mask defining objects.
        intensity_image (np.ndarray): Intensity image.

    Returns:
        list: List of region properties objects.
    """
    import skimage.measure

    # If intensity image has more than 2 dimensions, consider only the first channel
    if intensity_image.ndim == 2:
        base_image = intensity_image
    else:
        base_image = intensity_image[..., 0, :, :]

    # Compute region properties using skimage.measure.regionprops
    regions = skimage.measure.regionprops(labeled, intensity_image=base_image)

    # Iterate over regions and add the 'intensity_image_full' attribute
    for region in regions:
        b = region.bbox  # Get bounding box coordinates
        # Extract the corresponding sub-image from the intensity image and assign it to the 'intensity_image_full' attribute
        region.intensity_image_full = intensity_image[..., b[0]:b[2], b[1]:b[3]]

    return regions


def regionprops_multichannel(labeled, intensity_image):
    """
    Format intensity image axes for compatibility with updated skimage.measure.regionprops
    that allows multichannel images. Some operations are faster than `ops.utils.regionprops`,
    others are slower.

    Args:
        labeled (np.ndarray): Labeled segmentation mask defining objects.
        intensity_image (np.ndarray): Multichannel intensity image.

    Returns:
        list: List of region properties objects.
    """
    import skimage.measure

    # If intensity image has only 2 dimensions, consider it as a single-channel image
    if intensity_image.ndim == 2:
        base_image = intensity_image
    else:
        # Move the channel axis to the last position for compatibility with skimage.measure.regionprops
        base_image = np.moveaxis(intensity_image, range(intensity_image.ndim-2), range(-1, -(intensity_image.ndim-1), -1))
        
    # Compute region properties using skimage.measure.regionprops
    regions = skimage.measure.regionprops(labeled, intensity_image=base_image)

    return regions


def match_size(image, target, order=None):
    """
    Resize the input image to match the size of the target image without changing the data range or type.

    Args:
        image (np.ndarray): Input image to be resized.
        target (np.ndarray): Target image whose size the input image will be matched to.
        order (int, optional): Order of interpolation. Default is None.

    Returns:
        np.ndarray: Resized image with the same data range and type as the input image.
    """
    from skimage.transform import resize

    # Resize the input image to match the size of the target image
    # Preserve the original data range and type
    return (resize(image, target.shape, preserve_range=True, order=order)
            .astype(image.dtype))
