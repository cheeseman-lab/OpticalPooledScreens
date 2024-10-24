"""
Triangle Hash-Based Image Alignment

This module provides functions for aligning images based on triangular hashing of feature points
for aligning microscopy images from different acquisition modalities (relating to 3 -- merge). 
It includes functions for:

1. Triangulation: Creating and hashing Delaunay triangulations of feature points.
2. Matching: Finding corresponding triangles between two sets of feature points.
3. Alignment: Computing and refining transformation matrices between image pairs.
4. Evaluation: Assessing the quality of matches and alignments.
5. Visualization: Plotting alignments for visual inspection.

"""

import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.linear_model import RANSACRegressor, LinearRegression
import warnings

from . import utils

def find_triangles(df):
    """
    Turns a table of i,j coordinates (typically of nuclear centroids)
    into a table containing a hashed Delaunay triangulation of the 
    input points. Process each tile/site independently. The output for all
    tiles/sites within a single well is concatenated and used as input to 
    `multistep_alignment`.

    Parameters
    ----------
    df : pandas DataFrame
        Table of points with columns `i` and `j`.

    Returns
    -------
    df_dt : pandas DataFrame
        Table containing a hashed Delaunay triangulation, one line
        per simplex (triangle).

    """
    # Extract the coordinates from the dataframe and compute the Delaunay triangulation
    v, c = get_vectors(df[['i', 'j']].values)

    # Create a dataframe from the vectors and rename the columns with a prefix 'V_'
    df_vectors = pd.DataFrame(v).rename(columns='V_{0}'.format)
    
    # Create a dataframe from the coordinates and rename the columns with a prefix 'c_'
    df_coords = pd.DataFrame(c).rename(columns='c_{0}'.format)
    
    # Concatenate the two dataframes along the columns
    df_combined = pd.concat([df_vectors, df_coords], axis=1)
    
    # Assign a new column 'magnitude' which is the Euclidean distance (magnitude) of each vector
    df_result = df_combined.assign(magnitude=lambda x: x.eval('(V_0**2 + V_1**2)**0.5'))
    
    return df_result


def nine_edge_hash(dt, i):
    """
    For triangle `i` in Delaunay triangulation `dt`, extract the vector 
    displacements of the 9 edges containing at least one vertex in the 
    triangle.

    Raises an error if triangle `i` lies on the outer boundary of the triangulation.

    Example:
    dt = Delaunay(X_0)
    i = 0
    segments, vector = nine_edge_hash(dt, i)
    plot_nine_edges(X_0, segments)

    Parameters
    ----------
    dt : scipy.spatial.Delaunay
        Delaunay triangulation object containing points and simplices.
    i : int
        Index of the triangle in the Delaunay triangulation.

    Returns
    -------
    segments : list of tuples
        List of vertex pairs representing the 9 edges.
    vector : numpy.ndarray
        Array containing vector displacements for the 9 edges.
    """
    # Indices of inner three vertices in CCW order
    a, b, c = dt.simplices[i]

    # Reorder vertices so that the edge 'ab' is the longest
    X = dt.points
    start = np.argmax((np.diff(X[[a, b, c, a]], axis=0)**2).sum(axis=1)**0.5)
    if start == 0:
        order = [0, 1, 2]
    elif start == 1:
        order = [1, 2, 0]
    elif start == 2:
        order = [2, 0, 1]
    a, b, c = np.array([a, b, c])[order]

    # Get indices of outer three vertices connected to the inner vertices
    a_ix, b_ix, c_ix = dt.neighbors[i]
    inner = {a, b, c}
    outer = lambda xs: [x for x in xs if x not in inner][0]

    try:
        bc = outer(dt.simplices[dt.neighbors[i, order[0]]])
        ac = outer(dt.simplices[dt.neighbors[i, order[1]]])
        ab = outer(dt.simplices[dt.neighbors[i, order[2]]])
    except IndexError:
        return None

    if any(x == -1 for x in (bc, ac, ab)):
        error = 'triangle on outer boundary, neighbors are: {0} {1} {2}'
        raise ValueError(error.format(bc, ac, ab))

    # Define the 9 edges
    segments = [
        (a, b),
        (b, c),
        (c, a),
        (a, ab),
        (b, ab),
        (b, bc),
        (c, bc),
        (c, ac),
        (a, ac),
    ]

    # Extract the vector displacements for the 9 edges
    i_coords = X[segments, 0]
    j_coords = X[segments, 1]
    vector = np.hstack([np.diff(i_coords, axis=1), np.diff(j_coords, axis=1)])
    
    return segments, vector

def plot_nine_edges(X, segments):
    """
    Plot the 9 edges of a triangle in a Delaunay triangulation.

    Parameters
    ----------
    X : numpy.ndarray
        Array of points used for the Delaunay triangulation.
    segments : list of tuples
        List of vertex pairs representing the 9 edges.
    
    Returns
    -------
    ax : matplotlib.axes.Axes
        The plot axes.
    """
    fig, ax = plt.subplots()
    
    [(a, b),
     (b, c),
     (c, a),
     (a, ab),
     (b, ab),
     (b, bc),
     (c, bc),
     (c, ac),
     (a, ac)] = segments
    
    # Plot each edge
    for i0, i1 in segments:
        ax.plot(X[[i0, i1], 0], X[[i0, i1], 1])

    # Label the vertices
    labels = {'a': a, 'b': b, 'c': c, 'ab': ab, 'bc': bc, 'ac': ac}
    for label, vertex in labels.items():
        i, j = X[vertex]
        ax.text(i, j, label)

    # Scatter plot of the points
    ax.scatter(X[:, 0], X[:, 1])

    # Set plot limits
    s = X[np.array(segments).flatten()]
    lim0 = s.min(axis=0) - 100
    lim1 = s.max(axis=0) + 100

    ax.set_xlim([lim0[0], lim1[0]])
    ax.set_ylim([lim0[1], lim1[1]])
    
    return ax


def get_vectors(X):
    """
    Get the nine edge vectors and centers for all the faces in the 
    Delaunay triangulation of point array `X`.

    Parameters
    ----------
    X : numpy.ndarray
        Array of points to be triangulated.

    Returns
    -------
    vectors : numpy.ndarray
        Array of shape (n_faces, 18) containing the vector displacements for the nine edges of each triangle.
    centers : numpy.ndarray
        Array of shape (n_faces, 2) containing the center points of each triangle.
    """
    dt = Delaunay(X)  # Create Delaunay triangulation of the points
    vectors, centers = [], []  # Initialize lists to store vectors and centers
    
    for i in range(dt.simplices.shape[0]):
        # Skip triangles with an edge on the outer boundary
        if (dt.neighbors[i] == -1).any():
            continue
        
        result = nine_edge_hash(dt, i)  # Get the nine edge vectors for the current triangle
        # Some rare event where hashing fails
        if result is None:
            continue
        
        _, v = result  # Unpack the result to get the vectors
        c = X[dt.simplices[i], :].mean(axis=0)  # Calculate the center of the triangle
        vectors.append(v)  # Append the vectors to the list
        centers.append(c)  # Append the center to the list

    # Convert lists to numpy arrays and reshape vectors to (n_faces, 18)
    return np.array(vectors).reshape(-1, 18), np.array(centers)


def nearest_neighbors(V_0, V_1):
    """
    Compute the nearest neighbors between two sets of vectors V_0 and V_1.

    Parameters
    ----------
    V_0 : numpy.ndarray
        First set of vectors.
    V_1 : numpy.ndarray
        Second set of vectors.

    Returns
    -------
    ix_0 : numpy.ndarray
        Indices of the nearest neighbors in V_0.
    ix_1 : numpy.ndarray
        Indices of the nearest neighbors in V_1.
    distances : numpy.ndarray
        Distances between the nearest neighbors.
    """
    Y = cdist(V_0, V_1, metric='sqeuclidean')  # Compute squared Euclidean distances
    distances = np.sqrt(Y.min(axis=1))  # Compute the smallest distances and take the square root
    ix_0 = np.arange(V_0.shape[0])  # Indices of V_0
    ix_1 = Y.argmin(axis=1)  # Indices of nearest neighbors in V_1
    return ix_0, ix_1, distances  # Return indices and distances


def get_vc(df, normalize=True):
    """
    Extract vectors and centers from the DataFrame and optionally normalize the vectors.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing vectors and centers.
    normalize : bool, optional
        Whether to normalize the vectors (default is True).

    Returns
    -------
    V : numpy.ndarray
        Array of vectors.
    c : numpy.ndarray
        Array of centers.
    """
    V, c = (df.filter(like='V').values, df.filter(like='c').values)  # Extract vectors and centers
    if normalize:
        V = V / df['magnitude'].values[:, None]  # Normalize the vectors by their magnitudes
    return V, c  # Return vectors and centers


def evaluate_match(df_0, df_1, threshold_triangle=0.3, threshold_point=2):
    """
    Evaluate the match between two sets of vectors and centers.

    Parameters
    ----------
    df_0 : pandas.DataFrame
        DataFrame containing the first set of vectors and centers.
    df_1 : pandas.DataFrame
        DataFrame containing the second set of vectors and centers.
    threshold_triangle : float, optional
        Threshold for matching triangles (default is 0.3).
    threshold_point : float, optional
        Threshold for matching points (default is 2).

    Returns
    -------
    rotation : numpy.ndarray
        Rotation matrix of the transformation.
    translation : numpy.ndarray
        Translation vector of the transformation.
    score : float
        Score of the transformation based on the matching points.
    """
    V_0, c_0 = get_vc(df_0)  # Extract vectors and centers from the first DataFrame
    V_1, c_1 = get_vc(df_1)  # Extract vectors and centers from the second DataFrame

    i0, i1, distances = nearest_neighbors(V_0, V_1)  # Find nearest neighbors between the vectors

    # Filter triangles based on distance threshold
    filt = distances < threshold_triangle
    X, Y = c_0[i0[filt]], c_1[i1[filt]]  # Get the matching centers

    # Minimum number of matching triangles required to proceed
    if sum(filt) < 5:
        return None, None, -1

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        # Use matching triangles to define transformation
        model = RANSACRegressor()
        model.fit(X, Y)  # Fit the RANSAC model to the matching centers

    rotation = model.estimator_.coef_  # Extract rotation matrix
    translation = model.estimator_.intercept_  # Extract translation vector

    # Score transformation based on the triangle centers
    distances = cdist(model.predict(c_0), c_1, metric='sqeuclidean')
    threshold_region = 50  # Threshold for the region to consider
    filt = np.sqrt(distances.min(axis=0)) < threshold_region
    score = (np.sqrt(distances.min(axis=0))[filt] < threshold_point).mean()  # Calculate score

    return rotation, translation, score  # Return rotation, translation, and score


def build_linear_model(rotation, translation):
    """
    Build a linear regression model using the provided rotation matrix and translation vector.

    Parameters
    ----------
    rotation : numpy.ndarray
        Rotation matrix for the model.
    translation : numpy.ndarray
        Translation vector for the model.

    Returns
    -------
    m : sklearn.linear_model.LinearRegression
        Linear regression model with the specified rotation and translation.
    """
    m = LinearRegression()
    m.coef_ = rotation  # Set the rotation matrix as the model's coefficients
    m.intercept_ = translation  # Set the translation vector as the model's intercept
    return m  # Return the linear regression model


def prioritize(df_info_0, df_info_1, matches):
    """
    Produce an Nx2 array of tile (site) identifiers predicted to match within a search radius based on existing matches.

    Parameters
    ----------
    df_info_0 : pandas.DataFrame
        DataFrame containing tile (site) information for the first set.
    df_info_1 : pandas.DataFrame
        DataFrame containing tile (site) information for the second set.
    matches : numpy.ndarray
        Nx2 array of tile (site) identifiers representing existing matches.

    Returns
    -------
    candidates : list of tuples
        List of predicted matching tile (site) identifiers.
    """
    a = df_info_0.loc[matches[:, 0]].values  # Get coordinates of matching tiles from the first set
    b = df_info_1.loc[matches[:, 1]].values  # Get coordinates of matching tiles from the second set

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        model = RANSACRegressor()
        model.fit(a, b)  # Fit the RANSAC model to the matching coordinates

    # Predict coordinates for the first set and calculate distances to the second set
    predicted = model.predict(df_info_0.values)
    distances = cdist(predicted, df_info_1.values, metric='sqeuclidean')
    ix = np.argsort(distances.flatten())  # Sort distances to find the closest matches
    ix_0, ix_1 = np.unravel_index(ix, distances.shape)  # Get indices of the closest matches

    candidates = list(zip(df_info_0.index[ix_0], df_info_1.index[ix_1]))  # Create list of candidate matches

    return remove_overlap(candidates, matches)  # Remove overlapping matches


def remove_overlap(xs, ys):
    """
    Remove overlapping pairs from a list of candidates based on an existing set of matches.

    Parameters
    ----------
    xs : list of tuples
        List of candidate pairs.
    ys : list of tuples
        List of existing matches.

    Returns
    -------
    result : list of tuples
        List of candidate pairs with overlaps removed.
    """
    ys = set(map(tuple, ys))  # Convert existing matches to a set of tuples for fast lookup
    return [tuple(x) for x in xs if tuple(x) not in ys]  # Return candidates that are not in existing matches


def brute_force_pairs(df_0, df_1, n_jobs=-2, tqdn=True):
    """
    Evaluate all pairs of tiles (sites) between two DataFrames to find the best matches.

    Parameters
    ----------
    df_0 : pandas.DataFrame
        DataFrame containing the first set of tiles (sites).
    df_1 : pandas.DataFrame
        DataFrame containing the second set of tiles (sites).
    n_jobs : int, optional
        Number of jobs to run in parallel (default is -2, which uses all but one core).
    tqdn : bool, optional
        Whether to use tqdm for progress indication (default is True).

    Returns
    -------
    result : pandas.DataFrame
        DataFrame containing the results of the matching process, sorted by score.
    """
    if tqdn:
        from tqdm.auto import tqdm
        work = tqdm(df_1.groupby('site'), desc='site')  # Show progress bar if tqdn is True
    else:
        work = df_1.groupby('site')

    arr = []
    for site, df_s in work:
        def work_on(df_t):
            rotation, translation, score = evaluate_match(df_t, df_s)  # Evaluate match for each tile pair
            determinant = None if rotation is None else np.linalg.det(rotation)  # Calculate determinant if rotation is not None
            result = pd.Series({'rotation': rotation, 
                                'translation': translation, 
                                'score': score, 
                                'determinant': determinant})  # Create a result series
            return result

        (df_0
         .pipe(utils.gb_apply_parallel, 'tile', work_on, n_jobs=n_jobs)  # Apply work_on function in parallel
         .assign(site=site)  # Assign site to the results
         .pipe(arr.append)  # Append results to the list
        )

    return (pd.concat(arr).reset_index()  # Concatenate all results and reset index
            .sort_values('score', ascending=False))  # Sort results by score in descending order

def parallel_process(func, args_list, n_jobs, tqdn=True):
    """
    Parallelize the execution of a function over a list of arguments using Joblib.

    Parameters
    ----------
    func : callable
        The function to be executed in parallel.
    args_list : list of tuples
        List of arguments to be passed to the function. Each element in the list
        is a tuple of arguments for a single function call.
    n_jobs : int
        The number of jobs (processes) to run in parallel. Use -1 to run as many jobs as there are
        CPUs.
    tqdn : bool, optional
        Whether to display a progress bar using tqdm (default is True).

    Returns
    -------
    results : list
        List of results from the function calls, in the same order as the input arguments.
    """
    from joblib import Parallel, delayed  # Import Joblib's parallel processing utilities

    # Conditionally import tqdm for progress indication
    if tqdn:
        from tqdm.auto import tqdm
        work = tqdm(args_list, desc='work')  # Wrap the arguments list with tqdm for progress bar
    else:
        work = args_list  # Use the arguments list as-is if tqdm is not enabled

    # Execute the function in parallel over the argument list
    return Parallel(n_jobs=n_jobs)(delayed(func)(*w) for w in work)


def merge_sbs_phenotype(df_0_, df_1_, model, threshold=2):
    """
    Fine alignment of one (tile, site) match found using `multistep_alignment`.

    Parameters
    ----------
    df_0_ : pandas DataFrame
        Table of coordinates to align (e.g., nuclei centroids) 
        for one tile of dataset 0. Expects `i` and `j` columns.
    df_1_ : pandas DataFrame
        Table of coordinates to align (e.g., nuclei centroids) 
        for one site of dataset 1 that was identified as a match
        to the tile in df_0_ using `multistep_alignment`. Expects 
        `i` and `j` columns.
    model : sklearn.linear_model.LinearRegression 
        Linear alignment model is suggested to be passed in, functions
        between tile of df_0_ and site of df_1_. Produced using 
        `build_linear_model` with the rotation and translation matrix 
        determined in `multistep_alignment`.
    threshold : float, default 2
        Maximum euclidean distance allowed between matching points.

    Returns
    -------
    df_merge : pandas DataFrame
        Table of merged identities of cell labels from df_0_ and 
        df_1_.
    """
    
    # Extract coordinates from the DataFrames
    X = df_0_[['i', 'j']].values  # Coordinates from dataset 0
    Y = df_1_[['i', 'j']].values  # Coordinates from dataset 1
    
    # Predict coordinates for dataset 0 using the alignment model
    Y_pred = model.predict(X)

    # Calculate squared Euclidean distances between predicted coordinates and dataset 1 coordinates
    distances = cdist(Y, Y_pred, metric='sqeuclidean')
    
    # Find the index of the nearest neighbor in Y_pred for each point in Y
    ix = distances.argmin(axis=1)
    
    # Filter matches based on the threshold distance
    filt = np.sqrt(distances.min(axis=1)) < threshold
    
    # Define new column names for merging
    columns_0 = {'tile': 'tile', 'cell': 'cell_0',
                 'i': 'i_0', 'j': 'j_0'}
    columns_1 = {'site': 'site', 'cell': 'cell_1',
                 'i': 'i_1', 'j': 'j_1'}
    
    # Final columns for the merged DataFrame
    cols_final = ['well', 'tile', 'cell_0', 'i_0', 'j_0', 
                  'site', 'cell_1', 'i_1', 'j_1', 'distance']

    # Prepare the target DataFrame with matched coordinates from dataset 0
    target = df_0_.iloc[ix[filt]].reset_index(drop=True).rename(columns=columns_0)
    
    # Merge DataFrames and calculate distances
    return (df_1_[filt].reset_index(drop=True)  # Filtered rows from dataset 1
            [list(columns_1.keys())]  # Select columns for dataset 1
            .rename(columns=columns_1)  # Rename columns for dataset 1
            .pipe(lambda x: pd.concat([target, x], axis=1))  # Concatenate with target DataFrame
            .assign(distance=np.sqrt(distances.min(axis=1))[filt])  # Assign distance column
            [cols_final]  # Select final columns
           )


def initial_alignment(df_0, df_1, initial_sites=8):
    """
    Finds tiles of two different acquisitions with matching Delaunay 
    triangulations within the same well. Cells must not have moved significantly
    between acquisitions and segmentations approximately equivalent.

    Parameters
    ----------
    df_0 : pandas DataFrame
        Hashed Delaunay triangulation for all tiles of dataset 0. Produced by 
        concatenating outputs of `find_triangles` from individual tiles of a 
        single well. Expects a `tile` column.

    df_1 : pandas DataFrame
        Hashed Delaunay triangulation for all sites of dataset 1. Produced by 
        concatenating outputs of `find_triangles` from individual sites of a 
        single well. Expects a `site` column.

    initial_sites : int or list of 2-tuples, default 8
        If int, the number of sites to sample from df_1 for initial brute force 
        matching of tiles to build an initial global alignment model. Brute force 
        can be inefficient and inaccurate. If a list of 2-tuples, these are known 
        matches of (tile,site) to initially evaluate and start building a global 
        alignment model. 5 or more intial pairs of known matching sites should be 
        sufficient.

    Returns
    -------
    df_align : pandas DataFrame
        Table of possible (tile,site) matches with corresponding rotation and translation 
        transformations. All tested matches are included here, should query based on `score` 
        and `determinant` to keep only valid matches.
    """

    # Define a function to work on individual (tile,site) pairs
    def work_on(df_t, df_s):
        rotation, translation, score = evaluate_match(df_t, df_s)
        determinant = None if rotation is None else np.linalg.det(rotation)
        result = pd.Series({'rotation': rotation, 
                            'translation': translation, 
                            'score': score, 
                            'determinant': determinant})
        return result


    arr = []
    for tile, site in initial_sites:
        result = work_on(df_0.query('tile==@tile'), df_1.query('site==@site'))
        result.at['site'] = site
        result.at['tile'] = tile
        arr.append(result)
    df_initial = pd.DataFrame(arr)
    
    return df_initial

    

def multistep_alignment(df_0, df_1, df_info_0, df_info_1, 
                        det_range=(1.125,1.186), score=0.1,
                        initial_sites=8, batch_size=180, 
                        tqdn=True, n_jobs=None):
    """
    Finds tiles of two different acquisitions with matching Delaunay 
    triangulations within the same well. Cells must not have moved significantly
    between acquisitions and segmentations approximately equivalent.

    Parameters
    ----------
    df_0 : pandas DataFrame
        Hashed Delaunay triangulation for all tiles of dataset 0. Produced by 
        concatenating outputs of `find_triangles` from individual tiles of a 
        single well. Expects a `tile` column.

    df_1 : pandas DataFrame
        Hashed Delaunay triangulation for all sites of dataset 1. Produced by 
        concatenating outputs of `find_triangles` from individual sites of a 
        single well. Expects a `site` column.

    df_info_0 : pandas DataFrame
        Table of global coordinates for each tile acquisition to match tiles
        of `df_0`. Expects `tile` as index and two columns of coordinates.

    df_info_1 : pandas DataFrame
        Table of global coordinates for each site acquisition to match sites 
        of `df_1`. Expects `site` as index and two columns of coordinates.

    det_range : 2-tuple, default (1.125,1.186)
        Range of acceptable values for the determinant of the rotation matrix 
        when evaluating an alignment of a tile, site pair. Rotation matrix determinant
        is a measure of the scaling between sites, should be consistent within microscope
        acquisition settings. Calculate determinant for several known matches in a dataset 
        to determine (used for initial alignment)

    score : float, default 0.1
        Threshold score value to consider when filtering matches for good matches 
        vs. spurious ones that should be discarded (used for initial alignment)

    initial_sites : int or list of 2-tuples, default 8
        If int, the number of sites to sample from df_1 for initial brute force 
        matching of tiles to build an initial global alignment model. Brute force 
        can be inefficient and inaccurate. If a list of 2-tuples, these are known 
        matches of (tile,site) to initially evaluate and start building a global 
        alignment model. 5 or more intial pairs of known matching sites should be 
        sufficient.

    batch_size : int, default 180
        Number of (tile,site) matches to evaluate in a batch between updates of the global 
        alignment model.

    tqdn : boolean, default True
        Displays tqdm progress bar if True.

    n_jobs : int or None, default None
        Number of parallelized jobs to deploy using joblib.

    Returns
    -------
    df_align : pandas DataFrame
        Table of possible (tile,site) matches with corresponding rotation and translation 
        transformations. All tested matches are included here, should query based on `score` 
        and `determinant` to keep only valid matches.
    """

    # If n_jobs is not provided, set it to one less than the number of CPU cores
    if n_jobs is None:
        import multiprocessing
        n_jobs = multiprocessing.cpu_count() - 1

    # Define a function to work on individual (tile,site) pairs
    def work_on(df_t, df_s):
        rotation, translation, score = evaluate_match(df_t, df_s)
        determinant = None if rotation is None else np.linalg.det(rotation)
        result = pd.Series({'rotation': rotation, 
                            'translation': translation, 
                            'score': score, 
                            'determinant': determinant})
        return result

    # If initial_sites is provided as a list of known matches, use it directly
    if isinstance(initial_sites, list):
        arr = []
        for tile, site in initial_sites:
            result = work_on(df_0.query('tile==@tile'), df_1.query('site==@site'))
            result.at['site'] = site
            result.at['tile'] = tile
            arr.append(result)
        df_initial = pd.DataFrame(arr)
    else:
        # Otherwise, sample initial_sites number of sites randomly from df_1
        sites = (pd.Series(df_info_1.index)
                 .sample(initial_sites, replace=False, random_state=0)
                 .pipe(list))
        # Use brute force to find initial pairs of matches between tiles and sites
        df_initial = brute_force_pairs(df_0, df_1.query('site == @sites'), tqdn=tqdn, n_jobs=n_jobs)

    # Unpack det_range tuple into d0 and d1
    d0, d1 = det_range

    # Define the gate condition for filtering matches based on determinant and score
    gate = '@d0 <= determinant <= @d1 & score > @score'

    # Initialize alignments list with the initial matches
    alignments = [df_initial.query(gate)]

    # Main loop for iterating until convergence
    while True:
        # Concatenate alignments and remove duplicates
        df_align = (pd.concat(alignments, sort=True)
                    .drop_duplicates(['tile', 'site']))

        # Extract tested and matched pairs
        tested = df_align.reset_index()[['tile', 'site']].values
        matches = (df_align.query(gate).reset_index()[['tile', 'site']].values)

        # Prioritize candidate pairs based on certain criteria
        candidates = prioritize(df_info_0, df_info_1, matches)
        candidates = remove_overlap(candidates, tested)

        print('matches so far: {0} / {1}'.format(
            len(matches), df_align.shape[0]))

        # Prepare data for parallel processing
        work = []
        d_0 = dict(list(df_0.groupby('tile')))
        d_1 = dict(list(df_1.groupby('site')))
        for ix_0, ix_1 in candidates[:batch_size]:
            if ix_0 in d_0 and ix_1 in d_1:  # Only process if both keys exist
                work.append([d_0[ix_0], d_1[ix_1]])
            else:
                print(f"Skipping tile {ix_0}, site {ix_1} - not found in data")

        if not work:  # If no valid pairs found, end alignment
            print("No valid pairs to process")
            break

        # Perform parallel processing of work
        df_align_new = (pd.concat(parallel_process(work_on, work, n_jobs=n_jobs, tqdn=tqdn), axis=1).T
                        .assign(tile=[t for t, _ in candidates[:len(work)]], 
                                site=[s for _, s in candidates[:len(work)]])
                        )

        # Append new alignments to the list
        alignments += [df_align_new]

        if len(df_align_new.query(gate)) == 0:
            break
            
    return df_align

def plot_alignment_quality(df_align, det_range, score, xlim=(0, 0.1), ylim=(0, 1), figsize=(10, 6)):
    """
    Creates a scatter plot visualizing alignment quality based on determinant and score values.
    
    Parameters
    ----------
    df_align : pandas DataFrame
        DataFrame containing alignment results. Must have columns: 'determinant', 'score', 
        'tile', and 'site'.
    det_range : tuple
        (min, max) range for acceptable determinant values.
    score : float
        Minimum acceptable score value.
    xlim : tuple, optional
        (min, max) range for x-axis (determinant). Default is (0, 0.1).
    ylim : tuple, optional
        (min, max) range for y-axis (score). Default is (0, 1).
    figsize : tuple, optional
        Figure size in inches. Default is (10, 6).
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure object.
    ax : matplotlib.axes.Axes
        The created axes object.
    """
    # Create figure and axes
    fig, ax = plt.subplots(figsize=figsize)
    
    # Construct filtering condition
    gate = '{0} <= determinant <= {1} & score > {2}'.format(*det_range, score)
    
    # Plot scatter points
    scatter = ax.scatter(df_align['determinant'], 
                        df_align['score'],
                        c='blue',
                        alpha=0.6)
    
    # Add labels for each point
    for idx, row in df_align.iterrows():
        ax.annotate(f"PH:{row['tile']}\nSBS:{row['site']}", 
                    (row['determinant'], row['score']),
                    xytext=(5, 5), textcoords='offset points')
    
    # Add threshold lines
    ax.axhline(y=score, color='r', linestyle='--', 
               label=f'Score threshold = {score}')
    ax.axvline(x=det_range[0], color='g', linestyle='--', 
               label=f'Det min = {det_range[0]}')
    ax.axvline(x=det_range[1], color='g', linestyle='--', 
               label=f'Det max = {det_range[1]}')
    
    # Shade valid region
    ax.axvspan(det_range[0], det_range[1], ymin=score/ylim[1], 
               alpha=0.1, color='green', label='Valid region')
    
    # Set axis labels and title
    ax.set_xlabel('Determinant')
    ax.set_ylabel('Score')
    ax.set_title('Alignment Quality Check\nScore vs Determinant')
    
    # Set axis ranges
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    
    # Add legend in top left corner
    ax.legend(loc='upper left')
    
    # Show grid
    ax.grid(True, alpha=0.3)
    
    # Calculate and add statistics
    passing = df_align.query(gate).shape[0]
    total = df_align.shape[0]
    stats_text = f'Passing alignments: {passing}/{total}\n' \
                 f'({passing/total*100:.1f}%)'
    ax.text(0.95, 0.95, stats_text,
            transform=ax.transAxes,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    return fig, ax

def plot_merge_example(df_ph, df_sbs, alignment_vec, threshold=2):
    """
    Visualizes the merge process for a single tile-site pair.
    
    Parameters
    ----------
    df_ph : pandas DataFrame
        Phenotype data with 'i', 'j' columns
    df_sbs : pandas DataFrame
        SBS data with 'i', 'j' columns
    alignment_vec : dict
        Contains 'rotation' and 'translation' for alignment
    threshold : float, default 2
        Distance threshold for matching points
    """
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 10))
    
    # Filter for specific tile and site
    df_ph_filtered = df_ph[df_ph['tile'] == alignment_vec['tile']]
    df_sbs_filtered = df_sbs[df_sbs['tile'] == alignment_vec['site']]
    
    # Get coordinates
    X = df_ph_filtered[['i', 'j']].values
    Y = df_sbs_filtered[['i', 'j']].values
    
    # Build model and predict
    model = build_linear_model(alignment_vec['rotation'], 
                             alignment_vec['translation'])
    Y_pred = model.predict(X)
    
    # Calculate distances
    distances = cdist(Y, Y_pred, metric='sqeuclidean')
    ix = distances.argmin(axis=1)
    filt = np.sqrt(distances.min(axis=1)) < threshold
    
    # Filter out Y_pred based on filt
    Y_pred = Y_pred[ix[filt]]

    # Calculate statistics
    n_ph = len(X)
    n_sbs = len(Y)
    n_matched = len(Y_pred)
    ph_match_rate = n_matched/n_ph * 100
    sbs_match_rate = n_matched/n_sbs * 100
    
    # Plot 1: Original Scale
    ax1.scatter(X[:, 0], X[:, 1], c='blue', s=20, alpha=0.5, 
              label=f'Phenotype ({n_ph} points)')
    ax1.scatter(Y_pred[:, 0], Y_pred[:, 1], c='red', s=20, 
              alpha=0.5, label=f'Aligned SBS ({n_matched}) points)')
    ax1.scatter(Y[:, 0], Y[:, 1], c='green', s=20, alpha=0.5, 
              label=f'Original SBS ({n_sbs} points)')
    
    # Draw lines between matched points that pass threshold
    for i in range(len(Y)):
        if filt[i]:
            ax1.plot([X[ix[i], 0], Y[i, 0]], 
                   [X[ix[i], 1], Y[i, 1]], 
                   'k-', alpha=0.1)
    
    ax1.set_title(f'Original Scale View\nPH:{alignment_vec["tile"]}, SBS:{alignment_vec["site"]}')
    ax1.legend()
    
    # Plot 2: Scale PH values to SBS axis
    X_norm = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

    # Get the range and minimum of aligned SBS points (Y_pred)
    Y_pred_range = Y_pred.max(axis=0) - Y_pred.min(axis=0)
    Y_pred_min = Y_pred.min(axis=0)

    # Scale and translate phenotype points to align with SBS field
    X_scaled = (X_norm * Y_pred_range) + Y_pred_min

    ax2.scatter(Y[:, 0], Y[:, 1], c='lightgray', s=20, alpha=0.1,
            label=f'SBS Field ({n_sbs} points)')
    ax2.scatter(Y_pred[:, 0], Y_pred[:, 1], c='red', s=20,
            alpha=0.25, label=f'Aligned SBS ({n_matched} points)')
    ax2.scatter(X_scaled[:, 0], X_scaled[:, 1], c='blue', s=20,
        alpha=0.25, label=f'Phenotype ({n_ph} points)')

    ax2.set_title('Normalized Scale For PH Points Relative to SBS')
    ax2.legend()

    # Plot 3: Scale PH values to SBS axis
    X_norm = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    # Get the range and minimum of aligned SBS points (Y_pred)
    Y_pred_range = Y_pred.max(axis=0) - Y_pred.min(axis=0)
    Y_pred_min = Y_pred.min(axis=0)
    # Scale and translate phenotype points to align with SBS field
    X_scaled = (X_norm * Y_pred_range) + Y_pred_min
    # Find unmatched phenotype points
    matched_ph_ix = np.unique(ix[filt])
    unmatched_ph_mask = ~np.isin(np.arange(len(X)), matched_ph_ix)
    # Plot SBS field and aligned points
    ax3.scatter(Y[:, 0], Y[:, 1], c='lightgray', s=20, alpha=0.1,
            label=f'SBS Field ({n_sbs} points)')
    ax3.scatter(Y_pred[:, 0], Y_pred[:, 1], c='red', s=20,
            alpha=0.25, label=f'Aligned SBS ({n_matched} points)')
    # Plot matched phenotype points in blue
    ax3.scatter(X_scaled[~unmatched_ph_mask][:, 0], X_scaled[~unmatched_ph_mask][:, 1], 
            c='blue', s=20, alpha=0.25, 
            label=f'Matched Phenotype ({n_matched} points)')
    # Plot unmatched phenotype points in yellow with star marker
    ax3.scatter(X_scaled[unmatched_ph_mask][:, 0], X_scaled[unmatched_ph_mask][:, 1],
            marker='*', c='yellow', s=100, alpha=1,
            label=f'Unmatched Phenotype ({sum(unmatched_ph_mask)} points)')
    # Optionally add labels for unmatched points
    for i in np.where(unmatched_ph_mask)[0]:
        ax3.annotate(f'Cell {i}',
                    (X_scaled[i, 0], X_scaled[i, 1]),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round', fc='white', alpha=0.7))
    ax3.set_title('Normalized Scale For PH Points Relative to SBS (with unmatched points)')
    ax3.legend()
    
    plt.tight_layout()
    plt.show()