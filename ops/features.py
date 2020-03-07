import numpy as np
import ops.utils

# FUNCTIONS


def correlate_channels(r, first, second):
    """Cross-correlation between non-zero pixels. 
    Uses `first` and `second` to index channels from `r.intensity_image_full`.
    """
    A, B = r.intensity_image_full[[first, second]]

    filt = A > 0
    if filt.sum() == 0:
        return np.nan

    A = A[filt]
    B  = B[filt]
    corr = (A - A.mean()) * (B - B.mean()) / (A.std() * B.std())

    return corr.mean()


def masked(r, index):
    return r.intensity_image_full[index][r.image]

def correlate_channels_masked(r, first, second):
    """Cross-correlation between non-zero pixels. 
    Uses `first` and `second` to index channels from `r.intensity_image_full`.
    """
    A = masked(r,first)
    B = masked(r, second)

    filt = A > 0
    if filt.sum() == 0:
        return np.nan

    A = A[filt]
    B  = B[filt]
    corr = (A - A.mean()) * (B - B.mean()) / (A.std() * B.std())

    return corr.mean()


# FEATURES
# these functions expect an `skimage.measure.regionprops` region as input

intensity = {
    'mean': lambda r: r.intensity_image[r.image].mean(),
    'median': lambda r: np.median(r.intensity_image[r.image]),
    'max': lambda r: r.intensity_image[r.image].max(),
    'min': lambda r: r.intensity_image[r.image].min(),
    }

geometry = {
    'area'    : lambda r: r.area,
    'i'       : lambda r: r.centroid[0],
    'j'       : lambda r: r.centroid[1],
    'bounds'  : lambda r: r.bbox,
    # 'contour' : lambda r: ops.utils.binary_contours(r.image, fix=True, labeled=False)[0],
    'label'   : lambda r: r.label,
    # 'mask':     lambda r: ops.utils.Mask(r.image),
    'eccentricity': lambda r: r.eccentricity,
    'solidity': lambda r: r.solidity,
    'convex_area': lambda r: r.convex_area,
    'perimeter': lambda r: r.perimeter
    }

# DAPI, HA, myc
frameshift = {
    'dapi_ha_corr' : lambda r: correlate_channels(r, 0, 1),
    'dapi_myc_corr': lambda r: correlate_channels(r, 0, 2),
    'ha_median'    : lambda r: np.median(r.intensity_image_full[1]),
    'myc_median'   : lambda r: np.median(r.intensity_image_full[2]),
    'cell'         : lambda r: r.label,
    }

translocation = {
    'dapi_gfp_corr' : lambda r: correlate_channels(r, 0, 1),
    'dapi_mean'  : lambda r: masked(r, 0).mean(),
    'dapi_median': lambda r: np.median(masked(r, 0)),
    'gfp_median' : lambda r: np.median(masked(r, 1)),
    'gfp_mean'   : lambda r: masked(r, 1).mean(),
    'dapi_int'   : lambda r: masked(r, 0).sum(),
    'gfp_int'    : lambda r: masked(r, 1).sum(),
    'dapi_max'   : lambda r: masked(r, 0).max(),
    'gfp_max'    : lambda r: masked(r, 1).max(),
    }

viewRNA = {
    'cy3_median': lambda r: np.median(masked(r, 1)),
    'cy5_median': lambda r: np.median(masked(r, 2)),
    'cy5_80p'   : lambda r: np.percentile(masked(r, 2), 80),
    'cy3_int': lambda r: masked(r, 1).sum(),
    'cy5_int': lambda r: masked(r, 2).sum(),
    'cy5_mean': lambda r: masked(r, 2).sum(),
    'cy5_max': lambda r: masked(r, 2).max(),
}

all_features = [
    intensity, 
    geometry,
    translocation,
    frameshift,
    viewRNA
    ]


def validate_features():
    names = sum(map(list, all_features), [])
    assert len(names) == len(set(names))

def make_feature_dict(feature_names):
    features = {}
    [features.update(d) for d in all_features]
    return {n: features[n] for n in feature_names}

validate_features()

features_basic = make_feature_dict(('area', 'i', 'j', 'label','bounds'))

features_geom = make_feature_dict((
    'area', 'eccentricity', 'convex_area', 'perimeter'))

features_translocation_nuclear = make_feature_dict((
	'dapi_gfp_corr', 
	'eccentricity', 'solidity',
	'dapi_median', 'dapi_mean', 'dapi_int', 'dapi_max',
	'gfp_median',  'gfp_mean',  'gfp_int',  'gfp_max',
    'area'))

features_translocation_cell = make_feature_dict((	
	'dapi_gfp_corr', 
	'eccentricity', 'solidity',
	'dapi_median', 'dapi_mean', 'dapi_int', 'dapi_max',
	'gfp_median',  'gfp_mean',  'gfp_int',  'gfp_max',
    'area'))

features_frameshift = make_feature_dict((
    'dapi_ha_corr', 
    'dapi_median', 'dapi_max', 
    'ha_median'))

features_frameshift_myc = make_feature_dict((
    'dapi_ha_corr', 'dapi_myc_corr', 
    'dapi_median', 'dapi_max', 
    'ha_median', 'myc_median'))

features_translocation_nuclear_simple = make_feature_dict((
	'dapi_gfp_corr', 
	'dapi_mean', 'dapi_max', 'gfp_mean', 'gfp_max',
    'area'))