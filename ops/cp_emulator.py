# Features adapted from:
# Bray et al. 2016 Nature Protocols 11:1757-1774
# Cell Painting, a high-content image-based assay for morphological profiling using multiplexed fluorescent dyes
#
# Also helpful: 
# 	https://github.com/carpenterlab/2016_bray_natprot/wiki/What-do-Cell-Painting-features-mean%3F
#	https://raw.githubusercontent.com/wiki/carpenterlab/2016_bray_natprot/attachments/feature_names.txt
#	http://cellprofiler-manual.s3.amazonaws.com/CellProfiler-3.1.5/modules/measurement.html

# Note: their protocol uses 20X objective, 2x2 binning. 
# Length scales needed for feature extraction should technically be correspondingly scaled,
# e.g., 20X with 1x1 binning images should use suggested linear length scales * 2

# potential features to add: Hu moments, PFTAS

import numpy as np
from scipy.stats import median_absolute_deviation, rankdata # new in version 1.3.0
from scipy.spatial.distance import cdist
from scipy.ndimage.morphology import distance_transform_edt as distance_transform
from scipy.ndimage import map_coordinates
from mahotas.features import zernike_moments, haralick
from scipy.spatial import ConvexHull
from functools import partial
from itertools import starmap, combinations
from warnings import catch_warnings,simplefilter
import skimage.morphology
from skimage import img_as_ubyte
from ops.features import correlate_channels_masked, masked
from ops.utils import subimage
from ops.io_hdf import read_hdf_image
from ops.io import read_stack as read
from ops.firesnake import Snake
import pandas

def apply_extract_features_cp(well_tile,filepattern):
    wildcards = {'well':well_tile[0],'tile':well_tile[1]}
    filepattern.update(wildcards)
    stacked = read_hdf_image(name(filepattern))
    nuclei = read(name(filepattern,subdir='process_ph',tag='nuclei',ext='tif'))
    cells = read(name(filepattern,subdir='process_ph',tag='cells',ext='tif'))
    df_result = Snake._extract_phenotype_cp(data_phenotype=stacked,
                                            nuclei=nuclei,
                                            cells=cells,
                                            wildcards=wildcards,
                                            nucleus_channels=[0,1,2,3],
                                            cell_channels=[0,1,2,3],
                                            channel_names=['dapi','tubulin','gh2ax','phalloidin']
                                           )
    df_result.to_csv(name(filepattern,subdir='process_ph',tag='cp_phenotype',ext='csv'))

EDGE_CONNECTIVITY = 2

ZERNIKE_DEGREE = 9

GRANULARITY_BACKGROUND = 10 #this should be a bit larger than the radius of the features, i.e., "granules", of interest after downsampling
GRANULARITY_BACKGROUND_DOWNSAMPLE = 1
GRANULARITY_DOWNSAMPLE = 1
GRANULARITY_LENGTH = 16

# # MeasureCorrelation:'Measure the intensity correlation between all channels, within all objects.'
# #     Hidden:5
# #     Hidden:3
# #     Select an image to measure:DNA
# #     Select an image to measure:Mito
# #     Select an image to measure:ER
# #     Select an image to measure:RNA
# #     Select an image to measure:AGP
# #     Select where to measure correlation:Within objects
# #     Select an object to measure:Nuclei
# #     Select an object to measure:Cells
# #     Select an object to measure:Cytoplasm

# This module is now called "MeasureColocalization" in CellProfiler

correlation_features ={
	'correlation' : lambda r: [correlate_channels_masked(r,first,second) 
	for first,second in combinations(list(range(r.intensity_image_full.shape[-3])),2)],
	'lstsq_slope' : lambda r: [lstsq_slope(r,first,second) 
	for first,second in combinations(list(range(r.intensity_image_full.shape[-3])),2)],
	# costes threshold algorithm not working well/at all currently
	'colocalization' : lambda r: cp_colocalization_all_channels(r,costes=False)
}

correlation_columns = [
'correlation_{first}_{second}',
'lstsq_slope_{first}_{second}'
]

colocalization_columns = [
'overlap_{first}_{second}',
'K_{first}_{second}',
'K_{second}_{first}',
'manders_{first}_{second}',
'manders_{second}_{first}',
'rwc_{first}_{second}',
'rwc_{second}_{first}'
]

# # MeasureGranularity:'Measure the granularity characteristics across all images. Note that this is a per-image measure, and therefore, will not appear in the per-well profiles generated by the per-cell features.'
# #     Image count:5
# #     Object count:1
# #     Select an image to measure:DNA
# #     Subsampling factor for granularity measurements:0.25
# #     Subsampling factor for background reduction:0.25
# #     Radius of structuring element:10
# #     Range of the granular spectrum:16
# #     Select objects to measure:Nuclei
# #     Object count:3
# #     Select an image to measure:Mito
# #     Subsampling factor for granularity measurements:0.25
# #     Subsampling factor for background reduction:0.25
# #     Radius of structuring element:10
# #     Range of the granular spectrum:16
# #     Select objects to measure:Cells
# #     Select objects to measure:Nuclei
# #     Select objects to measure:Cytoplasm
# #     Object count:3
# #     Select an image to measure:ER
# #     Subsampling factor for granularity measurements:0.25
# #     Subsampling factor for background reduction:0.25
# #     Radius of structuring element:10
# #     Range of the granular spectrum:16
# #     Select objects to measure:Cells
# #     Select objects to measure:Nuclei
# #     Select objects to measure:Cytoplasm
# #     Object count:3
# #     Select an image to measure:AGP
# #     Subsampling factor for granularity measurements:0.25
# #     Subsampling factor for background reduction:0.25
# #     Radius of structuring element:10
# #     Range of the granular spectrum:16
# #     Select objects to measure:Cells
# #     Select objects to measure:Cytoplasm
# #     Select objects to measure:Nuclei
# #     Object count:3
# #     Select an image to measure:RNA
# #     Subsampling factor for granularity measurements:0.25
# #     Subsampling factor for background reduction:0.25
# #     Radius of structuring element:10
# #     Range of the granular spectrum:16
# #     Select objects to measure:Cells
# #     Select objects to measure:Cytoplasm
# #     Select objects to measure:Nuclei

# In CellProfiler this is a per-image metric, but is implemented here as a per-object metric.
# to re-produce values from paper, use start_radius = 10, spectrum_length = 16, sample=sample_background=0.25
# values here to optimize for fine speckles in single cells: THESE PARAMETERS ARE HIGHLY EXPERIMENT-DEPENDENT

granularity_features = {
	'granularity_spectrum' : lambda r: granularity_spectrum(r.intensity_image_full, r.image, 
		background_radius=GRANULARITY_BACKGROUND, spectrum_length=GRANULARITY_LENGTH, 
		downsample=GRANULARITY_DOWNSAMPLE, background_downsample=GRANULARITY_BACKGROUND_DOWNSAMPLE)
}

# # MeasureObjectIntensity:'Measure the intensity characteristics from all channels, within all objects.'
# # 	Hidden:5
# #     Select an image to measure:DNA
# #     Select an image to measure:ER
# #     Select an image to measure:RNA
# #     Select an image to measure:AGP
# #     Select an image to measure:Mito
# #     Select objects to measure:Nuclei
# #     Select objects to measure:Cytoplasm
# #     Select objects to measure:Cells

intensity_features = {
	'int': lambda r: r.intensity_image[r.image].sum(),
	'mean': lambda r: r.intensity_image[r.image].mean(),
	'std': lambda r: np.std(r.intensity_image[r.image]),
	'max': lambda r: r.intensity_image[r.image].max(),
	'min': lambda r: r.intensity_image[r.image].min(),
	'int_edge': lambda r: r.intensity_image[boundaries(r.filled_image,mode='inner',connectivity=EDGE_CONNECTIVITY)].sum(),
	'mean_edge': lambda r: r.intensity_image[boundaries(r.filled_image,mode='inner',connectivity=EDGE_CONNECTIVITY)].mean(),
	'std_edge': lambda r: np.std(r.intensity_image[boundaries(r.filled_image,mode='inner',connectivity=EDGE_CONNECTIVITY)]),
	'max_edge': lambda r: r.intensity_image[boundaries(r.filled_image,mode='inner',connectivity=EDGE_CONNECTIVITY)].max(),
	'min_edge': lambda r: r.intensity_image[boundaries(r.filled_image,mode='inner',connectivity=EDGE_CONNECTIVITY)].min(),
	'mass_displacement': lambda r: np.sqrt(((np.array(r.local_centroid) - np.array(r.weighted_local_centroid))**2).sum()),
	'lower_quartile': lambda r: np.percentile(r.intensity_image[r.image],25),
    'median': lambda r: np.median(r.intensity_image[r.image]),
    'mad': lambda r: median_absolute_deviation(r.intensity_image[r.image],scale=1),
    'upper_quartile': lambda r: np.percentile(r.intensity_image[r.image],75),
    'center_mass_r': lambda r: r.weighted_local_centroid[0],
    'center_mass_c': lambda r: r.weighted_local_centroid[1],
    'max_location_r': lambda r: np.unravel_index(np.argmax(r.intensity_image), (r.image).shape)[0],
    'max_location_c': lambda r: np.unravel_index(np.argmax(r.intensity_image), (r.image).shape)[1]
    }


# # MeasureObjectNeighbors:'Measure the adjacency statistics for the cells. Cells within 5 pixels of each other are considered neighbors.'
# #     Select objects to measure:Cells
# #     Select neighboring objects to measure:Cells
# #     Method to determine neighbors:Within a specified distance
# #     Neighbor distance:5
# #     Retain the image of objects colored by numbers of neighbors?:No
# #     Name the output image:ObjectNeighborCount
# #     Select colormap:Default
# #     Retain the image of objects colored by percent of touching pixels?:No
# #     Name the output image:PercentTouching
# #     Select a colormap:Default

# # MeasureObjectNeighbors:'Measure the adjacency statistics for the nuclei. Nuclei within 1 pixel of each other are considered neighbors.'
# # 	Select objects to measure:Nuclei
# #     Select neighboring objects to measure:Nuclei
# #     Method to determine neighbors:Within a specified distance
# #     Neighbor distance:1
# #     Retain the image of objects colored by numbers of neighbors?:No
# #     Name the output image:ObjectNeighborCount
# #     Select colormap:Default
# #     Retain the image of objects colored by percent of touching pixels?:No
# #     Name the output image:PercentTouching
# #     Select a colormap:Default

# # MeasureObjectNeighbors:'Measure the adjacency statistics for the cells. Cells touching each other are considered neighbors.'
# #     Select objects to measure:Cells
# #     Select neighboring objects to measure:Cells
# #     Method to determine neighbors:Adjacent
# #     Neighbor distance:5
# #     Retain the image of objects colored by numbers of neighbors?:No
# #     Name the output image:ObjectNeighborCount
# #     Select colormap:Default
# #     Retain the image of objects colored by percent of touching pixels?:No
# #     Name the output image:PercentTouching
# #     Select a colormap:Default

# appears that CellProfiler calculates FirstClosestDistance, SecondClosestDistance, and AngleBetweenNeighbors
# as closest distance between centers of objects identified as neighbors using distances to perimeter. If no
# neighbor close enough to perimeter, then no distance calculated. Here, I have calculated first_neighbor_distance,
# second_neighbor_distances, and angle_between_neighbors using objects with smallest distance between centers, 
# regardless of distance between perimeters. This produces a single metric for all cells, even if multiple distance 
# thresholds are used to find number of perimeter neighbors.

# these features are dependent on information from the entire field-of-view, thus are not directly extracted with regionprops

def neighbor_measurements(labeled, distances=[1,10],n_cpu=1):
	from pandas import concat

	dfs = [object_neighbors(labeled,distance=distance).rename(columns=lambda x: x+'_'+str(distance)) for distance in distances]

	dfs.append(closest_objects(labeled,n_cpu=n_cpu).drop(columns=['first_neighbor','second_neighbor']))

	return concat(dfs,axis=1,join='outer').reset_index()

# # MeasureObjectRadialDistribution:'Measure the radial intensity distribution characteristics in all objects. The object is "binned" into radial annuli and statistics are measured for each bin.'
# #     Hidden:4
# #     Hidden:3
# #     Hidden:1
# #     Select an image to measure:ER
# #     Select an image to measure:RNA
# #     Select an image to measure:AGP
# #     Select an image to measure:Mito
# #     Select objects to measure:Cells
# #     Object to use as center?:These objects
# #     Select objects to use as centers:None
# #     Select objects to measure:Nuclei
# #     Object to use as center?:These objects
# #     Select objects to use as centers:None
# #     Select objects to measure:Cytoplasm
# #     Object to use as center?:These objects
# #     Select objects to use as centers:None
# #     Scale the bins?:Yes
# #     Number of bins:4
# #     Maximum radius:100

# This module no longer exists in CellProfiler -> MeasureObjectIntensityDistribution
# But do not use intensity zernike's--slow and not useful: https://github.com/CellProfiler/CellProfiler/issues/2220
# center defined as point farthest from edge = np.argmax(distance_transform(np.pad(r.filled_image,1,'constant')))

intensity_distribution_features = {
	'intensity_distribution' : lambda r: np.array(
		measure_intensity_distribution(r.filled_image,r.image,r.intensity_image,bins=4)
		).reshape(-1)
	# to minimize re-computing values, outputs a numpy array of length 3*bins. order is [FracAtD, MeanFrac, RadialCV]*bins
}

intensity_distribution_columns = {
	'intensity_distribution_0':'frac_at_d_0',
	'intensity_distribution_1':'frac_at_d_1',
	'intensity_distribution_2':'frac_at_d_2',
	'intensity_distribution_3':'frac_at_d_3',
	'intensity_distribution_4':'mean_frac_0',
	'intensity_distribution_5':'mean_frac_1',
	'intensity_distribution_6':'mean_frac_2',
	'intensity_distribution_7':'mean_frac_3',
	'intensity_distribution_8':'radial_cv_0',
	'intensity_distribution_9':'radial_cv_1',
	'intensity_distribution_10':'radial_cv_2',
	'intensity_distribution_11':'radial_cv_3',
}

# # MeasureObjectSizeShape:'Measure the morpholigical features of all objects.'
# #     Select objects to measure:Cells
# #     Select objects to measure:Nuclei
# #     Select objects to measure:Cytoplasm
# #     Calculate the Zernike features?:Yes

shape_features = {
	'area'    : lambda r: r.area,
	'perimeter' : lambda r: r.perimeter,
	'form_factor': lambda r:4*np.pi*r.area/(r.perimeter)**2, #isoperimetric quotient
	'solidity': lambda r: r.solidity,
	'extent': lambda r: r.extent,
	'euler_number': lambda r: r.euler_number,
	'centroid_r': lambda r: r.local_centroid[0], # ACTUALLY SHOULD BE POINT FARTHEST FROM EDGE?
	'centroid_c': lambda r: r.local_centroid[1], # ACTUALLY SHOULD BE POINT FARTHEST FROM EDGE?
	'eccentricity': lambda r: r.eccentricity,
	'major_axis' : lambda r: r.major_axis_length,
    'minor_axis' : lambda r: r.minor_axis_length,
    'orientation' : lambda r: r.orientation,
    'compactness' : lambda r: 2*np.pi*(r.moments_central[0,2]+r.moments_central[2,0])/(r.area**2),
    'max_radius' : lambda r: distance_transform(np.pad(r.filled_image,1,'constant')).max(), #filled_image or image?
    'median_radius' : lambda r: np.median(distance_transform(np.pad(r.filled_image,1,'constant'))[1:-1,1:-1][r.filled_image]), #filled_image or image?
    'mean_radius' : lambda r: distance_transform(np.pad(r.filled_image,1,'constant'))[1:-1,1:-1][r.filled_image].mean(), #filled_image or image?
    'feret_diameter' : lambda r: min_max_feret_diameter(r.coords),
    # 'min_feret' : lambda r: minimum_feret_diameter(r.coords),
    # 'max_feret' : lambda r: maximum_feret_diameter(r.coords),
    'zernike' : lambda r: zernike_minimum_enclosing_circle(r.coords, degree=ZERNIKE_DEGREE) # cp/centrosome zernike divides zernike magnitudes by minimum enclosing circle magnitude; unclear why
}

zernike_nums = ['zernike_'+str(radial)+'_'+str(azimuthal) 
for radial in range(ZERNIKE_DEGREE+1) 
for azimuthal in range(radial%2,radial+2,2)]

shape_columns = {'zernike_'+str(num):zernike_num for num,zernike_num in enumerate(zernike_nums)}
shape_columns.update({
	'feret_diameter_0':'min_feret_diameter',
	'feret_diameter_1':'max_feret_diameter',
})

# # MeasureTexture:'Measure the texture features in all objects, against all 5 channels, using multiple spatial scales.'
# #     Hidden:5
# #     Hidden:3
# #     Hidden:3
# #     Select an image to measure:DNA
# #     Select an image to measure:ER
# #     Select an image to measure:RNA
# #     Select an image to measure:AGP
# #     Select an image to measure:Mito
# #     Select objects to measure:Cells
# #     Select objects to measure:Cytoplasm
# #     Select objects to measure:Nuclei
# #     Texture scale to measure:3
# #     Angles to measure:Horizontal
# #     Texture scale to measure:5
# #     Angles to measure:Horizontal
# #     Texture scale to measure:10
# #     Angles to measure:Horizontal
# #     Measure Gabor features?:Yes
# #     Number of angles to compute for Gabor:4

# Gabor features not used in more recent version of cell painting analysis pipeline
# scales actually used are 5, 10, 20
# each haralick feature outputs 13 features
# unclear how cell profiler aggregates results from all 4 directions, most likely is mean

# Haralick references:
#	Haralick RM, Shanmugam K, Dinstein I. (1973), “Textural Features for Image Classification” IEEE Transaction on Systems Man, Cybernetics, SMC-3(6):610-621.
#	http://murphylab.web.cmu.edu/publications/boland/boland_node26.html

# if having issues with ValueError's can try: haralick, except: return [np.nan]*13

texture_features = {
	'haralick_5'  : lambda r: ubyte_haralick(r.intensity_image, ignore_zeros=True, distance=5,  return_mean=True),
	'haralick_10' : lambda r: ubyte_haralick(r.intensity_image, ignore_zeros=True, distance=10, return_mean=True),
	'haralick_20' : lambda r: ubyte_haralick(r.intensity_image, ignore_zeros=True, distance=20, return_mean=True)
}

######################################################################################################################################

grayscale_features = {**intensity_features,**intensity_distribution_features,**texture_features,**granularity_features}

######################################################################################################################################

def lstsq_slope(r,first,second):
	A = masked(r,first)
	B = masked(r,second)

	filt = A > 0
	if filt.sum() == 0:
	    return np.nan

	A = A[filt]
	B  = B[filt]
	slope = np.linalg.lstsq(np.vstack([A,np.ones(len(A))]).T,B,rcond=-1)[0][0]

	return slope

def cp_colocalization_all_channels(r,**kwargs):
	results = [cp_colocalization(r,first,second,**kwargs) 
	for first,second in combinations(list(range(r.intensity_image_full.shape[-3])),2)]

	return [single_result for combination_results in results for single_result in combination_results]

def cp_colocalization(r,first,second,threshold=0.15,costes=False):
	"""Measures overlap, k1/k2, manders, and rank weighted colocalization coefficients.
	References:
	http://www.scian.cl/archivos/uploads/1417893511.1674 starting at slide 35
	Singan et al. (2011) "Dual channel rank-based intensity weighting for quantitative 
	co-localization of microscopy images", BMC Bioinformatics, 12:407.
	"""
	results = []

	A = masked(r,first).astype(float)
	B = masked(r,second).astype(float)

	filt = A > 0
	if filt.sum() == 0:
	    return np.nan

	A = A[filt]
	B  = B[filt]

	A_thresh,B_thresh = (threshold*A.max(), threshold*B.max())

	A_total, B_total = A[A>A_thresh].sum(), B[B>B_thresh].sum()

	mask = (A > A_thresh) & (B > B_thresh)

	overlap = (A[mask]*B[mask]).sum()/np.sqrt((A[mask]**2).sum()*(B[mask]**2).sum())

	results.append(overlap)

	K1 = (A[mask]*B[mask]).sum()/(A[mask]**2).sum()
	K2 = (A[mask]*B[mask]).sum()/(B[mask]**2).sum()

	results.extend([K1,K2])

	M1 = A[mask].sum()/A_total
	M2 = B[mask].sum()/B_total

	results.extend([M1,M2])

	A_ranks = rankdata(A,method='dense')
	B_ranks = rankdata(B,method='dense')

	R = max([A_ranks.max(),B_ranks.max()])
	weight = ((R-abs(A_ranks-B_ranks))/R)[mask]
	RWC1 = (A[mask]*weight).sum()/A_total
	RWC2 = (B[mask]*weight).sum()/B_total

	results.extend([RWC1,RWC2])

	if costes:
		A_costes,B_costes = costes_threshold(A,B)
		mask_costes = (A > A_costes) & (B > B_costes)

		C1 = A[mask_costes].sum()/A[A>A_costes].sum()
		C2 = B[mask_costes].sum()/B[B>B_costes].sum()

		results.extend([C1,C2])

	return results

def costes_threshold(A,B,step=1,pearson_cutoff=0):
	# Costes et al. (2004) Biophysical Journal, 86(6) 3993-4003
	mask = (A>0)|(B>0)

	A = A[mask]
	B = B[mask]

	A_var = np.var(A,ddof=1)
	B_var = np.var(B,ddof=1)

	Z = A+B
	Z_var = np.var(Z,ddof=1)

	covar = 0.5 * (Z_var - (A_var+B_var))

	a = (B_var-A_var)+np.sqrt((B_var-A_var)**2 + 4*(covar**2))/(2*covar)

	b = B.mean()-a*A.mean()

	threshold = A.max()

	if (len(np.unique(A)) > 10**4) & (step<100):
		step = 100

	# could also try the histogram bisection method used in Coloc2
	# https://github.com/fiji/Colocalisation_Analysis
	for threshold in np.unique(A)[::-step]:
		below = (A<threshold)|(B<(a*threshold+b))
		pearson = np.mean((A[below]-A[below].mean())*(B[below]-B[below].mean())/(A[below].std()*B[below].std()))

		if pearson <= pearson_cutoff:
			break

	return (threshold,a*threshold+b)

def granularity_spectrum(grayscale, labeled, background_radius=5, spectrum_length=16, downsample=1, background_downsample=0.5):
	"""Returns granularity spectrum as defined in the CellProfiler documentation.
	Scaled so that units are approximately the % of new granules stuck in imaginary sieve when moving to 
	size specified by spectrum component
	Helpful resources:
	Maragos P. “Pattern spectrum and multiscale shape representation”,
		IEEE Transactions on Pattern Analysis and Machine Intelligence, 
		VOL 11, NO 7, pp. 701-716, 1989
	Vincent L. (1992) “Morphological Area Opening and Closing for
		Grayscale Images”, Proc. NATO Shape in Picture Workshop,
		Driebergen, The Netherlands, pp. 197-208.
	https://en.wikipedia.org/wiki/Granulometry_(morphology)
	http://www.ravkin.net/presentations/Statistical%20properties%20of%20algorithms%20for%20analysis%20of%20cell%20images.pdf
	"""
	intensity_image = grayscale.copy()
	image = labeled.copy()


	i_sub,j_sub = np.mgrid[0:image.shape[0]*downsample, 0:image.shape[1]*downsample].astype(float)/downsample
	if downsample < 1:
		intensity_image = map_coordinates(intensity_image,(i_sub,j_sub),order=1)
		image = map_coordinates(image.astype(float),(i_sub,j_sub))>0.9

	if background_downsample <1:
		i_sub_sub,j_sub_sub = (np.mgrid[0:image.shape[0]*background_downsample, 
			0:image.shape[1]*background_downsample].astype(float)/background_downsample)
		background_intensity = map_coordinates(intensity_image,(i_sub_sub,j_sub_sub),order=1)
		background_mask = map_coordinates(image.astype(float),(i_sub_sub,j_sub_sub))>0.9
	else:
		background_intensity = intensity_image
		background_mask = image

	selem = skimage.morphology.disk(background_radius,dtype=bool)

	# cellprofiler masks before and between erosion/dilation steps here--
	# this creates unwanted edge effects here. Combine erosion/dilation into opening
	# background = skimage.morphology.erosion(background_intensity*background_mask,selem=selem)
	# background = skimage.morphology.dilation(background,selem=selem)
	background = skimage.morphology.opening(background_intensity,selem=selem)

	# rescaling
	if background_downsample < 1:
		# rescale background to match intensity_image
		i_sub *= float(background.shape[0]-1)/float(image.shape[0]-1)
		j_sub *= float(background.shape[1]-1)/float(image.shape[1]-1)
		background = map_coordinates(background,(i_sub,j_sub),order=1)

	# remove background
	intensity_image -= background
	intensity_image[intensity_image<0] = 0

	# calculate granularity spectrum
	start = np.mean(intensity_image[image])

	# cellprofiler also does unwanted masking step here
	erosion = intensity_image

	current = start

	footprint = skimage.morphology.disk(1,dtype=bool)

	spectrum = []
	for _ in range(spectrum_length):
		previous = current.copy()
		# cellprofiler does unwanted masking step here
		erosion = skimage.morphology.erosion(erosion, selem=footprint)
		# masking okay here--inhibits bright regions from outside object being propagated into the image
		reconstruction = skimage.morphology.reconstruction(erosion*image, intensity_image, selem=footprint)
		current = np.mean(reconstruction[image])
		spectrum.append((previous - current) * 100 / start)

	return spectrum

def boundaries(labeled,connectivity=1,mode='inner',background=0):
    """Supplement skimage.segmentation.find_boundaries to include image edge pixels of 
    labeled regions as boundary
    """
    from skimage.segmentation import find_boundaries
    kwargs = dict(connectivity=connectivity,
        mode=mode,
        background=background
        )
    # if mode == 'inner':
    pad_width = 1
    # else:
    #     pad_width = connectivity

    padded = np.pad(labeled,pad_width=pad_width,mode='constant',constant_values=background)
    return find_boundaries(padded,**kwargs)[...,pad_width:-pad_width,pad_width:-pad_width]

def closest_objects(labeled,n_cpu=1):
	from ops.process import feature_table
	from scipy.spatial import cKDTree

	features = {
	'i'       : lambda r: r.centroid[0],
    'j'       : lambda r: r.centroid[1],
    'label'   : lambda r: r.label
    }
	
	df = feature_table(labeled,labeled,features)

	kdt = cKDTree(df[['i','j']])

	distances,indexes = kdt.query(df[['i','j']],3,n_jobs=n_cpu)

	df['first_neighbor'],df['first_neighbor_distance'] = indexes[:,1],distances[:,1]
	df['second_neighbor'],df['second_neighbor_distance'] = indexes[:,2],distances[:,2]

	first_neighbors = df[['i','j']].values[df['first_neighbor'].values]
	second_neighbors = df[['i','j']].values[df['second_neighbor'].values]

	angles = [angle(v,p0,p1) 
          for v,p0,p1 
          in zip(df[['i','j']].values,first_neighbors,second_neighbors)]

	df['angle_between_neighbors'] = np.array(angles)*(180/np.pi)

	return df.drop(columns=['i','j']).set_index('label')

def object_neighbors(labeled, distance=1):
	from skimage.measure import regionprops
	from pandas import DataFrame
	
	outlined = boundaries(labeled,connectivity=EDGE_CONNECTIVITY,mode='inner')*labeled

	regions = regionprops(labeled)

	bboxes = [r.bbox for r in regions]

	labels = [r.label for r in regions]

	neighbors_disk = skimage.morphology.disk(distance)

	perimeter_disk = cp_disk(distance+0.5)

	info_dicts = [neighbor_info(labeled,outlined,label,bbox,distance,neighbors_disk,perimeter_disk) for label,bbox in zip(labels,bboxes)]

	return DataFrame(info_dicts).set_index('label')

def neighbor_info(labeled,outlined,label,bbox,distance,neighbors_disk=None,perimeter_disk=None):
	if neighbors_disk is None:
		neighbors_disk = skimage.morphology.disk(distance)
	if perimeter_disk is None:
		perimeter_disk = cp_disk(distance+0.5)

	label_mask = subimage(labeled,bbox,pad=distance)
	outline_mask = subimage(outlined,bbox,pad=distance) == label

	dilated = skimage.morphology.binary_dilation(label_mask==label,selem=neighbors_disk)
	neighbors = np.unique(label_mask[dilated])
	neighbors = neighbors[(neighbors!=0)&(neighbors!=label)]
	n_neighbors = len(neighbors)

	dilated_neighbors = skimage.morphology.binary_dilation((label_mask!=label)&(label_mask!=0),selem=perimeter_disk)
	percent_touching = (outline_mask&dilated_neighbors).sum()/outline_mask.sum()

	return {'label':label,'number_neighbors':n_neighbors,'percent_touching':percent_touching}

def cp_disk(radius):
    """Create a disk structuring element for morphological operations
    
    radius - radius of the disk
    """
    iradius = int(radius)
    x, y = np.mgrid[-iradius : iradius + 1, -iradius : iradius + 1]
    radius2 = radius * radius
    strel = np.zeros(x.shape)
    strel[x * x + y * y <= radius2] = 1
    return strel

def measure_intensity_distribution(filled_image, image, intensity_image, bins=4):
	binned, center = binned_rings(filled_image,image,bins)

	frac_at_d = np.array([intensity_image[binned==bin_ring].sum() for bin_ring in range(1,bins+1)])/intensity_image[image].sum()

	frac_pixels_at_d = np.array([(binned==bin_ring).sum() for bin_ring in range(1,bins+1)])/image.sum()

	mean_frac = frac_at_d/frac_pixels_at_d

	wedges = radial_wedges(image,center)

	# for the case where no pixels are in a bin + wedge combination
	with catch_warnings():
		simplefilter("ignore",category=RuntimeWarning)
		mean_binned_wedges = np.array([np.array([intensity_image[(wedges==wedge)&(binned==bin_ring)].mean() 
			for wedge in range(1,9)]) 
			for bin_ring in range(1,bins+1)])

	radial_cv = np.nanstd(mean_binned_wedges,axis=1)/np.nanmean(mean_binned_wedges,axis=1)

	return frac_at_d,mean_frac,radial_cv

def binned_rings(filled_image,image,bins):
	"""takes filled image, separates into number of rings specified by bins, 
	with the ring size normalized by the radius at that approximate angle"""

	# returns distance to center point, normalized by distance to edge along
	# that direction, [0,1]; 0 = center point, 1 = points outside the image
	normalized_distance,center = normalized_distance_to_center(filled_image)

	binned = np.ceil(normalized_distance*bins)

	binned[binned==0]=1

	return np.multiply(np.ceil(binned),image),center

def normalized_distance_to_center(filled_image):
	"""regions outside of labeled image have normalized distance of 1"""

	distance_to_edge = distance_transform(np.pad(filled_image,1,'constant'))[1:-1,1:-1]

	max_distance = distance_to_edge.max()

	# median of all points furthest from edge
	center = tuple(np.median(np.where(distance_to_edge==max_distance),axis=1).astype(int))

	mask = np.ones(filled_image.shape)
	mask[center[0],center[1]] = 0

	distance_to_center = distance_transform(mask)

	return distance_to_center/(distance_to_center+distance_to_edge),center

def radial_wedges(image, center):
	"""returns shape divided into 8 radial wedges, each comprising a 45 degree slice
	of the shape from center. Output labeleing convention:
	    i > +
	      \\ 3 || 4 // 
	 +  7  \\  ||  // 8
	 ^  ===============
	 j  5  //  ||  \\ 6
	      // 1 || 2 \\ 
	"""
	i, j = np.mgrid[0 : image.shape[0], 0 : image.shape[1]]

	positive_i,positive_j = (i > center[0], j> center[1])

	abs_i_greater_j = abs(i - center[0]) > abs(j - center[1])

	return ((positive_i + positive_j * 2 + abs_i_greater_j * 4 + 1)*image).astype(int)

def min_max_feret_diameter(coords):
	hull_vertices = coords[ConvexHull(coords).vertices]

	antipodes = get_antipodes(hull_vertices)

	point_distances = np.array(list(starmap(cdist,zip(np.stack([antipodes[:,:2],antipodes[:,2:4]]),antipodes[None,:,4:6]))))

	return antipodes[:,6].min(),point_distances.max()

# def minimum_feret_diameter(coords):
# 	hull_vertices = coords[ConvexHull(coords).vertices]

# 	distances = get_antipodes(hull_vertices)[:,6]

# 	return distances.min()

# def maximum_feret_diameter(coords):
# 	hull_vertices = coords[ConvexHull(coords).vertices]

# 	antipodes = get_antipodes(hull_vertices)[:,:6]

# 	distances = np.array(list(starmap(cdist,zip(np.stack([antipodes[:,:2],antipodes[:,2:4]]),antipodes[None,:,4:6]))))

# 	return distances.max()

def get_antipodes(vertices):
    """rotating calipers"""
    antipodes = []
    # iterate through each vertex
    for v_index,vertex in enumerate(vertices):
        current_distance = 0
        candidates = vertices[circular_index(v_index+1,v_index-2,len(vertices))]

        # iterate through each vertex except current and previous
        for c_index,candidate in enumerate(candidates):

            #calculate perpendicular distance from candidate_antipode to line formed by current and previous vertex
            d = perpendicular_distance(vertex,vertices[v_index-1],candidate)
            
            if d < current_distance:
                # previous candidate is a "breaking" antipode
                antipodes.append(np.concatenate([vertex,vertices[v_index-1],candidates[c_index-1],current_distance[None]]))
                break
                
            elif d >= current_distance:
                # not a breaking antipode
                if d == current_distance:
                    # previous candidate is a "non-breaking" antipode
                    antipodes.append(np.concatenate([vertex,vertices[v_index-1],candidates[c_index-1],current_distance[None]]))
                    if c_index == len(candidates)-1:
                        antipodes.append(np.concatenate([vertex,vertices[v_index-1],candidates[c_index],current_distance[None]]))
                current_distance = d

    return np.array(antipodes)

def circular_index(first,last,length):
    if last<first:
        last += length
        return np.arange(first, last+1)%length
    elif last==first:
        return np.roll(range(length),-first)
    else:
        return np.arange(first,last+1)

def perpendicular_distance(line_p0,line_p1,p0):
    if line_p0[0]==line_p1[0]:
        return abs(line_p0[0]-p0[0])
    elif line_p0[1]==line_p1[1]:
        return abs(line_p0[1]-p0[1])
    else:
        return abs(((line_p1[1]-line_p0[1])*(line_p0[0]-p0[0])-(line_p1[0]-line_p0[0])*(line_p0[1]-p0[1]))/
                np.sqrt((line_p1[1]-line_p0[1])**2+(line_p1[0]-line_p0[0])**2))

# class FeretHull(ConvexHull):
# 	"""Subclass of scipy.spatial.ConvexHull,
# 	adds Feret Diamter functionality
# 	"""
# 	def __init__(self,points,incremental=False,qhull_options=None):
# 		import pandas as pd

# 		super().__init__(points,incremental,qhull_options) # inherit ConvexHull methods

# 		def perpendicular_distance(line_p0,line_p1,p0):
# 		    if line_p0[0]==line_p1[0]:
# 		        return abs(line_p0[0]-p0[0])
# 		    elif line_p0[1]==line_p1[1]:
# 		        return abs(line_p0[1]-p0[1])
# 		    else:
# 		        return abs(((line_p1[1]-line_p0[1])*(line_p0[0]-p0[0])-(line_p1[0]-line_p0[0])*(line_p0[1]-p0[1]))/
# 		                np.sqrt((line_p1[1]-line_p0[1])**2+(line_p1[0]-line_p0[0])**2))

# 		def get_antipodes_dataframe(vertices):
# 		    """rotating calipers"""
# 		    antipodes = []
# 		    # iterate through each vertex
# 		    for v_index,vertex in enumerate(vertices):
# 		        current_distance = 0
# 		        candidates = vertices[circular_index(v_index+1,v_index-2,len(vertices))]

# 		        # iterate through each vertex except current and previous
# 		        for c_index,candidate in enumerate(candidates):

# 		            #calculate perpendicular distance from candidate_antipode to line formed by current and previous vertex
# 		            d = perpendicular_distance(vertex,vertices[v_index-1],candidate)
		            
# 		            if d < current_distance:
# 		                # previous candidate is a "breaking" antipode
# 		                antipodes.append({'line_vertex_0':vertex,'line_vertex_1':vertices[v_index-1],'point_vertex':candidates[c_index-1],'distance':current_distance})
# 		                break
		                
# 		            elif d >= current_distance:
# 		                # not a breaking antipode
# 		                if d == current_distance:
# 		                    # previous candidate is a "non-breaking" antipode
# 		                    antipodes.append({'line_vertex_0':vertex,'line_vertex_1':vertices[v_index-1],'point_vertex':candidates[c_index-1],'distance':current_distance})
# 		                    if c_index == len(candidates)-1:
# 		                        antipodes.append({'line_vertex_0':vertex,'line_vertex_1':vertices[v_index-1],'point_vertex':candidates[c_index],'distance':current_distance})
# 		                current_distance = d

# 		    return antipodes

# 		self.antipodes = pd.DataFrame(get_antipodes_dataframe(points[self.vertices]))
		
# 	def get_min_feret(self):
# 		# distance between antipode point and each line formed by other antipode point and adjacent vertices
# 		return self.antipodes['distance'].values.min()

# 	def get_max_feret(self):
# 		# distance between antipode points
# 		return np.array([cdist([antipode.line_vertex_0,antipode.line_vertex_1],[antipode.point_vertex]).max() 
# 			for _,antipode in self.antipodes.iterrows()]).max()

def zernike_minimum_enclosing_circle(coords,degree=9):
	image, center, diameter = minimum_enclosing_circle_shift(coords)

	return zernike_moments(image, radius=diameter/2, degree=degree, cm=center)

def minimum_enclosing_circle_shift(coords,pad=1):
	diameter,center = minimum_enclosing_circle(coords)

	# diameter = np.ceil(diameter)

	# have to adjust image size to fit minimum enclosing circle
	shift = np.round(diameter/2 - center)
	shifted = np.zeros((int(np.ceil(diameter)+pad),int(np.ceil(diameter)+pad)))
	# shift = np.round(np.array(shifted.shape)/2 - center)
	coords_shifted = (coords + shift).astype(int)
	shifted[coords_shifted[:,0],coords_shifted[:,1]] = 1
	center_shifted = center + shift

	return shifted, center_shifted, np.ceil(diameter)

def minimum_enclosing_circle(coords):
	# http://www.personal.kent.edu/~rmuhamma/Compgeometry/MyCG/CG-Applets/Center/centercli.htm
	# https://www.cs.princeton.edu/courses/archive/spring09/cos226/checklist/circle.html
	hull_vertices = coords[ConvexHull(coords).vertices]

	s0 = hull_vertices[0]
	s1 = hull_vertices[1]

	iterations = 0

	while True:

		remaining = hull_vertices[(hull_vertices!=s0).max(axis=1)&(hull_vertices!=s1).max(axis=1)]

		angles = np.array(list(map(partial(angle,p0=s0,p1=s1),remaining)))
		
		min_angle = angles.min()

		if min_angle >= np.pi/2:
			# circle diameter is s0-s1, center is mean of s0,s1
			diameter = np.sqrt(((s0-s1)**2).sum())
			center = (s0+s1)/2
			break

		vertex = remaining[np.argmin(angles)]

		remaining_angles = np.array(list(starmap(angle,zip([s1,s0],[s0,vertex],[vertex,s1]))))

		if remaining_angles.max() <= np.pi/2:
			# use circumscribing circle of s0,s1,vertex
			diameter,center = circumscribed_circle(s0,s1,vertex)
			break

		keep = [s0,s1][np.argmax(remaining_angles)]

		s0 = keep
		s1 = vertex

		iterations += 1

		if iterations == len(hull_vertices):
			print('maximum_enclosing_circle did not converge')
			diameter = center = None

	return diameter,center

def angle(vertex, p0, p1):
	v0 = p0 - vertex
	v1 = p1 - vertex

	cosine_angle = np.dot(v0, v1) / (np.linalg.norm(v0) * np.linalg.norm(v1))
	return np.arccos(cosine_angle)

def circumscribed_circle(p0,p1,p2):
	# https://en.wikipedia.org/wiki/Circumscribed_circle
	P = np.array([p0,p1,p2])

	Sx = (1/2)*np.linalg.det(np.concatenate([(P**2).sum(axis=1).reshape(3,1),P[:,1].reshape(3,1),np.ones((3,1))],axis=1))
	Sy = (1/2)*np.linalg.det(np.concatenate([P[:,0].reshape(3,1),(P**2).sum(axis=1).reshape(3,1),np.ones((3,1))],axis=1))
	a = np.linalg.det(np.concatenate([P,np.ones((3,1))],axis=1))
	b = np.linalg.det(np.concatenate([P,(P**2).sum(axis=1).reshape(3,1)],axis=1))

	center = np.array([Sx,Sy])/a
	diameter = 2*np.sqrt((b/a) + (np.array([Sx,Sy])**2).sum()/(a**2))
	return diameter,center

def ubyte_haralick(image,**kwargs):
	with catch_warnings():
		simplefilter("ignore",category=UserWarning)
		ubyte_image = img_as_ubyte(image)
	try:
		features = haralick(ubyte_image,**kwargs)
	except ValueError:
		features = [np.nan]*13

	return features