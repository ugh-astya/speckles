import napari
import numpy as np
from skimage import filters, morphology, measure, feature, segmentation, transform, draw, exposure
from scipy import ndimage
import re
import pandas as pd
import os
import warnings
from aicsimageio import AICSImage
import tifffile

def find_czi_files(folder_path,filter_str):
    czi_files = sorted([
        os.path.join(folder_path, f) for f in os.listdir(folder_path)
        if f.endswith('.czi') and (filter_str is None or filter_str in f)
    ])   #Finds all the czi files inside the path with the filter string and outputs their absolute path into a list. Does not search recursively inside folders.
    
    if czi_files:
        print(f"Found {len(czi_files)} .czi file(s) in '{folder_path}'" + 
              (f" matching '{filter_str}'" if filter_str else "") + ":\n")
        for f in czi_files:
            print(f" - {os.path.basename(f)}")
    else:
        print(f"No .czi files found in '{folder_path}'" +
              (f" matching '{filter_str}'" if filter_str else "") + ".")

    return czi_files

def find_tif_files(folder_path,filter_str):
    tif_files = sorted([
        os.path.join(folder_path, f) for f in os.listdir(folder_path)
        if f.endswith('.tif') and (filter_str is None or filter_str in f)
    ])   #Finds all the tif files inside the path with the filter string and outputs their absolute path into a list. Does not search recursively inside folders.
    
    if tif_files:
        print(f"Found {len(tif_files)} .tif file(s) in '{folder_path}'" + 
              (f" matching '{filter_str}'" if filter_str else "") + ":\n")
        for f in tif_files:
            print(f" - {os.path.basename(f)}")
    else:
        print(f"No .tif files found in '{folder_path}'" +
              (f" matching '{filter_str}'" if filter_str else "") + ".")

    return tif_files

def parse_image_properties(path):
    
    file=os.path.basename(path) #Extract filename from full path
    channels=file.split("_")[:3]    #Get the first 3 fields corresponding to channels from the filename
    c1=next((c for c in channels if '647' in c),None)   #First channel is 647, second 555 and so on..
    c2=next((c for c in channels if '555' in c),None)
    c3=next((c for c in channels if '488' in c),None)
    parts = file.split("_")
    cell_stage = parts[3]   #4th field should be embryo stage - L2C, 4C, Morula etc.
    condition  = parts[4]   #5th field should be the condition - WT, SON KD, SRRM2 KD etc.

    return c1,c2,c3,cell_stage,condition,file


def view_napari(image_path, c=4, colors=['yellow','red','green','blue'], layer_names=['647','555','488','DAPI'],show=True,title="napari"):
    #os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    viewer = napari.Viewer()
    viewer.open(image_path, plugin='napari-aicsimageio')

    # Validate channel count
    if len(viewer.layers) != c:
        print(f"Warning: expected {c} channels but found {len(viewer.layers)}")

    # Slice names and colors to match actual channel count
    names  = layer_names[4-c:]
    colormaps = colors[4-c:]

    # Rename and recolor in one pass
    image_layers = [l for l in viewer.layers if isinstance(l, napari.layers.Image)]
    for layer, name, color in zip(image_layers, names, colormaps):
        layer.name     = name
        layer.colormap = color

    return viewer

def view_tif_napari(path, c=4, colors=['yellow','red','green','blue'],
                     layer_names=['647','555','488','DAPI'], title="napari"):

    viewer = napari.Viewer(title=title)
    viewer.open(path)

    names = layer_names[4-c:]
    colormaps = colors[4-c:]

    image_layers = [l for l in viewer.layers if isinstance(l, napari.layers.Image)]

    if len(image_layers) == c:
        # Already split into separate layers
        for layer, name, color in zip(image_layers, names, colormaps):
            layer.name = name
            layer.colormap = color

    elif len(image_layers) == 1:
        combined = image_layers[0].data
        print(f"Loaded as single layer with shape: {combined.shape}")

        # Find which axis has size == c (channel axis)
        channel_axis = next((i for i, s in enumerate(combined.shape) if s == c), None)

        if channel_axis is None:
            print(f"Could not find an axis of size {c} in shape {combined.shape}")
            return viewer

        # Reorder so channel axis is first -> CZYX
        combined = np.moveaxis(combined, channel_axis, 0)
        channel_axis = 0
        print(f"Reordered to CZYX, new shape: {combined.shape}")

        viewer.layers.remove(image_layers[0])

        for i in range(c):
            channel_data = np.take(combined, indices=i, axis=channel_axis)
            viewer.add_image(
                channel_data,
                name=names[i],
                colormap=colormaps[i],
                blending='additive'
            )

    else:
        print(f"Warning: expected {c} channels but found {len(image_layers)} "
              f"layer(s) with shape {[l.data.shape for l in image_layers]}")

    try:
        _, scale = get_data_tifffile(path)  # returns (Z, Y, X) in microns
        for layer in viewer.layers:
            if isinstance(layer, napari.layers.Image):
                layer.scale = scale
    except Exception as e:
        warnings.warn(f"Could not apply physical scale to napari layers: {e}")

    return viewer

def get_data_aicsimageio(path):
    img = AICSImage(path)
    volume = img.get_image_data("CZYX")  # shape: (C, Z, Y, X)
    scale = (
        float(img.physical_pixel_sizes.Z),
        float(img.physical_pixel_sizes.Y),
        float(img.physical_pixel_sizes.X)
    )
    return volume,scale

def get_data_aicsimageio_v2(path):
    if path.lower().endswith((".tif", ".tiff")):
        return get_data_tifffile(path)
    else:
        img = AICSImage(path)
        volume = img.get_image_data("CZYX")  # (C, Z, Y, X)
        scale = (
            float(img.physical_pixel_sizes.Z),
            float(img.physical_pixel_sizes.Y),
            float(img.physical_pixel_sizes.X)
        )
        return volume, scale


def get_data_tifffile(path):
    with tifffile.TiffFile(path) as tif:
        volume = tif.asarray()  # currently (Z, C, X, Y)
        volume = np.swapaxes(volume, 0, 1)  # -> (C, Z, X, Y)

        page = tif.pages[0]
        x_size = y_size = None
        if "XResolution" in page.tags:
            num, den = page.tags["XResolution"].value
            x_size = den / num
        if "YResolution" in page.tags:
            num, den = page.tags["YResolution"].value
            y_size = den / num

        z_size = None
        if tif.imagej_metadata:
            z_size = tif.imagej_metadata.get("spacing")

    scale = (z_size, y_size, x_size)
    return volume, scale

def correct_z_intensity(volume):

    original_dtype = volume.dtype
    num_channels   = volume.shape[0]
    corrected_czyx = volume.copy().astype(np.float32)  # copy so DAPI is preserved as-is

    for c in range(num_channels - 1):  # skip last channel (DAPI)
        channel = volume[c]
        slice_means      = np.mean(channel, axis=(1, 2))
        target_intensity = np.mean(slice_means)
        corrected_channel = np.zeros_like(channel, dtype=np.float32)

        for z in range(channel.shape[0]):
            if slice_means[z] > 0:
                correction_factor    = target_intensity / slice_means[z]
                corrected_channel[z] = channel[z] * correction_factor
            else:
                corrected_channel[z] = channel[z]

        corrected_czyx[c] = exposure.rescale_intensity(
            corrected_channel,
            out_range=(np.iinfo(original_dtype).min, np.iinfo(original_dtype).max)
            if np.issubdtype(original_dtype, np.integer)
            else (0.0, 1.0)
        )

    return corrected_czyx.astype(original_dtype)

def get_scale(volume_layer):

    scale_zyx = tuple(volume_layer.scale[:3])
    ratio_zy  = scale_zyx[0] / scale_zyx[1]
    print("Scale (Z,Y,X):",scale_zyx,"\nRatio Z/Y:",ratio_zy)

    return (scale_zyx,ratio_zy)

def padded_mask(binary_mask,pad_size=1):
    #This function adds one/mulitple black slices to the top and bottom of a volume. Usually used when using binary dilation and erosion to prevent z boundary touching edge retention and extension.
    assert binary_mask.ndim == 3, "Expected a 3D binary mask (Z, Y, X)"
    black_slice = np.zeros((pad_size, binary_mask.shape[1], binary_mask.shape[2]), dtype=binary_mask.dtype)
    binary_mask_padded=np.concatenate([black_slice,binary_mask,black_slice],axis=0)

    return binary_mask_padded

def disk_footprint(pixels):
    #Creates a 3d footprint of a 2d disk. For use with 3D operations.
    d = morphology.disk(pixels)
    footprint = d[np.newaxis, :, :]
    return footprint

def segment_nucleus(volume,scale_zyx,view_napari=False,sigma=4,pad_size=1,remove_edge_nuclei=True, otsu_multiplier=1):

    # Fixing the nuclei segmentation code.
    ratio_zy=scale_zyx[0]/scale_zyx[1]
    sigma_tuple=(sigma*(1/ratio_zy),sigma,sigma) #blurring constants
    volume_blur= filters.gaussian(volume, sigma=sigma_tuple, mode='nearest', preserve_range=True)

    otsu_thresh = filters.threshold_otsu(volume_blur)
    otsu_thresh=otsu_thresh*otsu_multiplier
    otsu_mask = volume_blur > otsu_thresh

    # Filling holes slice-by-slice
    mask_filled = np.zeros_like(otsu_mask)
    for z in range(otsu_mask.shape[0]):
        mask_filled[z] = ndimage.binary_fill_holes(otsu_mask[z])

    dilated_mask = morphology.binary_dilation(mask_filled, footprint=disk_footprint(2)) 

    slice_area=mask_filled.shape[1]*mask_filled.shape[2]

    if mask_filled[0,:,:].sum() >= 0.1*slice_area or mask_filled[-1,:,:].sum() >= 0.1*slice_area: # Check if a large part of the nucleus (10% of the slice area) cuts through the top and bottom slices
        dilated_mask_padded=padded_mask(dilated_mask,pad_size=pad_size)
        eroded_mask_padded = morphology.binary_erosion(dilated_mask_padded, footprint=disk_footprint(2))
        eroded_mask = eroded_mask_padded[pad_size:-pad_size, :, :] #removing the padded black slices.
    else:
        eroded_mask = morphology.binary_erosion(dilated_mask, footprint=disk_footprint(2))

    # Condition to set minimum volume of nucleus. rarely in the images there are more than 5 nuclei, in such cases se the volume threshold accordingly.
    _,num_features=ndimage.label(eroded_mask)
    if num_features<=5:
        small_volume=eroded_mask.sum()/5
    else:
        small_volume=eroded_mask.sum()/num_features

    final_mask = morphology.remove_small_objects(eroded_mask, min_size=small_volume)
    labels,num_features=ndimage.label(final_mask)

    print(num_features," Nuclei detected!")
    if remove_edge_nuclei==True:
        print("Removing XY edge-touching nuclei")
        # Find labels touching XY border ONLY (across ALL z slices)
        border_xy = np.zeros_like(labels, dtype=bool)
        border_xy[:, 0, :]  = True   # top row    (all Z)
        border_xy[:, -1, :] = True   # bottom row (all Z)
        border_xy[:, :, 0]  = True   # left col   (all Z)
        border_xy[:, :, -1] = True   # right col  (all Z)
        # Z faces are NOT included → labels touching only Z edges are kept 
        touching_xy = np.unique(labels[border_xy])
        touching_xy = touching_xy[touching_xy != 0]  # remove background
        labels[np.isin(labels, touching_xy)] = 0
    
    nuclei_labels=labels
    nuclei_mask=nuclei_labels > 0
    
    num_nuclei=len(np.unique(nuclei_labels))-1

    if remove_edge_nuclei == True:
        print(num_nuclei, " Nuclei remaining after removing edge-touching nuclei.") 
        
    if view_napari==True:
        viewer.add_labels(nuclei_labels, name="nuclei_labels",scale=scale_zyx)

    return (nuclei_labels, nuclei_mask, num_features)

def segment_nucleus_multiotsu(volume,scale_zyx,view_napari=False,sigma=4,pad_size=1,remove_edge_nuclei=True, otsu_multiplier=1):

    # Fixing the nuclei segmentation code.
    ratio_zy=scale_zyx[0]/scale_zyx[1]
    sigma_tuple=(sigma*(1/ratio_zy),sigma,sigma) #blurring constants
    volume_blur= filters.gaussian(volume, sigma=sigma_tuple, mode='nearest', preserve_range=True)

    otsu_thresh = filters.threshold_multiotsu(volume_blur)
    otsu_thresh=otsu_thresh[-1]
    otsu_thresh=otsu_thresh*otsu_multiplier
    otsu_mask = volume_blur > otsu_thresh

    # Filling holes slice-by-slice
    mask_filled = np.zeros_like(otsu_mask)
    for z in range(otsu_mask.shape[0]):
        mask_filled[z] = ndimage.binary_fill_holes(otsu_mask[z])

    dilated_mask = morphology.binary_dilation(mask_filled, footprint=disk_footprint(2)) 

    slice_area=mask_filled.shape[1]*mask_filled.shape[2]

    if mask_filled[0,:,:].sum() >= 0.1*slice_area or mask_filled[-1,:,:].sum() >= 0.1*slice_area: # Check if a large part of the nucleus (10% of the slice area) cuts through the top and bottom slices
        dilated_mask_padded=padded_mask(dilated_mask,pad_size=pad_size)
        eroded_mask_padded = morphology.binary_erosion(dilated_mask_padded, footprint=disk_footprint(2))
        eroded_mask = eroded_mask_padded[pad_size:-pad_size, :, :] #removing the padded black slices.
    else:
        eroded_mask = morphology.binary_erosion(dilated_mask, footprint=disk_footprint(2))

    # Condition to set minimum volume of nucleus. rarely in the images there are more than 5 nuclei, in such cases se the volume threshold accordingly.
    _,num_features=ndimage.label(eroded_mask)
    if num_features<=5:
        small_volume=eroded_mask.sum()/5
    else:
        small_volume=eroded_mask.sum()/num_features

    final_mask = morphology.remove_small_objects(eroded_mask, min_size=small_volume)
    labels,num_features=ndimage.label(final_mask)

    print(num_features," Nuclei detected!")
    if remove_edge_nuclei==True:
        print("Removing XY edge-touching nuclei")
        # Find labels touching XY border ONLY (across ALL z slices)
        border_xy = np.zeros_like(labels, dtype=bool)
        border_xy[:, 0, :]  = True   # top row    (all Z)
        border_xy[:, -1, :] = True   # bottom row (all Z)
        border_xy[:, :, 0]  = True   # left col   (all Z)
        border_xy[:, :, -1] = True   # right col  (all Z)
        # Z faces are NOT included → labels touching only Z edges are kept 
        touching_xy = np.unique(labels[border_xy])
        touching_xy = touching_xy[touching_xy != 0]  # remove background
        labels[np.isin(labels, touching_xy)] = 0
    
    nuclei_labels=labels
    nuclei_mask=nuclei_labels > 0
    
    num_nuclei=len(np.unique(nuclei_labels))-1

    if remove_edge_nuclei == True:
        print(num_nuclei, " Nuclei remaining after removing edge-touching nuclei.") 
        
    if view_napari==True:
        viewer.add_labels(nuclei_labels, name="nuclei_labels",scale=scale_zyx)

    return (nuclei_labels, nuclei_mask, num_features)

def identify_fishspots(volume,spot_name,nuclei_mask,spot_color,scale_zyx,view_napari=False,sigma=1,intensity_thresh_coeff=1.25):
    
    ratio_zy=scale_zyx[0]/scale_zyx[1]
    volume_nuclei_mask=nuclei_mask*volume
    _,num_nuclei=ndimage.label(nuclei_mask) # Only need number of nuclei and not the labels
    sigma_tuple=(sigma*(1/ratio_zy),sigma,sigma)
    #voxel_volume=np.prod(scale_zyx)
    voxel_volume=1
    volume_blur= filters.gaussian(volume_nuclei_mask, sigma=sigma_tuple, mode='nearest', preserve_range=True)

    pixels_inside_nucleus=volume_blur[nuclei_mask]

    triangle_thresh=filters.threshold_triangle(pixels_inside_nucleus)*intensity_thresh_coeff
    volume_mask=volume_blur>triangle_thresh

    volume_mask_clean=morphology.remove_small_objects(volume_mask,min_size=100)

    spot_labels, num_spots_unf = ndimage.label(volume_mask_clean)

    #Filtering for both mean intensity and spot volume if no. of spots is greater than twice the number of nuclei.
    if num_spots_unf > num_nuclei*2:
        print(f"Found {num_spots_unf} spots in {spot_name} before filtering")
        spot_prop = measure.regionprops(spot_labels, intensity_image=volume_blur)
        spot_intensities = np.array([spot.mean_intensity for spot in spot_prop])
        spot_volumes = np.array([spot.area * voxel_volume for spot in spot_prop])
        spot_max_intensities = np.array([spot.max_intensity for spot in spot_prop])
        intensity_thresh=filters.threshold_otsu(spot_max_intensities)
        #volume_thresh=filters.threshold_otsu(spot_volumes)
        volume_thresh=np.percentile(spot_volumes,0.15)
        
        filtered_spots = [spot for spot, intensity, vol in zip(spot_prop, spot_max_intensities, spot_volumes) if intensity >= intensity_thresh or vol >= volume_thresh]
        spot_ids = np.array([spot.label for spot in filtered_spots])
        filtered_spot_labels = np.where(np.isin(spot_labels, spot_ids), spot_labels, 0)
    else:
        filtered_spot_labels = spot_labels

    spot_prop = measure.regionprops(filtered_spot_labels)
    centroids = np.array([p.centroid for p in spot_prop])
    num_spots=len(centroids)

    if num_spots > 0:
        print(f"Found {num_spots} spots in {spot_name}")
    else:
        print(f"No spots in {spot_name}")
        centroids=None
    
    if view_napari and len(centroids) > 0:
        viewer.add_points(centroids, name=spot_name+"_spots", size=10, face_color=spot_color,scale=scale_zyx)

    return centroids,num_spots_unf

def identify_fishspots_only2(volume, spot_name, nuclei_mask, spot_color, scale_zyx, view_napari=False, sigma=1, intensity_thresh_coeff=1.25):

    ratio_zy = scale_zyx[0] / scale_zyx[1]
    nuclei_mask_dilated = morphology.binary_dilation(nuclei_mask, footprint=morphology.ball(1))
    volume_nuclei_mask = nuclei_mask_dilated * volume
    nuclei_labels, num_nuclei = ndimage.label(nuclei_mask)
    sigma_tuple = (sigma * (1 / ratio_zy), sigma, sigma)
    voxel_volume = 1
    volume_blur = filters.gaussian(volume_nuclei_mask, sigma=sigma_tuple, mode='nearest', preserve_range=True)

    pixels_inside_nucleus = volume_blur[nuclei_mask_dilated]
    triangle_thresh = filters.threshold_triangle(pixels_inside_nucleus) * intensity_thresh_coeff
    volume_mask = volume_blur > triangle_thresh
    volume_mask_clean = morphology.remove_small_objects(volume_mask, min_size=100)

    # Watershed
    spot_edt, spot_markers = perform_edt(volume_mask_clean, scale_zyx=scale_zyx, min_dist_um=0.2)
    spot_labels = segmentation.watershed(-spot_edt, markers=spot_markers, mask=volume_mask_clean)
    num_spots_unf = len(np.unique(spot_labels)) - 1

    # Per-nucleus filtering
    spot_prop = measure.regionprops(spot_labels, intensity_image=volume_blur)
    kept_spot_ids = []

    for nuc_id in np.unique(nuclei_labels):
        if nuc_id == 0:
            continue

        # Get nucleus mask (use dilated for spot assignment)
        nuc_mask = nuclei_labels == nuc_id

        # Find spots whose centroid falls inside this nucleus
        nuc_spots = [
            spot for spot in spot_prop
            if nuc_mask[int(round(spot.centroid[0])),
                        int(round(spot.centroid[1])),
                        int(round(spot.centroid[2]))]
        ]

        if len(nuc_spots) > 2:
            print(f"Nucleus {nuc_id}: {len(nuc_spots)} spots found — filtering")
            nuc_max_intensities = np.array([spot.max_intensity for spot in nuc_spots])
            nuc_volumes         = np.array([spot.area * voxel_volume for spot in nuc_spots])
            intensity_thresh    = np.sort(nuc_max_intensities)[-2]  # second highest
            volume_thresh       = filters.threshold_otsu(nuc_volumes)

            kept = [spot.label for spot, intensity in zip(nuc_spots, nuc_max_intensities)
                    if intensity >= intensity_thresh]
        else:
            kept = [spot.label for spot in nuc_spots]

        kept_spot_ids.extend(kept)

    # Build filtered labels array
    filtered_spot_labels = np.where(np.isin(spot_labels, kept_spot_ids), spot_labels, 0)

    spot_prop    = measure.regionprops(filtered_spot_labels)
    centroids    = np.array([p.centroid for p in spot_prop])
    num_spots    = len(centroids)

    if num_spots > 0:
        print(f"Found {num_spots} spots in {spot_name}")
    else:
        print(f"No spots in {spot_name}")
        centroids = None

    if view_napari and centroids is not None and len(centroids) > 0:
        viewer.add_points(centroids, name=spot_name + "_spots", size=10, face_color=spot_color, scale=scale_zyx)

    return centroids, num_spots_unf

def identify_fishspots_soph(volume, spot_name, nuclei_mask, spot_color, scale_zyx, view_napari=False, sigma=1, intensity_thresh_coeff=1.25, min_dist_um=0.2):

    ratio_zy = scale_zyx[0] / scale_zyx[1]
    nuclei_mask_dilated = morphology.binary_dilation(nuclei_mask, footprint=morphology.ball(1))
    volume_nuclei_mask = nuclei_mask_dilated * volume
    nuclei_labels, num_nuclei = ndimage.label(nuclei_mask)
    sigma_tuple = (sigma * (1 / ratio_zy), sigma, sigma)
    voxel_volume = 1
    volume_blur = filters.gaussian(volume_nuclei_mask, sigma=sigma_tuple, mode='nearest', preserve_range=True)

    pixels_inside_nucleus = volume_blur[nuclei_mask_dilated]
    triangle_thresh = filters.threshold_triangle(pixels_inside_nucleus) * intensity_thresh_coeff
    volume_mask = volume_blur > triangle_thresh
    volume_mask_clean = morphology.remove_small_objects(volume_mask, min_size=100)

    # Basic thresholding labels — used for nuclei with > 2 spots
    basic_spot_labels, _ = ndimage.label(volume_mask_clean)
    basic_spot_prop = measure.regionprops(basic_spot_labels, intensity_image=volume_blur)

    kept_spot_ids_basic     = []  # spot IDs from basic_spot_labels
    kept_centroids_watershed = []  # centroids from per-nucleus watershed (separate label space)

    def filter_spots(nuc_spots, spot_source):
        """Filter spots by mean+max intensity and volume using Otsu thresholds."""
        nuc_max_intensities  = np.array([s.max_intensity  for s in nuc_spots])
        nuc_mean_intensities = np.array([s.mean_intensity for s in nuc_spots])
        nuc_volumes          = np.array([s.area * voxel_volume for s in nuc_spots])
        print(f"  Max intensities:  {nuc_max_intensities}")
        print(f"  Mean intensities: {nuc_mean_intensities}")
        print(f"  Volumes:          {nuc_volumes}")

        max_thresh  = np.ceil(filters.threshold_otsu(nuc_max_intensities))
        mean_thresh = np.ceil(filters.threshold_otsu(nuc_mean_intensities))
        vol_thresh  = np.ceil(filters.threshold_otsu(nuc_volumes))

        kept = [
            s.label for s, mean_i, max_i, vol
            in zip(nuc_spots, nuc_mean_intensities, nuc_max_intensities, nuc_volumes)
            if (mean_i > mean_thresh and max_i > max_thresh) or vol > vol_thresh
            #if max_i >= max_thresh or vol >= vol_thresh
        ]
        return kept
    spots_unf=[]
    # Iterate through each nucleus
    for nuc_id in np.unique(nuclei_labels):
        if nuc_id == 0:
            continue

        single_nucleus_mask = nuclei_labels == nuc_id
        
        # Find basic spots whose centroid falls inside this nucleus
        nuc_spots_basic = [
            spot for spot in basic_spot_prop
            if single_nucleus_mask[
                int(round(spot.centroid[0])),
                int(round(spot.centroid[1])),
                int(round(spot.centroid[2]))
            ]
        ]

        if len(nuc_spots_basic) > 2:
            # --- Path 1: Basic thresholding, filter if > 2 spots ---
            print(f"Nucleus {nuc_id}: {len(nuc_spots_basic)} {spot_name} spots (basic) — filtering")
            kept = filter_spots(nuc_spots_basic, 'basic')
            kept_spot_ids_basic.extend(kept)

        elif len(nuc_spots_basic) == 2:
            print(f"Nucleus {nuc_id}: exactly 2 {spot_name} spots (basic) — keeping both")
            kept_spot_ids_basic.extend([s.label for s in nuc_spots_basic])

        else:
            # --- Path 2: Watershed on this nucleus ---
            print(f"Nucleus {nuc_id}: {len(nuc_spots_basic)} {spot_name} spots (basic) — running Watershed")
            spot_volume_nuc = volume_mask_clean * single_nucleus_mask

            if spot_volume_nuc.sum() == 0:
                print(f"  Nucleus {nuc_id}: no foreground voxels — skipping")
                continue

            spot_edt, spot_markers = perform_edt(spot_volume_nuc, scale_zyx=scale_zyx, min_dist_um=min_dist_um)
            ws_labels = segmentation.watershed(-spot_edt, markers=spot_markers, mask=spot_volume_nuc)
            ws_spot_prop = measure.regionprops(ws_labels, intensity_image=volume_blur)
            num_ws_spots = len(ws_spot_prop)

            if num_ws_spots > 2:
                # Filter watershed spots
                print(f"  Nucleus {nuc_id}: {num_ws_spots} {spot_name} spots (watershed) — filtering")
                kept_ws = filter_spots(ws_spot_prop, 'watershed')
                ws_centroids = [
                    s.centroid for s in ws_spot_prop if s.label in kept_ws
                ]
            else:
                # Keep all watershed spots
                print(f"  Nucleus {nuc_id}: {num_ws_spots} {spot_name} spots (watershed) — keeping all")
                ws_centroids = [s.centroid for s in ws_spot_prop]

            kept_centroids_watershed.extend(ws_centroids)

        spots_unf.extend(nuc_spots_basic)
        
    num_spots_unf=len(spots_unf)

    # Build filtered basic labels
    filtered_basic_labels = np.where(np.isin(basic_spot_labels, kept_spot_ids_basic), basic_spot_labels, 0)
    basic_prop = measure.regionprops(filtered_basic_labels)
    basic_centroids = [p.centroid for p in basic_prop]

    # Combine centroids from both paths
    all_centroids = basic_centroids + kept_centroids_watershed
    num_spots = len(all_centroids)

    if num_spots > 0:
        centroids = np.array(all_centroids)
        print(f"Found {num_spots} total {spot_name} spots\n")
    else:
        centroids = np.empty((0, 3))  # empty array with correct shape
        num_spots_unf = 0
        print(f"No spots found in {spot_name}")

    if view_napari and len(centroids) > 0:
        viewer.add_points(centroids, name=spot_name + "_spots", size=10, face_color=spot_color, scale=scale_zyx)

    return centroids, num_spots_unf

def update_spots():
    for layer in viewer.layers:
        if re.search('spot',layer.name):
            layer=layer
    return layer

def perform_edt(binary_mask,scale_zyx,pad_size=1,min_dist_um = 0.2):
    # Padding the z-stack with black slice at top and bottom to prevent EDT aberrations.
    binary_mask_padded=padded_mask(binary_mask,pad_size)

    edt_padded = ndimage.distance_transform_edt(binary_mask_padded,sampling=scale_zyx)
    min_distance_vox = int(min_dist_um / scale_zyx[1])
    max_coords = feature.peak_local_max(edt_padded, min_distance=min_distance_vox, labels=binary_mask_padded, threshold_abs=0.1) #Skips sorting background near-zero values
    edt = edt_padded[pad_size:-pad_size, :, :] #removing the padded black slices.

    # Creating a new image with same shape as edt, but filled with zeroes. Next, we add the maxima coordinates to it as True/ones
    seeds = np.zeros(edt.shape, dtype=bool)
    max_coords_unpadded = max_coords.copy()
    max_coords_unpadded[:, 0] -= pad_size  # correct Z offset
    seeds[tuple(max_coords_unpadded.T)] = True
    max_markers, _ = ndimage.label(seeds)

    return (edt,max_markers)

def segment_speckles(volume,nuclei_mask,scale_zyx,view_napari=False,sigma=1,min_size_um3 = 0.002):
    ratio_zy=scale_zyx[0]/scale_zyx[1]
    voxel_volume=np.prod(scale_zyx)
    volume_nuclei_mask=nuclei_mask*volume #volume is the speckle channel
    sigma_tuple=(sigma*(1/ratio_zy),sigma,sigma)
    volume_nuclei_mask = volume_nuclei_mask.astype(np.float32)
    volume_blur=filters.gaussian(volume_nuclei_mask, sigma=sigma_tuple, mode='nearest', preserve_range=True)
    triangle_thresh=filters.threshold_triangle(volume_blur[nuclei_mask]) #threshold only on pixels inside the nucleus
    speckles_mask_thresh=volume_blur>triangle_thresh

    min_size_vox = max(1, int(min_size_um3 / voxel_volume))
    #min_size_vox = int(min_size_um3 / voxel_volume)
    smallobject_speckles_mask=morphology.remove_small_objects(speckles_mask_thresh,min_size=min_size_vox)
    speckles_mask = morphology.binary_closing(smallobject_speckles_mask, footprint=disk_footprint(1))

    speckles_edt, markers = perform_edt(speckles_mask,scale_zyx=scale_zyx)
    speckles_labels = segmentation.watershed(-speckles_edt, markers=markers, mask=speckles_mask)

    if view_napari==True:
        viewer.add_labels(speckles_labels,name="speckles_labels",scale=scale_zyx)
    print("Speckles segmented")
    return (speckles_labels,speckles_mask)

def segment_speckles_mulitotsu(volume,nuclei_mask,scale_zyx,view_napari=False,sigma=1,min_size_um3 = 0.002, otsu_multiplier=1):
    ratio_zy=scale_zyx[0]/scale_zyx[1]
    voxel_volume=np.prod(scale_zyx)
    volume_nuclei_mask=nuclei_mask*volume
    sigma_tuple=(sigma*(1/ratio_zy),sigma,sigma)
    volume_nuclei_mask = volume_nuclei_mask.astype(np.float32)
    volume_blur=filters.gaussian(volume_nuclei_mask, sigma=sigma_tuple, mode='nearest', preserve_range=True)
    otsu_thresh=filters.threshold_multiotsu(volume_blur[nuclei_mask]) #threshold only on pixels inside the nucleus
    otsu_threshold_mult=otsu_thresh[1]*otsu_multiplier
    speckles_mask_thresh=volume_blur>otsu_threshold_mult

    min_size_vox = max(1, int(min_size_um3 / voxel_volume))
    #min_size_vox = int(min_size_um3 / voxel_volume)
    smallobject_speckles_mask=morphology.remove_small_objects(speckles_mask_thresh,min_size=min_size_vox)
    speckles_mask = morphology.binary_closing(smallobject_speckles_mask, footprint=disk_footprint(1))

    speckles_edt, markers = perform_edt(speckles_mask,scale_zyx=scale_zyx)
    speckles_labels = segmentation.watershed(-speckles_edt, markers=markers, mask=speckles_mask)

    if view_napari==True:
        viewer.add_labels(speckles_labels,name="speckles_labels",scale=scale_zyx)
    print("Speckles segmented")
    return (speckles_labels,speckles_mask)

def mask_chromatin(volume,scale_zyx,nuclei_mask,view_napari=False,sigma=1,min_hole_um3 = 1, otsu_multiplier=1):
    ratio_zy=scale_zyx[0]/scale_zyx[1]
    sigma_tuple=(sigma*(1/ratio_zy),sigma,sigma)
    volume_blur= filters.gaussian(volume, sigma=sigma_tuple, mode='nearest', preserve_range=True)
    otsu_thresh = filters.threshold_otsu(volume_blur[nuclei_mask])
    otsu_mask = volume_blur > (otsu_thresh*otsu_multiplier)
    voxel_volume = np.prod(scale_zyx)
    area_threshold = int(min_hole_um3 / voxel_volume)
    chromatin_mask = morphology.remove_small_holes(otsu_mask, area_threshold=area_threshold)

    if view_napari==True:
        viewer.add_image(chromatin_mask,scale=scale_zyx)
    print("Chromatin masked")
    return chromatin_mask

def calculate_euclidean_distance_from_speckles(speckle_mask,nuclei_mask,chromatin_mask,scale_zyx,view_napari=False):
    # Invert speckles only inside nuclei
    speckle_mask_inv = np.where(nuclei_mask, ~speckle_mask, speckle_mask)

    speckles_inv_edt=perform_edt(speckle_mask_inv,scale_zyx=scale_zyx)[0]

    distance_from_speckles_edt=speckles_inv_edt*chromatin_mask  # Only keeping distance values where there is chromatin

    if view_napari==True:
        viewer.add_image(distance_from_speckles_edt,scale=scale_zyx,name="Distances_from_speckles_edt")
    print("EDT for speckles calculated")
    return distance_from_speckles_edt  

def summary_table_old(centroids1,nuclei_labels,speckles_labels,distance_from_lamina_edt,distance_from_speckles_edt,scale_zyx,path,centroids2=None,centroids1_name="gene1",centroids2_name="gene2"):
    #dilate the nuclei mask to incorporate spots detected at the edge
    nuclei_labels = morphology.dilation(nuclei_labels, footprint=morphology.ball(1))
    #Calculating volume of each nucleus
    voxel_volume=np.prod(scale_zyx)
    nuclei_properties=measure.regionprops(nuclei_labels)
    nuclei_ids=np.unique(nuclei_labels)
    nuclei_ids=nuclei_ids[nuclei_ids != 0]
    nuclei_volumes = {nuclei.label: nuclei.area * voxel_volume for nuclei in nuclei_properties}
    speckle_median_distances=ndimage.median(distance_from_speckles_edt, labels=nuclei_labels, index=nuclei_ids)
    lamina_median_distances=ndimage.median(distance_from_lamina_edt, labels=nuclei_labels, index=nuclei_ids)

    #Calculating speckles volume
    speckles_mask = speckles_labels > 0
    nucleus_ids_at_speckles = nuclei_labels[speckles_mask]
    # Count speckle voxels per nucleus using bincount
    max_nucleus = nuclei_labels.max()
    speckles_volume = np.bincount(nucleus_ids_at_speckles, minlength=max_nucleus + 1) # (nucleus ID is used as the bin index)

    # Distance from lamina and speckles for centroid1:
    indices1 = np.round(centroids1).astype(int)
    lamina_distances1 = distance_from_lamina_edt[indices1[:, 0], indices1[:, 1], indices1[:, 2]]
    speckle_distances1 = distance_from_speckles_edt[indices1[:, 0], indices1[:, 1], indices1[:, 2]]

    #Volume statistics
    nucleus_id1 = nuclei_labels[indices1[:, 0], indices1[:, 1], indices1[:, 2]] # finding the parent nuclei of spots
    spot_nuclei_volume1 = [nuclei_volumes[nid] for nid in nucleus_id1]
    speckle_volume1 = [speckles_volume[nid] * voxel_volume for nid in nucleus_id1 if speckles_volume[nid] > 0]

    # Distance from lamina and speckles for centroid2:
    if centroids2 is not None:
        indices2 = np.round(centroids2).astype(int)
        lamina_distances2 = distance_from_lamina_edt[indices2[:, 0], indices2[:, 1], indices2[:, 2]]
        speckle_distances2 = distance_from_speckles_edt[indices2[:, 0], indices2[:, 1], indices2[:, 2]]
        nucleus_id2 = nuclei_labels[indices2[:, 0], indices2[:, 1], indices2[:, 2]] # finding the parent nuclei of spots
        spot_nuclei_volume2 = [nuclei_volumes[nid] for nid in nucleus_id2]
        speckle_volume2 = [speckles_volume[nid] * voxel_volume for nid in nucleus_id2 if speckles_volume[nid] > 0]

    # Padding the control cells of the output df
    nuclei_volume_pad = [nuclei_volumes[nid] for nid in nuclei_ids]
    speckle_volume_pad = [speckles_volume[nid] * voxel_volume for nid in nuclei_ids if speckles_volume[nid] > 0]
    # creating list for naming the gene spots
    centroid1_list = [f"{centroids1_name}({i})" for i in range(1, len(centroids1) + 1)]
    centroid2_list = [f"{centroids2_name}({i})" for i in range(1, len(centroids2) + 1)] if centroids2 is not None else []
    control_list = [f"zz_Nucleus{i}" for i in range(1, len(nuclei_ids) + 1)]

    # Use empty arrays as fallback is centroids 2 doesn't exist.
    speckle_distances2   = speckle_distances2   if centroids2 is not None else np.array([])
    lamina_distances2    = lamina_distances2    if centroids2 is not None else np.array([])
    nucleus_id2          = nucleus_id2          if centroids2 is not None else np.array([])
    spot_nuclei_volume2  = spot_nuclei_volume2  if centroids2 is not None else []
    speckle_volume2      = speckle_volume2      if centroids2 is not None else []

    spot_ids=np.concatenate([centroid1_list,centroid2_list,control_list])
    speckles_distances=np.concatenate([speckle_distances1,speckle_distances2,speckle_median_distances])
    lamina_distances=np.concatenate([lamina_distances1,lamina_distances2,lamina_median_distances])
    nuclear_ids=np.concatenate([nucleus_id1,nucleus_id2,nuclei_ids])
    spot_nuclei_volume=np.concatenate([spot_nuclei_volume1,spot_nuclei_volume2,nuclei_volume_pad])
    spot_speckle_volume=np.concatenate([speckle_volume1,speckle_volume2,speckle_volume_pad])

    c1,c2,c3,cell_stage,condition,file=parse_image_properties(path)

    condition_type=len(spot_ids)*[condition]
    filename=len(spot_ids)*[file]
    stage=len(spot_ids)*[cell_stage]

    summary_df=pd.DataFrame({'spot_ids':spot_ids,'distance_from_nearest_speckle_um':speckles_distances,'distance_from_lamina_um':lamina_distances,'nucleus_id':nuclear_ids,'nucleus_volume_um3':spot_nuclei_volume,'total_speckle_volume_um3':spot_speckle_volume,'condition':condition_type,'stage':stage,'file':filename}).sort_values(by=['nucleus_id','spot_ids']).reset_index(drop=True)
    print("Results summarised")
    return summary_df

def summary_table(centroids1,nuclei_labels,speckles_labels,distance_from_lamina_edt,distance_from_speckles_edt,scale_zyx,path,centroids2=None,centroids1_name="gene1",centroids2_name="gene2"):
    # Dilate the nuclei mask to incorporate spots detected at the edge
    nuclei_labels = morphology.dilation(nuclei_labels, footprint=morphology.ball(1))

    # Calculating speckles volume
    speckles_mask = speckles_labels > 0
    nucleus_ids_at_speckles = nuclei_labels[speckles_mask]
    # Count speckle voxels per nucleus using bincount
    max_nucleus = nuclei_labels.max()
    speckles_volume = np.bincount(nucleus_ids_at_speckles, minlength=max_nucleus + 1) # (nucleus ID is used as the bin index)

    # Calculating volume of each nucleus
    voxel_volume=np.prod(scale_zyx)
    nuclei_properties=measure.regionprops(nuclei_labels)
    nuclei_ids=np.unique(nuclei_labels)
    nuclei_ids = np.array([nid for nid in nuclei_ids if nid != 0 and speckles_volume[nid] > 0])
    nuclei_volumes = {nuclei.label: nuclei.area * voxel_volume for nuclei in nuclei_properties}
    speckle_median_distances=ndimage.median(distance_from_speckles_edt, labels=nuclei_labels, index=nuclei_ids)
    lamina_median_distances=ndimage.median(distance_from_lamina_edt, labels=nuclei_labels, index=nuclei_ids)

    # Distance from lamina and speckles for centroid1:
    indices1 = np.round(centroids1).astype(int)
    lamina_distances1 = distance_from_lamina_edt[indices1[:, 0], indices1[:, 1], indices1[:, 2]]
    speckle_distances1 = distance_from_speckles_edt[indices1[:, 0], indices1[:, 1], indices1[:, 2]]

    # Volume statistics
    nucleus_id1 = nuclei_labels[indices1[:, 0], indices1[:, 1], indices1[:, 2]] # finding the parent nuclei of spots
    spot_nuclei_volume1 = [nuclei_volumes[nid] for nid in nucleus_id1]
    speckle_volume1 = [speckles_volume[nid] * voxel_volume for nid in nucleus_id1]

    # Distance from lamina and speckles for centroid2:
    if centroids2 is not None:
        indices2 = np.round(centroids2).astype(int)
        lamina_distances2 = distance_from_lamina_edt[indices2[:, 0], indices2[:, 1], indices2[:, 2]]
        speckle_distances2 = distance_from_speckles_edt[indices2[:, 0], indices2[:, 1], indices2[:, 2]]
        nucleus_id2 = nuclei_labels[indices2[:, 0], indices2[:, 1], indices2[:, 2]] # finding the parent nuclei of spots
        spot_nuclei_volume2 = [nuclei_volumes[nid] for nid in nucleus_id2]
        speckle_volume2 = [speckles_volume[nid] * voxel_volume for nid in nucleus_id2]

    # Padding the control cells of the output df
    nuclei_volume_pad = [nuclei_volumes[nid] for nid in nuclei_ids]
    speckle_volume_pad = [speckles_volume[nid] * voxel_volume for nid in nuclei_ids if speckles_volume[nid] > 0]
    # Creating list for naming the gene spots
    centroid1_list = [f"{centroids1_name}({i})" for i in range(1, len(centroids1) + 1)]
    centroid2_list = [f"{centroids2_name}({i})" for i in range(1, len(centroids2) + 1)] if centroids2 is not None else []
    control_list = [f"zz_Nucleus{i}" for i in range(1, len(nuclei_ids) + 1)]

    # Use empty arrays as fallback is centroids 2 doesn't exist.
    speckle_distances2   = speckle_distances2   if centroids2 is not None else np.array([])
    lamina_distances2    = lamina_distances2    if centroids2 is not None else np.array([])
    nucleus_id2          = nucleus_id2          if centroids2 is not None else np.array([])
    spot_nuclei_volume2  = spot_nuclei_volume2  if centroids2 is not None else []
    speckle_volume2      = speckle_volume2      if centroids2 is not None else []

    spot_ids=np.concatenate([centroid1_list,centroid2_list,control_list])
    speckles_distances=np.concatenate([speckle_distances1,speckle_distances2,speckle_median_distances])
    lamina_distances=np.concatenate([lamina_distances1,lamina_distances2,lamina_median_distances])
    nuclear_ids=np.concatenate([nucleus_id1,nucleus_id2,nuclei_ids])
    spot_nuclei_volume=np.concatenate([spot_nuclei_volume1,spot_nuclei_volume2,nuclei_volume_pad])
    spot_speckle_volume=np.concatenate([speckle_volume1,speckle_volume2,speckle_volume_pad])

    c1,c2,c3,cell_stage,condition,file=parse_image_properties(path)

    condition_type=len(spot_ids)*[condition]
    filename=len(spot_ids)*[file]
    stage=len(spot_ids)*[cell_stage]

    summary_df=pd.DataFrame({'spot_ids':spot_ids,'distance_from_nearest_speckle_um':speckles_distances,'distance_from_lamina_um':lamina_distances,'nucleus_id':nuclear_ids,'nucleus_volume_um3':spot_nuclei_volume,'total_speckle_volume_um3':spot_speckle_volume,'condition':condition_type,'stage':stage,'file':filename}).sort_values(by=['nucleus_id','spot_ids']).reset_index(drop=True)
    print("Results summarised")
    return summary_df