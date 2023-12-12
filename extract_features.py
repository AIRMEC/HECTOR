import argparse
import os
import time
import numpy as np

import openslide
import cv2
from PIL import Image, ImageDraw
from shapely.affinity import scale
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from collections import defaultdict

import nmslib
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity

# Optional if stain deconvolution is used.
import histomicstk as htk #pip install histomicstk --find-links https://girder.github.io/large_image_wheels

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from esvit.utils import bool_flag

# You can use your own encoder to extract features. Here's examples including EsVIT the one used in the publication. 
from encoders import load_encoder_esVIT, load_encoder_resnet

def get_args_parser():
    parser = argparse.ArgumentParser('Preprocessing script esvit', add_help=False)
    parser.add_argument(
        "--input_slide",
        type=str,
        help="Path to input WSI file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory to save output data",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Feature extractor weights checkpoint",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--tile_size",
        help="Desired tile size in microns (should be the same value as used in feature extraction model).",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--out_size",
        help="Resize the square tile to this output size (in pixels).",
        type=int,
        default=224,
    )
    parser.add_argument(
        "--method",
        help="Segmentation method, otsu or stain deconv",
        type=str,
        default='otsu',
    )
    parser.add_argument(
        "--dist_threshold",
        type=int,
        default=4,
        help="L2 norm distance when spatially merging pacthes.",
    )
    parser.add_argument(
        "--corr_threshold",
        type=float,
        default=0.6,
        help="Cosine similarity distance when semantically merging pacthes.",
    )
    parser.add_argument(
        "--workers",
        help="The number of workers to use for the data loader. Only relevant when using a GPU.",
        type=int,
        default=4,
    )
    parser.add_argument(
        '--cfg',
        help='experiment configure file name. See EsVIT repo.',
        type=str
    )
    parser.add_argument(
        '--arch', default='deit_small', type=str,
        choices=['cvt_tiny', 'swin_tiny','swin_small', 'swin_base', 'swin_large', 'swin', 'vil', 'vil_1281', 'vil_2262', 'deit_tiny', 'deit_small', 'vit_base'],
        help="""Name of architecture to train. For quick experiments with ViTs, we recommend using deit_tiny or deit_small. See EsVIT repo."""
    )
    parser.add_argument(
        '--n_last_blocks', 
        default=4, 
        type=int, 
        help="""Concatenate [CLS] tokens for the `n` last blocks. We use `n=4` when evaluating DeiT-Small and `n=1` with ViT-Base. See EsVIT repo."""
    )
    parser.add_argument(
        '--avgpool_patchtokens', 
        default=False, 
        type=bool_flag,
        help="""Whether ot not to concatenate the global average pooled features to the [CLS] token.
        We typically set this to False for DeiT-Small and to True with ViT-Base. See EsVIT repo."""
    )
    parser.add_argument(
        '--patch_size', 
        default=8, 
        type=int, 
        help='Patch resolution of the model. See EsVIT repo.'
    )
    parser.add_argument(
        'opts',
        help="Modify config options using the command-line. See EsVIT repo.",
        default=None,
        nargs=argparse.REMAINDER
    )
    parser.add_argument(
        "--rank", 
        default=0, 
        type=int, 
        help="Please ignore and do not set this argument.")

    return parser

def segment_tissue(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mthresh = 7
    img_med = cv2.medianBlur(img_hsv[:, :, 1], mthresh)
    _, img_prepped = cv2.threshold(img_med, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)

    close = 4
    kernel = np.ones((close, close), np.uint8)
    img_prepped = cv2.morphologyEx(img_prepped, cv2.MORPH_CLOSE, kernel)

    # Find and filter contours
    contours, hierarchy = cv2.findContours(
        img_prepped, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
    )
    return contours, hierarchy

def segment_tissue_deconv_stain(img):
    """
    Method 2: Tissue segmentation using stain deconvolution. Alternative to Otsu thresholding. 
    """
    image = img.copy()

    image[image[...,-1]==0] = [255,255,255,0]

    image = Image.fromarray(image)
    image = np.asarray(image.convert('RGB'))
    
    I_0 = 255
    
    # Create stain to color map
    stain_color_map = htk.preprocessing.color_deconvolution.stain_color_map

    # Specify stains of input image
    stains = ['hematoxylin',  # nuclei stain
              'eosin']        # cytoplasm stain
    
    w_est = htk.preprocessing.color_deconvolution.rgb_separate_stains_macenko_pca(image, I_0)
    deconv_result = htk.preprocessing.color_deconvolution.color_deconvolution(image, w_est, I_0)
    
    final_mask = np.zeros(image.shape[0:2], np.uint8)

    for i in 0, 1: 
        channel = htk.preprocessing.color_deconvolution.find_stain_index(
            stain_color_map[stains[i]], w_est)

        img_for_thresholding = 255 - deconv_result.Stains[:, :, channel]
        _, img_prepped = cv2.threshold(
            img_for_thresholding, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY)

        final_mask = cv2.bitwise_or(final_mask, img_prepped)
        
    for i in range(5):
        close = 3
        kernel = np.ones((close, close), np.uint8)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
    
    return final_mask

def mask_to_polygons(mask, min_area, min_area_holes=10., epsilon=10.):
    """Convert a mask ndarray (binarized image) to Multipolygons"""
    # first, find contours with cv2: it's much faster than shapely
    contours, hierarchy = cv2.findContours(mask,
                                  cv2.RETR_CCOMP,
                                  cv2.CHAIN_APPROX_NONE)
    if not contours:
        return MultiPolygon()

    cnt_children = defaultdict(list)
    child_contours = set()
    assert hierarchy.shape[0] == 1
    
    # http://docs.opencv.org/3.1.0/d9/d8b/tutorial_py_contours_hierarchy.html
    for idx, (_, _, _, parent_idx) in enumerate(hierarchy[0]):
        if parent_idx != -1:
            child_contours.add(idx)
            cnt_children[parent_idx].append(contours[idx])
            
    # create actual polygons filtering by area (removes artifacts)
    all_polygons = []

    for idx, cnt in enumerate(contours):

        if idx not in child_contours and cv2.contourArea(cnt) >= min_area:
            assert cnt.shape[1] == 1
            poly = Polygon(
                shell=cnt[:, 0, :],
                holes=[c[:, 0, :] for c in cnt_children.get(idx, [])
                       if cv2.contourArea(c) >= min_area_holes])
            
            if not poly.is_valid:
                # This is likely becausee the polygon is self-touching or self-crossing.
                # Try and 'correct' the polygon using the zero-length buffer() trick.
                # See https://shapely.readthedocs.io/en/stable/manual.html#object.buffer
                poly = poly.buffer(0)
    
            all_polygons.append(poly)

    if len(all_polygons) == 0:
        raise Exception("Raw tissue mask consists of 0 polygons")

    # if this raises an issue - try instead unary_union(all_polygons)        
    all_polygons = MultiPolygon(all_polygons)

    return all_polygons

def detect_foreground(contours, hierarchy):
    hierarchy = np.squeeze(hierarchy, axis=(0,))[:, 2:]

    # find foreground contours (parent == -1)
    hierarchy_1 = np.flatnonzero(hierarchy[:, 1] == -1)
    foreground_contours = [contours[cont_idx] for cont_idx in hierarchy_1]

    all_holes = []
    for cont_idx in hierarchy_1:
        all_holes.append(np.flatnonzero(hierarchy[:, 1] == cont_idx))

    hole_contours = []
    for hole_ids in all_holes:
        holes = [contours[idx] for idx in hole_ids]
        hole_contours.append(holes)

    return foreground_contours, hole_contours

def construct_polygon(foreground_contours, hole_contours, min_area):
    polys = []
    for foreground, holes in zip(foreground_contours, hole_contours):
        # We remove all contours that consist of fewer than 3 points, as these won't work with the Polygon constructor.
        if len(foreground) < 3:
            continue

        # remove redundant dimensions from the contour and convert to Shapely Polygon
        poly = Polygon(np.squeeze(foreground))

        # discard all polygons that are considered too small
        if poly.area < min_area:
            continue

        if not poly.is_valid:
            # This is likely becausee the polygon is self-touching or self-crossing.
            # Try and 'correct' the polygon using the zero-length buffer() trick.
            # See https://shapely.readthedocs.io/en/stable/manual.html#object.buffer
            poly = poly.buffer(0)

        # Punch the holes in the polygon
        for hole_contour in holes:
            if len(hole_contour) < 3:
                continue

            hole = Polygon(np.squeeze(hole_contour))

            if not hole.is_valid:
                continue

            # ignore all very small holes
            if hole.area < min_area:
                continue

            poly = poly.difference(hole)

        polys.append(poly)

    if len(polys) == 0:
        raise Exception("Raw tissue mask consists of 0 polygons")

    # If we have multiple polygons, we merge any overlap between them using unary_union().
    # This will result in a Polygon or MultiPolygon with most tissue masks.
    return unary_union(polys)

def generate_tiles(tile_width_pix, tile_height_pix, img_width, img_height, offsets=[(0, 0)]):
    # Generate tiles covering the entire image.
    # Provide an offset (x,y) to create a stride-like overlap effect.
    # Add an additional tile size to the range stop to prevent tiles being cut off at the edges.
    range_stop_width = int(np.ceil(img_width + tile_width_pix))
    range_stop_height = int(np.ceil(img_height + tile_height_pix))

    rects = []
    for xmin, ymin in offsets:
        cols = range(int(np.floor(xmin)), range_stop_width, tile_width_pix)
        rows = range(int(np.floor(ymin)), range_stop_height, tile_height_pix)
        for x in cols:
            for y in rows:
                rect = Polygon(
                    [
                        (x, y),
                        (x + tile_width_pix, y),
                        (x + tile_width_pix, y - tile_height_pix),
                        (x, y - tile_height_pix),
                    ]
                )
                rects.append(rect)
    return rects

def make_tile_QC_fig(tiles, slide, level, line_width_pix=1, extra_tiles=None):
    # Render the tiles on an image derived from the specified zoom level
    img = slide.read_region((0, 0), level, slide.level_dimensions[level])
    downsample = 1 / slide.level_downsamples[level]

    draw = ImageDraw.Draw(img, "RGBA")
    for tile in tiles:
        bbox = tuple(np.array(tile.bounds) * downsample)
        draw.rectangle(bbox, outline="lightgreen", width=line_width_pix)

    # allow to display other tiles, such as excluded or sampled
    if extra_tiles:
        for tile in extra_tiles:
            bbox = tuple(np.array(tile.bounds) * downsample)
            draw.rectangle(bbox, outline="blue", width=line_width_pix + 1)

    return img

def create_tissue_mask(wsi, seg_level, method='otsu'):
    # Determine the best level to determine the segmentation on
    level_dims = wsi.level_dimensions[seg_level]

    img = np.array(wsi.read_region((0, 0), seg_level, level_dims))

    # Get the total surface area of the slide level that was used
    level_area = level_dims[0] * level_dims[1]

    # Minimum surface area of tissue polygons (in pixels)
    # Note that this value should be sensible in the context of the chosen tile size
    min_area = level_area / 500

    if method=='stain_deconv':
        tissue_mask = segment_tissue_deconv_stain(img)
        tissue_mask = mask_to_polygons(tissue_mask, min_area)
    else:
        contours, hierarchy = segment_tissue(img)
        foreground_contours, hole_contours = detect_foreground(contours, hierarchy)
        tissue_mask = construct_polygon(foreground_contours, hole_contours, min_area)

    # Scale the tissue mask polygon to be in the coordinate space of the slide's level 0
    scale_factor = wsi.level_downsamples[seg_level]
    tissue_mask_scaled = scale(
        tissue_mask, xfact=scale_factor, yfact=scale_factor, zfact=1.0, origin=(0, 0)
    )

    return tissue_mask_scaled

def create_tissue_tiles(wsi, tissue_mask_scaled, tile_size_microns, offsets_micron=None):

    print(f"tile size is {tile_size_microns} um")

    # Compute the tile size in pixels from the desired tile size in microns and the image resolution
    assert (
        openslide.PROPERTY_NAME_MPP_X in wsi.properties
    ), "microns per pixel along X-dimension not available"
    assert (
        openslide.PROPERTY_NAME_MPP_Y in wsi.properties
    ), "microns per pixel along Y-dimension not available"

    mpp_x = float(wsi.properties[openslide.PROPERTY_NAME_MPP_X])
    mpp_y = float(wsi.properties[openslide.PROPERTY_NAME_MPP_Y])

    # For larger tiles in micron, NKI scanner outputs mppx slight different than mppy.
    # Force tiles to be squared.
    mpp_scale_factor = min(mpp_x, mpp_y)
    if mpp_x != mpp_y:
        print(
            f"mpp_x of {mpp_x} and mpp_y of {mpp_y} are not the same. Using smallest value: {mpp_scale_factor}"
        )

    tile_size_pix = round(tile_size_microns / mpp_scale_factor)

    # Use the tissue mask bounds as base offsets (+ a margin of a few tiles) to avoid wasting CPU power creating tiles that are never going
    # to be inside the tissue mask.
    tissue_margin_pix = tile_size_pix * 2
    minx, miny, maxx, maxy = tissue_mask_scaled.bounds
    min_offset_x = minx - tissue_margin_pix
    min_offset_y = miny - tissue_margin_pix
    offsets = [(min_offset_x, min_offset_y)]

    if offsets_micron is not None:
        assert (
            len(offsets_micron) > 0
        ), "offsets_micron needs to contain at least one value"
        # Compute the offsets in micron scale
        offset_pix = [round(o / mpp_scale_factor) for o in offsets_micron]
        offsets = [(o + min_offset_x, o + min_offset_y) for o in offset_pix]

    # Generate tiles covering the entire WSI
    all_tiles = generate_tiles(
        tile_size_pix,
        tile_size_pix,
        maxx + tissue_margin_pix,
        maxy + tissue_margin_pix,
        offsets=offsets,
    )

    # Retain only the tiles that sit within the tissue mask polygon
    filtered_tiles = [rect for rect in all_tiles if tissue_mask_scaled.intersects(rect)]

    return filtered_tiles

def tile_is_not_empty(tile, threshold_white=20):
    histogram = tile.histogram()

    # Take the median of each RGB channel. Alpha channel is not of interest.
    # If roughly each chanel median is below a threshold, i.e close to 0 till color value around 250 (white reference) then tile mostly white.
    whiteness_check = [0, 0, 0]
    for channel_id in (0, 1, 2):
        whiteness_check[channel_id] = np.median(
            histogram[256 * channel_id : 256 * (channel_id + 1)][100:200]
        )

    if all(c <= threshold_white for c in whiteness_check):
        # exclude tile
        return False

    # keep tile
    return True

def crop_rect_from_slide(slide, rect):
    minx, miny, maxx, maxy = rect.bounds
    # Note that the y-axis is flipped in the slide: the top of the shapely polygon is y = ymax,
    # but in the slide it is y = 0. Hence: miny instead of maxy.
    top_left_coords = (int(minx), int(miny))
    return slide.read_region(top_left_coords, 0, (int(maxx - minx), int(maxy - miny)))

class BagOfTiles(Dataset):
    def __init__(self, wsi, tiles, resize_to=224):
        self.wsi = wsi
        self.tiles = tiles

        self.roi_transforms = transforms.Compose(
            [
                # As we can't be sure that the input tile dimensions are all consistent, we resize
                # them to a commonly used size before feeding them to the model.
                # Note: assumes a square image.
                transforms.Resize(resize_to),
                # Turn the PIL image into a (C x H x W) float tensor in the range [0.0, 1.0]
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        tile = self.tiles[idx]
        img = crop_rect_from_slide(self.wsi, tile)

        # RGB filtering - calling here speeds up computation since it requires crop_rect_from_slide function.
        #is_tile_kept = tile_is_not_empty(img, threshold_white=20)
        is_tile_kept = True

        # Ensure the img is RGB, as expected by the pretrained model.
        # See https://pytorch.org/docs/stable/torchvision/models.html
        img = img.convert("RGB")

        # Ensure we have a square tile in our hands.
        # We can't handle non-squares currently, as this would requiring changes to
        # the aspect ratio when resizing.
        width, height = img.size
        assert width == height, "input image is not a square"

        img = self.roi_transforms(img).unsqueeze(0)
        coord = tile.bounds
        return img, coord, is_tile_kept

def collate_features(batch):
    # Item 2 is the boolean value from tile filtering.
    img = torch.cat([item[0] for item in batch if item[2]], dim=0)
    coords = np.vstack([item[1] for item in batch if item[2]])
    return [img, coords]

def mergedpatch_gen(features, coords, dist_threshold=4, corr_threshold = 0.6):

    # Get patch distance in pixels with rendered segmentation level. Note that each patch is squared and therefore same distance.
    patch_dist = abs(coords[0,2] - coords[0,0]) 
    print(patch_dist)
    
    # Compute feature similarity (cosine) and nearby pacthes (L2 norm - only need the top left x,y coordinates)
    cosine_matrix = cosine_similarity(features, features)
    coordinate_matrix = euclidean_distances(coords[:,:2], coords[:,:2])

    # NOTE: random selection for the first patch for patch merging might be less biased towards tissue orientation and size. 
    indices_avail = np.arange(features.shape[0])
    np.random.seed(0)  
    np.random.shuffle(indices_avail)

    # Merging together nearby patches and similar within pre-defined threshold. 
    mergedfeatures = []
    indices_used = []
    for ref in indices_avail:

        # This has been merged already
        if ref not in indices_used:

            # Making sure they won't be selected once more
            if indices_used:
                coordinate_matrix[ref,indices_used] = [np.Inf]*len(indices_used)
                cosine_matrix[ref,indices_used] = [0.0]*len(indices_used)
            
            indices_dist = np.where(coordinate_matrix[ref] < patch_dist*dist_threshold, 1 , 0)
            indices_corr = np.where(cosine_matrix[ref] > corr_threshold, 1 , 0)
            final_indices = indices_dist * indices_corr

            # which includes already the ref patch
            indices_used.extend(list(np.where(final_indices == 1)[0]))
            mergedfeatures.append(tuple((features[final_indices==1,:], coords[final_indices==1,:])))
        else:
            continue
        
    assert len(indices_used)==features.shape[0], f'Probably issue in contruscting merged features for graph {len(indices_used)}!={features.shape[0]}'

    return mergedfeatures

class HNSW:
    def __init__(self, space):
        self.space = space

    def fit(self, X):
        # See https://nmslib.github.io/nmslib/quickstart.html
        index = nmslib.init(space=self.space, method='hnsw')
        index.addDataPointBatch(X)
        index.createIndex()
        self.index_ = index
        return self

    def query(self, vector, topn):
        indices, dist = self.index_.knnQuery(vector, k=topn)
        return indices, dist

@torch.no_grad()
def extract_features(model, device, wsi, filtered_tiles, workers, out_size, batch_size, n_last_blocks, avgpool_patchtokens, depths):
    # Use multiple workers if running on the GPU, otherwise we'll need all workers for evaluating the model.
    kwargs = (
        {"num_workers": workers, "pin_memory": True} if device.type == "cuda" else {}
    )
    loader = DataLoader(
        dataset=BagOfTiles(wsi, filtered_tiles, resize_to=out_size),
        batch_size=batch_size,
        collate_fn=collate_features,
        **kwargs,
    )
    features_ = []
    coords_ = []
    for batch, coords in loader:
        batch = batch.to(device, non_blocking=True)
        # NOTE: Example using EsVIT. You may want to call your own feature extractor otherwise. 
        features = model.forward_return_n_last_blocks(batch, n_last_blocks, avgpool_patchtokens, depths).cpu().numpy()
        features_.extend(features)
        coords_.extend(coords)
    return np.asarray(features_), np.asarray(coords_)

def extract_save_features(args):
    # Derive the slide ID from its name.
    slide_id, _ = os.path.splitext(os.path.basename(args.input_slide))
    wip_file_path = os.path.join(args.output_dir, slide_id + "_wip.h5")
    output_file_path = os.path.join(args.output_dir, slide_id + "_features.h5")

    os.makedirs(args.output_dir, exist_ok=True)

    # Check if the _features output file already exist. If so, we terminate to avoid
    # overwriting it by accident. This also simplifies resuming bulk batch jobs.
    if os.path.exists(output_file_path):
        raise Exception(f"{output_file_path} already exists")

    # Open the slide for reading.
    wsi = openslide.open_slide(args.input_slide)

    # Decide on which slide level we want to base the segmentation.
    seg_level = wsi.get_best_level_for_downsample(64)

    # Run the segmentation and  tiling procedure.
    start_time = time.time()
    tissue_mask_scaled = create_tissue_mask(wsi, seg_level, method=args.method)
    filtered_tiles = create_tissue_tiles(wsi, tissue_mask_scaled, args.tile_size)

    # Build a figure for quality control purposes, to check if the tiles are where we expect them.
    qc_img = make_tile_QC_fig(filtered_tiles, wsi, seg_level, 2)
    qc_img_target_width = 1920
    qc_img = qc_img.resize((qc_img_target_width, int(qc_img.height / (qc_img.width / qc_img_target_width))))
    qc_img_file_path = os.path.join(args.output_dir, f"{slide_id}_features_QC.png")
    qc_img.save(qc_img_file_path)
    print(f"Finished creating {len(filtered_tiles)} tissue tiles in {time.time() - start_time}s")

    # Save QC figure.
    qc_img_file_path = os.path.join(
        args.output_dir, f"{slide_id}_N{len(mergedpatches)}mergedpatches_distThreshold{args.dist_threshold}_corrThreshold{args.corr_threshold}.png"
    )

    # Extract the rectangles, and compute the feature vectors. Example using EsVIT. 
    device = torch.device("cuda") 
    model, _, depths = load_encoder_esVIT(args, device)
    
    features, coords = extract_features(
        model,
        device,
        wsi,
        filtered_tiles,
        args.workers,
        args.out_size,
        args.batch_size,
        n_last_blocks = args.n_last_blocks, 
        avgpool_patchtokens = args.avgpool_patchtokens,
        depths = depths,
    )
    
    print(f'Number of features N={len(features)}')
    # Merging nearby patches with similar semantic. 
    mergedpatches = mergedpatch_gen(features, coords, dist_threshold=args.dist_threshold, corr_threshold=args.corr_threshold)
    print(f'Merging step => N={len(mergedpatches)}')

    # Saving features.
    torch.save(mergedpatches, wip_file_path)

    # Rename the file containing the patches to ensure we can easily
    # distinguish incomplete bags of patches (due to e.g. errors) from complete ones in case a job fails.
    os.rename(wip_file_path, output_file_path)

    print('Done.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Preprocessing script esvit', parents=[get_args_parser()])
    args = parser.parse_args()

    assert os.path.isfile(args.checkpoint), f'{args.checkpoint} does not exist'
    assert torch.cuda.is_available(), 'Need cuda for this job'
    assert os.path.isfile(args.input_slide), f'{args.input_slide} does not exist'

    extract_save_features(args)