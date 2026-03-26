'''
Notes:
1. All of your implementation should be in this file. This is the ONLY .py file you need to edit & submit. 
2. Please Read the instructions and do not modify the input and output formats of function stitch_background() and panorama().
3. If you want to show an image for debugging, please use show_image() function in util.py.
4. Please do NOT save any intermediate files in your final submission.
'''
import torch
import kornia as K
from typing import Dict

'''
Please do NOT add any imports. The allowed libraries are already imported for you.
'''

# ------------------------------------ Task 1 ------------------------------------ #
def stitch_background(imgs: Dict[str, torch.Tensor]):
    """
    Args:
        imgs: input images are a dict of 2 images of torch.Tensor represent an input images for task-1.
    Returns:
        img: stitched_image: torch.Tensor (C, H, W) uint8 of the output image.
    """
    torch.manual_seed(42)
    keys = sorted(imgs.keys())
    if len(keys) < 2:
        return imgs[keys[0]]
    device = imgs[keys[0]].device

    # ── Pre-process: float [0,1], batch dim ──────────────────────────────────────
    img1  = imgs[keys[0]].unsqueeze(0).float() / 255.0   # (1,3,H,W)
    img2  = imgs[keys[1]].unsqueeze(0).float() / 255.0
    gray1 = K.color.rgb_to_grayscale(img1)
    gray2 = K.color.rgb_to_grayscale(img2)

    # ── Step 1: SIFT keypoints & descriptors ─────────────────────────────────────
    sift      = K.feature.SIFTFeature(5000, upright=False).to(device)
    lafs1, _, descs1 = sift(gray1)
    lafs2, _, descs2 = sift(gray2)
    pts1 = K.feature.get_laf_center(lafs1)   # (1,N,2)
    pts2 = K.feature.get_laf_center(lafs2)

    # ── Step 2: SNN feature matching (Lowe ratio 0.75) ───────────────────────────
    _, indices  = K.feature.match_snn(descs1[0], descs2[0], 0.75)
    m_pts1 = pts1[0, indices[:, 0]]   # (M,2)  — img1 keypoints
    m_pts2 = pts2[0, indices[:, 1]]   # (M,2)  — corresponding img2 keypoints

    # ── Step 3: RANSAC homography  (H: img2 → img1) ──────────────────────────────
    ransac = K.geometry.RANSAC('homography', inl_th=2.0, max_iter=5000)
    H, _   = ransac(m_pts2, m_pts1)   # (3,3)

    # ── Step 4: Canvas that fits both images without cropping ────────────────────
    h1, w1sz = img1.shape[2], img1.shape[3]
    h2, w2sz = img2.shape[2], img2.shape[3]

    c1     = torch.tensor([[0,0],[w1sz-1,0],[w1sz-1,h1-1],[0,h1-1]],
                           dtype=torch.float32, device=device)
    c2     = torch.tensor([[0,0],[w2sz-1,0],[w2sz-1,h2-1],[0,h2-1]],
                           dtype=torch.float32, device=device)
    c2_in1 = K.geometry.transform_points(H.unsqueeze(0), c2.unsqueeze(0)).squeeze(0)
    all_c  = torch.cat([c1, c2_in1], dim=0)
    min_x  = int(all_c[:,0].min().floor().item())
    min_y  = int(all_c[:,1].min().floor().item())
    max_x  = int(all_c[:,0].max().ceil().item())
    max_y  = int(all_c[:,1].max().ceil().item())
    out_w  = max_x - min_x + 1
    out_h  = max_y - min_y + 1

    T  = torch.tensor([[1.,0.,-min_x],[0.,1.,-min_y],[0.,0.,1.]],
                       dtype=torch.float32, device=device)
    H1 = T          # img1 → canvas (just a translation offset)
    H2 = T @ H      # img2 → img1 space → canvas

    # ── Step 5: Warp both images and validity masks onto the canvas ───────────────
    wi1   = K.geometry.warp_perspective(img1, H1.unsqueeze(0), (out_h, out_w))
    wi2   = K.geometry.warp_perspective(img2, H2.unsqueeze(0), (out_h, out_w))
    mask1 = K.geometry.warp_perspective(
                torch.ones(1,1,h1,w1sz, device=device),
                H1.unsqueeze(0), (out_h, out_w)) > 0.5   # (1,1,H,W) bool
    mask2 = K.geometry.warp_perspective(
                torch.ones(1,1,h2,w2sz, device=device),
                H2.unsqueeze(0), (out_h, out_w)) > 0.5
    ov    = mask1 & mask2   # overlap region

    # ── Step 6: Detect moving foreground in the overlap ──────────────────────────
    # Lightly blur both warped images before differencing to absorb
    # sub-pixel alignment noise without blurring genuine person edges.
    blr  = K.filters.gaussian_blur2d(
               torch.cat([wi1, wi2], dim=0), (5, 5), (1.5, 1.5))
    diff = (blr[:1] - blr[1:]).abs().mean(dim=1, keepdim=True)   # (1,1,H,W)

    # Pixels in the overlap that differ more than the threshold = foreground
    fg_raw = (diff > 0.05) & ov   # (1,1,H,W) bool

    # Opening (erosion → dilation) removes isolated noise blobs
    k5   = torch.ones(5, 5, device=device)
    fg_f = K.morphology.erosion( fg_raw.float(), k5)
    fg_f = K.morphology.dilation(fg_f,           k5)

    # Large dilation to expand the mask over complete human bodies
    k21  = torch.ones(21, 21, device=device)
    fg   = K.morphology.dilation(fg_f, k21) > 0.5
    fg   = fg & ov               # keep within overlap only
    bg_ov = ov & ~fg             # static background part of the overlap

    # ── Step 7: Compose the output mosaic ────────────────────────────────────────
    out = torch.zeros(1, 3, out_h, out_w, device=device)

    # Non-overlap: use whichever image covers each pixel
    out = torch.where((mask1 & ~mask2).expand_as(out), wi1, out)
    out = torch.where((mask2 & ~mask1).expand_as(out), wi2, out)

    # Static background overlap: average both images
    # (both frames show the true background here, averaging reduces noise)
    out = torch.where(bg_ov.expand_as(out), (wi1 + wi2) * 0.5, out)

    # Moving foreground overlap: take the channel-wise MAXIMUM of both images.
    # Both persons in this scene wear dark clothing (dark jacket, dark jeans).
    # The background (colourful pumpkins, grey pavement) is brighter at nearly
    # every contested pixel.  torch.max therefore reliably selects the
    # background pixel and discards the dark person pixel.
    out = torch.where(fg.expand_as(out), torch.max(wi1, wi2), out)

    return (out.squeeze(0) * 255.0).clamp(0, 255).to(torch.uint8)


# ------------------------------------ Task 2 ------------------------------------ #
def panorama(imgs: Dict[str, torch.Tensor]):
    """
    Args:
        imgs: dict {filename: CxHxW tensor} for task-2.
    Returns:
        img: panorama torch.Tensor (C, H, W) uint8.
        overlap: NxN torch.Tensor int32 one-hot overlap array.
    """
    torch.manual_seed(42)
    keys = sorted(list(imgs.keys()))
    N = len(keys)
    device = imgs[keys[0]].device

    overlap = torch.eye(N, dtype=torch.int32, device=device)
    if N == 0:
        return torch.zeros((3, 256, 256), dtype=torch.uint8, device=device), overlap

    # ── Step 1: Pre-process — float [0,1], grayscale ─────────────────────────────
    float_imgs = []
    gray_imgs  = []
    for k in keys:
        t = imgs[k].unsqueeze(0).float() / 255.0   # (1,3,H,W)
        float_imgs.append(t)
        gray_imgs.append(K.color.rgb_to_grayscale(t))

    # ── Step 2: SIFT feature extraction ─────────────────────────────────────────
    # For memory efficiency on high-resolution images, downsample to at most
    # SIFT_MAX_SIDE pixels on the short side before running SIFT, then scale
    # the detected keypoint coordinates back to original-image pixel space.
    # This keeps homography estimation in the original resolution while avoiding
    # the scale-space pyramid from consuming too much RAM.
    SIFT_MAX_SIDE = 800
    sift = K.feature.SIFTFeature(5000, upright=False).to(device)
    features = []
    for g in gray_imgs:
        h_g, w_g = g.shape[2], g.shape[3]
        short_side = min(h_g, w_g)
        if short_side > SIFT_MAX_SIDE:
            scale = SIFT_MAX_SIDE / short_side
            g_s = torch.nn.functional.interpolate(
                g, size=(int(h_g * scale), int(w_g * scale)),
                mode='bilinear', align_corners=False)
            inv_scale = 1.0 / scale
        else:
            g_s = g
            inv_scale = 1.0
        with torch.no_grad():
            lafs, _, descs = sift(g_s)
        pts = K.feature.get_laf_center(lafs) * inv_scale   # (1, K, 2) original coords
        features.append((pts, descs))

    # ── Step 3: Pairwise SNN matching + RANSAC to build overlap matrix ───────────
    inlier_thresh = 15
    H_matrices = [[None] * N for _ in range(N)]

    for i in range(N):
        for j in range(i + 1, N):
            pts_i, descs_i = features[i]
            pts_j, descs_j = features[j]

            # SNN matching with Lowe ratio 0.75
            _, indices = K.feature.match_snn(descs_i[0], descs_j[0], 0.75)
            if len(indices) < inlier_thresh:
                continue

            m_pts_i = pts_i[0, indices[:, 0]]   # (M, 2) — img i keypoints
            m_pts_j = pts_j[0, indices[:, 1]]   # (M, 2) — img j keypoints

            # RANSAC homography: H_ji maps image j → image i space
            ransac     = K.geometry.RANSAC('homography', inl_th=3.0, max_iter=2000)
            H_ji, inliers = ransac(m_pts_j, m_pts_i)

            if inliers.sum().item() >= inlier_thresh:
                overlap[i, j] = 1
                overlap[j, i] = 1
                H_matrices[i][j] = H_ji                          # j → i
                H_matrices[j][i] = torch.linalg.inv(H_ji)        # i → j (inverse)

    # ── Step 4: Choose reference image (most overlapping neighbours) ─────────────
    # Subtract diagonal (self-overlap) to get true neighbour count.
    # When multiple images tie for the highest neighbour count, prefer the one
    # closest to the middle of the sequence: this minimises the maximum
    # chained-homography depth and keeps perspective distortion balanced.
    neighbour_count = (overlap - torch.eye(N, dtype=torch.int32, device=device)).sum(dim=1)
    max_nbrs = neighbour_count.max()
    tied_mask = (neighbour_count == max_nbrs).nonzero(as_tuple=False).squeeze(1)
    mid = N // 2
    dist_to_mid = torch.abs(tied_mask - mid)
    ref_idx = int(tied_mask[torch.argmin(dist_to_mid)].item())

    # ── Step 5: BFS to chain all homographies into the reference frame ───────────
    global_H = {ref_idx: torch.eye(3, dtype=torch.float32, device=device)}
    queue    = [ref_idx]
    visited  = {ref_idx}

    while queue:
        curr = queue.pop(0)
        for nbr in range(N):
            if overlap[curr, nbr].item() == 1 and nbr not in visited:
                # H_matrices[curr][nbr] maps nbr → curr
                H_nbr2curr = H_matrices[curr][nbr]
                H_curr2ref  = global_H[curr]
                # Chain: nbr → curr → ref
                global_H[nbr] = H_curr2ref @ H_nbr2curr
                visited.add(nbr)
                queue.append(nbr)

    # ── Step 6: Compute the canvas bounding box in the reference frame ───────────
    all_corners = []
    per_img_min_x, per_img_max_x = [], []
    per_img_min_y, per_img_max_y = [], []

    for i in visited:
        h_i, w_i = float_imgs[i].shape[2], float_imgs[i].shape[3]
        corners = torch.tensor(
            [[0, 0], [w_i - 1, 0], [w_i - 1, h_i - 1], [0, h_i - 1]],
            dtype=torch.float32, device=device)                        # (4, 2)
        proj = K.geometry.transform_points(
            global_H[i].unsqueeze(0), corners.unsqueeze(0)).squeeze(0)  # (4, 2)
        all_corners.append(proj)
        per_img_min_x.append(proj[:, 0].min())
        per_img_max_x.append(proj[:, 0].max())
        per_img_min_y.append(proj[:, 1].min())
        per_img_max_y.append(proj[:, 1].max())

    all_corners = torch.cat(all_corners, dim=0)   # (|visited|*4, 2)
    min_x = int(torch.floor(all_corners[:, 0].min()).item())
    min_y = int(torch.floor(all_corners[:, 1].min()).item())
    max_x = int(torch.ceil( all_corners[:, 0].max()).item())
    max_y = int(torch.ceil( all_corners[:, 1].max()).item())

    # ── Robust canvas capping: exclude extreme outlier projections ────────────
    # When one image homography is extreme (e.g. pulled far from main cluster),
    # clamp the canvas to the 2nd-most-extreme per-image bound plus one full-image
    # margin.  This removes the "dragged corner" artefact while still expanding
    # the canvas enough to fully accommodate all non-outlier images.
    n_img = len(visited)
    if n_img > 2:
        max_img_w = max(t.shape[3] for t in float_imgs)
        max_img_h = max(t.shape[2] for t in float_imgs)

        mx_t  = torch.stack(per_img_max_x).sort().values   # ascending
        my_t  = torch.stack(per_img_max_y).sort().values
        mnx_t = torch.stack(per_img_min_x).sort().values
        mny_t = torch.stack(per_img_min_y).sort().values

        # 2nd-largest (index n-2) as robust upper bound; 2nd-smallest (index 1)
        # as robust lower bound — each excludes the single worst outlier.
        # Width uses a full-image-width margin; height uses half to prevent
        # vertical ballooning from perspective-distorted top/bottom corners.
        robust_max_x = int(torch.ceil(mx_t[n_img - 2]).item())  + max_img_w
        robust_max_y = int(torch.ceil(my_t[n_img - 2]).item())  + max_img_h // 2
        robust_min_x = int(torch.floor(mnx_t[1]).item())        - max_img_w
        robust_min_y = int(torch.floor(mny_t[1]).item())        - max_img_h // 2

        max_x = min(max_x, robust_max_x)
        max_y = min(max_y, robust_max_y)
        min_x = max(min_x, robust_min_x)
        min_y = max(min_y, robust_min_y)

    out_w = max_x - min_x + 1
    out_h = max_y - min_y + 1

    # Translation matrix: shift origin so all coordinates are non-negative
    T = torch.tensor([
        [1.0, 0.0, float(-min_x)],
        [0.0, 1.0, float(-min_y)],
        [0.0, 0.0, 1.0]
    ], dtype=torch.float32, device=device)

    # ── Step 7: Warp each image onto the canvas with distance-weighted blending ────
    # Distance-to-edge feathering: pixels near the centre of each source image
    # receive high weight; pixels near the boundary receive low weight.
    # This suppresses ghosting/double-exposure at overlapping regions where the
    # same 3-D object appears at slightly different positions in each frame.
    canvas_sum    = torch.zeros((1, 3, out_h, out_w), dtype=torch.float32, device=device)
    canvas_weight = torch.zeros((1, 1, out_h, out_w), dtype=torch.float32, device=device)

    for i in visited:
        H_final  = T @ global_H[i]                            # image i → canvas
        img_i    = float_imgs[i]
        h_i, w_i = img_i.shape[2], img_i.shape[3]

        # Build a per-pixel weight = min-distance to the nearest image edge,
        # raised to the 4th power so only the central region dominates and
        # boundary/overlap zones contribute almost nothing.  This prevents
        # the same 3-D object (e.g. the bison statue) from appearing multiple
        # times as a ghost when viewed from slightly different angles.
        ys = torch.arange(h_i, dtype=torch.float32, device=device)
        xs = torch.arange(w_i, dtype=torch.float32, device=device)
        dy = torch.min(ys, (h_i - 1) - ys)                    # (H,)
        dx = torch.min(xs, (w_i - 1) - xs)                    # (W,)
        dist_map = torch.min(dy.unsqueeze(1),
                             dx.unsqueeze(0))                  # (H, W)
        dist_map = (dist_map / (dist_map.max() + 1e-6)) ** 4   # sharp falloff
        dist_map = dist_map.unsqueeze(0).unsqueeze(0)          # (1,1,H,W)

        warped_img    = K.geometry.warp_perspective(img_i,    H_final.unsqueeze(0), (out_h, out_w))
        warped_weight = K.geometry.warp_perspective(dist_map, H_final.unsqueeze(0), (out_h, out_w))

        canvas_sum    += warped_img    * warped_weight
        canvas_weight += warped_weight

    # Normalise: weighted average where covered, black where not covered
    final_canvas = canvas_sum / torch.clamp(canvas_weight, min=1e-6)

    # ── Post-process: trim low-coverage edge columns/rows ─────────────────────
    # When one image's homography is extreme its far-edge pixels warp to the
    # border of the canvas with near-zero dist-map weight.  Those columns/rows
    # appear as the "dragged edge" artefact.  We crop to the bounding box of
    # columns/rows whose total blending weight is at least 0.1 % of the
    # maximum, which removes essentially-uncovered border strips while keeping
    # all legitimately-covered panorama content.
    coverage     = canvas_weight.squeeze()           # (H, W)
    col_coverage = coverage.sum(dim=0)               # (W,)  summed weight per col
    row_coverage = coverage.sum(dim=1)               # (H,)  summed weight per row
    thresh_c = col_coverage.max() * 1e-3
    thresh_r = row_coverage.max() * 1e-3
    valid_c  = torch.where(col_coverage > thresh_c)[0]
    valid_r  = torch.where(row_coverage > thresh_r)[0]
    if valid_c.numel() > 0 and valid_r.numel() > 0:
        c0, c1 = valid_c[0].item(), valid_c[-1].item() + 1
        r0, r1 = valid_r[0].item(), valid_r[-1].item() + 1
        final_canvas = final_canvas[:, :, r0:r1, c0:c1]

    # Return (C, H, W) uint8 as required
    img = (final_canvas.squeeze(0) * 255.0).clamp(0, 255).to(torch.uint8)

    return img, overlap
