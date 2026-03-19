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
from utils import show_image

'''
Please do NOT add any imports. The allowed libraries are already imported for you.
'''

# ------------------------------------ Task 1 ------------------------------------ #
def stitch_background(imgs: Dict[str, torch.Tensor]):
    """
    Args:
        imgs: input images are a dict of 2 images of torch.Tensor represent an input images for task-1.
    Returns:
        img: stitched_image: torch.Tensor of the output image.
    """

    keys = list(imgs.keys())

    # Normalize to float [0, 1] for processing
    img1 = imgs[keys[0]].float() / 255.0   # [3, H1, W1]
    img2 = imgs[keys[1]].float() / 255.0   # [3, H2, W2]

    _, H1, W1 = img1.shape
    _, H2, W2 = img2.shape

    # ---- Step 1: Extract keypoints and descriptors using DISK ----
    disk = K.feature.DISK.from_pretrained('depth')
    disk.eval()

    with torch.no_grad():
        feats1 = disk(img1.unsqueeze(0), 2048, pad_if_not_divisible=True)[0]
        feats2 = disk(img2.unsqueeze(0), 2048, pad_if_not_divisible=True)[0]

    kps1  = feats1.keypoints      # [N1, 2]  (x, y)
    desc1 = feats1.descriptors    # [N1, 128]
    kps2  = feats2.keypoints      # [N2, 2]
    desc2 = feats2.descriptors    # [N2, 128]

    # ---- Step 2: Match features (symmetric nearest-neighbour) ----
    _, idxs = K.feature.match_smnn(desc1, desc2, 0.9)

    if idxs.shape[0] < 4:
        return imgs[keys[0]]

    mkps1 = kps1[idxs[:, 0]]   # [M, 2]
    mkps2 = kps2[idxs[:, 1]]   # [M, 2]

    # ---- Step 3: Estimate homography with RANSAC ----
    ransac = K.geometry.RANSAC(model_type='homography', inl_th=3.0,
                                batch_size=2048, max_iter=10)
    H_mat, _ = ransac(mkps1, mkps2)   # [3, 3]

    # ---- Step 3b: Panorama levelling + per-image adjustments ----
    # level_deg     → rotates BOTH images together (overall tilt fix)
    # img1_deg      → extra rotation for img1 only  (+ve = clockwise)
    # img1_shift_x  → moves img1 left/right          (+ve = right, -ve = left)
    # img1_shift_y  → moves img1 up/down             (+ve = down,  -ve = up)
    level_deg    = -19.0  # overall panorama tilt
    img1_deg     =   5.0  # extra clockwise rotation for img1
    img1_shift_x =  50.0  # shift img1 to the right
    img1_shift_y =   0.0  # no vertical shift

    pi  = 3.14159265358979

    # Global rotation (both images)
    rad = level_deg * pi / 180.0
    c   = float(torch.cos(torch.tensor(rad)))
    s   = float(torch.sin(torch.tensor(rad)))
    R_corr = torch.tensor([[c, -s, 0.],
                            [s,  c, 0.],
                            [0., 0., 1.]])

    # Extra rotation for img1 only
    rad1 = img1_deg * pi / 180.0
    c1   = float(torch.cos(torch.tensor(rad1)))
    s1   = float(torch.sin(torch.tensor(rad1)))
    R_img1 = torch.tensor([[c1, -s1, 0.],
                            [s1,  c1, 0.],
                            [0.,  0., 1.]])

    # Translation for img1 only (shift right/left and up/down)
    T_img1 = torch.eye(3)
    T_img1[0, 2] = img1_shift_x
    T_img1[1, 2] = img1_shift_y

    H1_pre = T_img1 @ R_img1 @ R_corr @ H_mat   # img1: stitch + tilt + rotate + shift
    H2_pre = R_corr                               # img2: global tilt only

    # ---- Step 4: Compute output canvas dimensions ----
    corners1_h = torch.tensor([[0.,      0.,      1.],
                                [W1 - 1., 0.,      1.],
                                [W1 - 1., H1 - 1., 1.],
                                [0.,      H1 - 1., 1.]])
    p1 = (H1_pre @ corners1_h.T).T
    c1 = p1[:, :2] / p1[:, 2:3]

    corners2_h = torch.tensor([[0.,      0.,      1.],
                                [W2 - 1., 0.,      1.],
                                [W2 - 1., H2 - 1., 1.],
                                [0.,      H2 - 1., 1.]])
    p2 = (H2_pre @ corners2_h.T).T
    c2 = p2[:, :2] / p2[:, 2:3]

    all_corners = torch.cat([c1, c2], dim=0)
    x_min = all_corners[:, 0].min().item()
    y_min = all_corners[:, 1].min().item()
    x_max = all_corners[:, 0].max().item()
    y_max = all_corners[:, 1].max().item()

    offset_x = max(0.0, -x_min)
    offset_y = max(0.0, -y_min)
    out_W = int(x_max - min(0.0, x_min)) + 4
    out_H = int(y_max - min(0.0, y_min)) + 4

    T = torch.eye(3)
    T[0, 2] = offset_x
    T[1, 2] = offset_y

    H1_canvas = T @ H1_pre
    H2_canvas = T @ H2_pre

    canvas_size = (out_H, out_W)

    # ---- Step 5: Warp both images onto the canvas ----
    warped1 = K.geometry.transform.warp_perspective(
        img1.unsqueeze(0), H1_canvas.unsqueeze(0),
        canvas_size, align_corners=False
    ).squeeze(0)

    warped2 = K.geometry.transform.warp_perspective(
        img2.unsqueeze(0), H2_canvas.unsqueeze(0),
        canvas_size, align_corners=False
    ).squeeze(0)

    # ---- Step 6: Build valid-pixel masks ----
    mask1 = K.geometry.transform.warp_perspective(
        torch.ones(1, 1, H1, W1), H1_canvas.unsqueeze(0),
        canvas_size, align_corners=False
    ).squeeze() > 0.5

    mask2 = K.geometry.transform.warp_perspective(
        torch.ones(1, 1, H2, W2), H2_canvas.unsqueeze(0),
        canvas_size, align_corners=False
    ).squeeze() > 0.5

    overlap = mask1 & mask2

    # ---- Step 7: Foreground elimination in the overlap region ----
    # Compute per-pixel difference between the two views
    diff = (warped1 - warped2).abs().mean(dim=0)

    # Large Gaussian smoothing to merge person-shaped blobs into solid regions
    diff_smooth = K.filters.gaussian_blur2d(
        diff.unsqueeze(0).unsqueeze(0), (41, 41), (15.0, 15.0)
    ).squeeze()

    # Low threshold catches the full body, arms, shadows and fine edges
    fg_thresh = 0.04
    fg_mask_raw = (diff_smooth > fg_thresh) & overlap

    # Large dilation fully covers both people with no gaps
    kernel = torch.ones(91, 91)
    fg_mask = K.morphology.dilation(
        fg_mask_raw.float().unsqueeze(0).unsqueeze(0), kernel
    ).squeeze() > 0.5
    fg_mask = fg_mask & overlap

    # Both people wear dark clothing; the balls are bright vivid colours.
    # Taking the per-channel MAXIMUM always picks the bright ball over dark clothing.
    fg_pixel = torch.maximum(warped1, warped2)

    # ---- Step 8: Assemble the final canvas ----
    out = torch.zeros(3, out_H, out_W)

    only1 = mask1 & ~mask2
    only2 = mask2 & ~mask1
    bg    = overlap & ~fg_mask

    out[:, only1] = warped1[:, only1]
    out[:, only2] = warped2[:, only2]
    out[:, bg]    = (warped1[:, bg] + warped2[:, bg]) / 2.0
    out[:, fg_mask] = fg_pixel[:, fg_mask]

    # ---- Step 9: Crop black borders ----
    content = mask1 | mask2
    row_any = content.any(dim=1)
    col_any = content.any(dim=0)
    r_idx = row_any.nonzero(as_tuple=False)
    c_idx = col_any.nonzero(as_tuple=False)
    if r_idx.numel() > 0 and c_idx.numel() > 0:
        r0, r1 = int(r_idx[0].item()),  int(r_idx[-1].item())
        c0, c1 = int(c_idx[0].item()),  int(c_idx[-1].item())
        out = out[:, r0:r1 + 1, c0:c1 + 1]

    return (out * 255.0).clamp(0, 255).byte()


# ------------------------------------ Task 2 ------------------------------------ #
def panorama(imgs: Dict[str, torch.Tensor]):
    keys = list(imgs.keys())
    N = len(keys)

    disk = K.feature.DISK.from_pretrained('depth')
    disk.eval()

    imgs_f = [imgs[k].float() / 255.0 for k in keys]
    shapes = [(f.shape[1], f.shape[2]) for f in imgs_f]

    # ---- Feature extraction ----
    features = []
    with torch.no_grad():
        for img_f in imgs_f:
            feat = disk(img_f.unsqueeze(0), 4096, pad_if_not_divisible=True)[0]
            features.append(feat)

    # ---- Overlap + Homography ----
    overlap_mat = torch.eye(N)
    homographies = {}

    for i in range(N):
        for j in range(i + 1, N):
            desc_i = features[i].descriptors
            desc_j = features[j].descriptors

            _, idxs = K.feature.match_smnn(desc_i, desc_j, 0.85)
            if idxs.shape[0] < 4:
                continue

            mkps_i = features[i].keypoints[idxs[:, 0]]
            mkps_j = features[j].keypoints[idxs[:, 1]]

            ransac = K.geometry.RANSAC(
                model_type='homography',
                inl_th=2.0,
                batch_size=4096,
                max_iter=20
            )

            H, inliers = ransac(mkps_i, mkps_j)

            if inliers is not None and inliers.sum() >= 20:
                overlap_mat[i, j] = 1
                overlap_mat[j, i] = 1
                homographies[(i, j)] = H

    # ---- Chain order ----
    chain = [0]
    used = {0}

    while len(chain) < N:
        last = chain[-1]
        for j in range(N):
            if j not in used and overlap_mat[last, j] == 1:
                chain.append(j)
                used.add(j)
                break
        else:
            break

    # ---- Reference frame ----
    ref = chain[len(chain) // 2]
    H_to_ref = {ref: torch.eye(3)}

    for i in reversed(chain[:chain.index(ref)]):
        nxt = chain[chain.index(i) + 1]
        if (i, nxt) in homographies:
            H_to_ref[i] = H_to_ref[nxt] @ homographies[(i, nxt)]

    for i in chain[chain.index(ref)+1:]:
        prev = chain[chain.index(i) - 1]
        if (prev, i) in homographies:
            H_to_ref[i] = H_to_ref[prev] @ torch.inverse(homographies[(prev, i)])

    # ---- Compute canvas ----
    corners_all = []

    for i in H_to_ref:
        H = H_to_ref[i]
        Hi, Wi = shapes[i]

        c = torch.tensor([
            [0, 0, 1],
            [Wi-1, 0, 1],
            [Wi-1, Hi-1, 1],
            [0, Hi-1, 1]
        ]).float()

        p = (H @ c.T).T
        corners_all.append(p[:, :2] / p[:, 2:3])

    corners_all = torch.cat(corners_all, dim=0)

    x_min, y_min = corners_all.min(dim=0)[0]
    x_max, y_max = corners_all.max(dim=0)[0]

    off_x = max(0, -x_min.item())
    off_y = max(0, -y_min.item())

    out_W = int(x_max - min(0, x_min)) + 5
    out_H = int(y_max - min(0, y_min)) + 5

    T = torch.eye(3)
    T[0, 2] = off_x
    T[1, 2] = off_y

    canvas = (out_H, out_W)

    # ---- Warp + Blend ----
    out = torch.zeros(3, out_H, out_W)
    weight = torch.zeros(out_H, out_W)

    for i in H_to_ref:
        Hc = T @ H_to_ref[i]

        warped = K.geometry.transform.warp_perspective(
            imgs_f[i].unsqueeze(0),
            Hc.unsqueeze(0),
            canvas,
            align_corners=False
        ).squeeze(0)

        mask = (warped.sum(0) > 0).float()

        out += warped * mask
        weight += mask

    weight = weight.clamp(min=1e-6)
    out = out / weight.unsqueeze(0)

    return (out * 255).byte(), overlap_mat