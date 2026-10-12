import numpy as np
import matplotlib.pyplot as plt

def compute_occlusion_map(image,score_fn,patch_size=4,stride=2,baseline=0.0,batch_size=64):
    image = np.asarray(image)
    h,w,_ = image.shape

    img_flatten = image.reshape(-1)
    score = float(score_fn(img_flatten[None, :])[0])

    rows = list(range(0, h - patch_size + 1, stride))
    cols = list(range(0, w - patch_size + 1, stride))

    if rows[-1] != h - patch_size:
        rows.append(h - patch_size)

    if cols[-1] != w - patch_size:
        cols.append(w - patch_size)

    locations, masked_inputs = [], []

    for top in rows:
        for left in cols:
            occluded = image.copy()
            occluded[top:top+patch_size,left:left+patch_size,:] = baseline
            locations.append((top, left))
            occluded_flatten = occluded.reshape(-1)
            masked_inputs.append(occluded_flatten)

    masked_inputs = np.asarray(masked_inputs)

    masked_scores = []

    for start in range(0, len(masked_inputs),batch_size):
        batch = masked_inputs[start:start+batch_size]
        masked_scores.extend(score_fn(batch))

    masked_scores = np.asarray(masked_scores)

    deltas = score - masked_scores
    sensitivity = np.zeros((h,w), dtype=float)
    counts = np.zeros((h,w),dtype=float)

    for (top, left), delta in zip(locations, deltas):
        region = np.s_[top:top + patch_size,left:left + patch_size]
        sensitivity[region] += delta
        counts[region] += 1

    sensitivity /= np.maximum(counts, 1)
    return score, sensitivity

def plot_node_occlusion(image,sensitivity,title=None):
    image = np.asarray(image)

    limit = max(float(np.max(np.abs(sensitivity))),1e-12)
    magnitude = np.abs(sensitivity)/limit

    fig, ax = plt.subplots(1,3)

    ax[0].imshow(np.clip(image, 0, 1))
    ax[0].set_title("Original image")

    signed = ax[1].imshow(sensitivity,cmap="coolwarm", vmin=-limit,vmax=limit)
    ax[1].set_title("Signed sensitivity")
    fig.colorbar(signed, ax=ax[1], fraction=0.046, pad=0.04)

    ax[2].imshow(np.clip(image, 0, 1))
    ax[2].imshow( magnitude,cmap="inferno",vmin=0,vmax=1,alpha=0.5)
    ax[2].set_title("Sensitivity magnitude")

    for a in ax:
        a.axis("off")

    plt.tight_layout()
    fig.suptitle(title)
    # plt.show()
    return fig