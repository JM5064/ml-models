import numpy as np


def mpjpe_3D(preds_kp, labels_kp, K, image_size):
    """Calculates MPJPE (mean per joint position error)
    args:
        preds_kp: torch tensor [batch, num_keypoints, 3] predicted keypoints
        labels_kp: torch tensor [batch, num_keypoints, 3] GT keypoints
        K: instrinsics matrix
        scale: scale factor

    returns:
        mpjpe
    """

    preds_kp = np.array(preds_kp).squeeze()
    labels_kp = np.array(labels_kp).squeeze()
    K = np.array(K).squeeze()

    print(preds_kp[0])
    # Unnormalize x, y
    preds_kp[:, :, 0:2] *= image_size
    labels_kp[:, :, 0:2] *= image_size

    print(preds_kp[0])

    # Unnormalize depth
    preds_wrist_depths = preds_kp[:, 0, 2]
    preds_unnormalized_depths = preds_kp[:, :, 2] + preds_wrist_depths[:, None]

    labels_wrist_depths = labels_kp[:, 0, 2]
    labels_unnormalized_depths = labels_kp[:, :, 2] + labels_wrist_depths[:, None]

    # Set Z coord to 1
    preds_kp[:, :, 2] = preds_unnormalized_depths
    labels_kp[:, :, 2] = labels_unnormalized_depths

    print(preds_kp[0])

    # Reproject coordinates
    preds_reprojected = reproject_xyz2XYZ(preds_kp, K)
    labels_reprojected = reproject_xyz2XYZ(labels_kp, K)

    distances = np.linalg.norm(preds_reprojected - labels_reprojected, axis=-1)

    return distances.mean()


def reproject_xyz2XYZ(xyz, K):
    # Decompose intrinsics
    fx = K[:, 0, 0][:, None]
    fy = K[:, 1, 1][:, None]
    cx = K[:, 0, 2][:, None]
    cy = K[:, 1, 2][:, None]

    x = xyz[:, :, 0]
    y = xyz[:, :, 1]
    Z = xyz[:, :, 2]

    # Reproject xyz to XYZ
    X = (x - cx) * Z / fx
    Y = (y - cy) * Z / fy

    XYZ = np.stack([X, Y, Z], axis=-1)

    return XYZ



if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from datasets.FreiHAND.freihand_dataset import FreiHAND

    images_dir = 'datasets/FreiHAND/FreiHAND/FreiHAND_pub_v2/FreiHAND64/rgb'
    keypoints_json = 'datasets/FreiHAND/FreiHAND/FreiHAND_pub_v2/FreiHAND64/training_xyz.json'
    intrinsics_json = 'datasets/FreiHAND/FreiHAND/FreiHAND_pub_v2/FreiHAND64/training_K.json'
    scale_json = 'datasets/FreiHAND/FreiHAND/FreiHAND_pub_v2/FreiHAND64/training_scale.json'

    freihand = FreiHAND(images_dir, keypoints_json, intrinsics_json, scale_json, transform=None)

    dl = DataLoader(freihand, batch_size=2, shuffle=False)

    for np_image, tensor_keypoints, heatmaps, offset_masks, K, scale in dl:
        mpjpe_3D(tensor_keypoints, tensor_keypoints, K, 224)

        break

