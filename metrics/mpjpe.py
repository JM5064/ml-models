
def mpjpe(preds_kp, labels_kp):
    """Calculates MPJPE (mean per joint position error)
    args:
        preds_kp: torch tensor [batch, num_keypoints, dims] predicted keypoints
        labels_kp: torch tensor [batch, num_keypoints, dims] GT keypoints

    returns:
        mpjpe
    """

    distances = (preds_kp - labels_kp).norm(dim=-1)

    return distances.mean()
