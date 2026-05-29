import numpy as np
import pandas as pd

from ball_estimation.camera import Camera


def regenerate_hom_smooth(camera_smooth_path: str) -> dict:
    cam_df = pd.read_csv(camera_smooth_path)

    h_data = []
    for _, row in cam_df.iterrows():
        cam = Camera.from_pandas(row)
        H_inv = cam.to_inverse_homography()
        h_row = list(H_inv.flatten()) + [row["error"], int(row["frame_index"])]
        h_data.append(h_row)

    columns = [f"h{i}" for i in range(9)] + ["error", "frame_index"]
    hom_df = pd.DataFrame(h_data, columns=columns)
    return hom_df