import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.iddped_loader import IDDPeDTrajectoryDataset


def test_mock_dataset_loading():
    dataset = IDDPeDTrajectoryDataset(data_path=None, t_obs=15, t_pred=30)
    assert len(dataset) == 100
    sample = dataset[0]
    assert sample["obs_boxes"].shape == (15, 4)
    assert sample["pred_boxes"].shape == (30, 4)

    obs_b, pred_b = dataset.get_batch(batch_size=16)
    assert obs_b.shape == (16, 15, 4)
    assert pred_b.shape == (16, 30, 4)
    print("Dataset loader tests passed successfully!")


if __name__ == "__main__":
    test_mock_dataset_loading()

