import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.trajectory_probe import ConstantVelocityPredictor, NeuralBBoxPredictor


def test_models():
    # Batch of 4 samples, 15 observation frames, 4 box coordinates
    obs = np.random.uniform(100, 300, (4, 15, 4)).astype(np.float32)
    # Ensure x2 > x1, y2 > y1
    obs[:, :, 2] += 40.0
    obs[:, :, 3] += 100.0

    # 1. Test Constant Velocity Predictor
    cv_model = ConstantVelocityPredictor()
    pred_cv = cv_model.predict(obs, t_pred=30)
    assert pred_cv.shape == (4, 30, 4)

    # 2. Test Neural BBox Predictor
    nn_model = NeuralBBoxPredictor()
    pred_nn = nn_model.predict(obs, t_pred=30)
    assert pred_nn.shape == (4, 30, 4)

    print("Model probing tests passed successfully!")


if __name__ == "__main__":
    test_models()

