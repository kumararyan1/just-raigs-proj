import random

import numpy
from PIL import Image
from helper import DEFAULT_GLAUCOMATOUS_FEATURES, inference_tasks, model_inference



def run():

    for image_file, save_prediction in inference_tasks():
        print(f"Running inference on {image_file}")

        # Call the binary model for binary classification
        is_referable_glaucoma_likelihood = model_inference(image_file, 'binary')[0, 0]
        is_referable_glaucoma = is_referable_glaucoma_likelihood > 0.5

        # If referable glaucoma is detected, run multi-label model
        if is_referable_glaucoma:
            features_prediction = model_inference(image_file, 'multi_label')[0]
            features = {key: bool(val) for key, val in zip(DEFAULT_GLAUCOMATOUS_FEATURES.keys(), features_prediction)}
        else:
            features = None

        save_prediction(
            is_referable_glaucoma,
            is_referable_glaucoma_likelihood,
            features,
        )
    return 0

if __name__ == "__main__":
    raise SystemExit(run())