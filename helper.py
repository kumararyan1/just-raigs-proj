from pathlib import Path
import json
import tempfile
from pprint import pprint

import SimpleITK as sitk
from PIL import Image

import tensorflow as tf
from tensorflow.keras.models import load_model

DEFAULT_GLAUCOMATOUS_FEATURES = {
    "appearance neuroretinal rim superiorly": None,
    "appearance neuroretinal rim inferiorly": None,
    "retinal nerve fiber layer defect superiorly": None,
    "retinal nerve fiber layer defect inferiorly": None,
    "baring of the circumlinear vessel superiorly": None,
    "baring of the circumlinear vessel inferiorly": None,
    "nasalization of the vessel trunk": None,
    "disc hemorrhages": None,
    "laminar dots": None,
    "large cup": None,
}



# Load your pre-trained models
binary_model = load_model('eye_model_vgg16_binary.keras')
multi_label_model = load_model('eye_model_vgg16_multi.keras')

IMG_WIDTH, IMG_HEIGHT = 224, 224
# BATCH_SIZE = 32

def preprocess_image(image):
    """Preprocess the image by resizing and converting to black and white."""
    image = tf.image.resize(image, (IMG_WIDTH, IMG_HEIGHT))
#     image = tf.image.rgb_to_grayscale(image)  # Convert to black and white
    image = tf.keras.applications.vgg16.preprocess_input(image)
    return image

def augment(image):
    """Apply random flips and rotations to the image."""
    # Random flip horizontally
    image = tf.image.random_flip_left_right(image)
    # Random flip vertically
    image = tf.image.random_flip_up_down(image)
    # Random rotation
    image = tf.image.rot90(image, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    return image

def load_and_preprocess_image(path):
    """Load, preprocess, and augment the image."""
    path = str(path)
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = preprocess_image(image)
    image = augment(image)
    return image

def model_inference(image_path, model_type):
    """
    Perform inference on a single image using specified model.
    """
    image = load_and_preprocess_image(image_path)
    image = tf.expand_dims(image, axis=0)  # Add batch dimension

    if model_type == 'binary':
        prediction = binary_model.predict(image)
    elif model_type == 'multi_label':
        prediction = multi_label_model.predict(image)

    return prediction

def inference_tasks():
    input_files = [x for x in Path("/input").rglob("*") if x.is_file()]

    print("Input Files:")
    pprint(input_files)

    is_referable_glaucoma_stacked = []
    is_referable_glaucoma_likelihood_stacked = []
    glaucomatous_features_stacked = []

    def save_prediction(
            is_referable_glaucoma,
            likelihood_referable_glaucoma,
            glaucomatous_features=None,
    ):
        is_referable_glaucoma_stacked.append(bool(is_referable_glaucoma))
        is_referable_glaucoma_likelihood_stacked.append(likelihood_referable_glaucoma)
        if glaucomatous_features is not None:
            glaucomatous_features_stacked.append({**DEFAULT_GLAUCOMATOUS_FEATURES, **glaucomatous_features})
        else:
            glaucomatous_features_stacked.append(DEFAULT_GLAUCOMATOUS_FEATURES)

    for file_path in input_files:
        if file_path.suffix == ".mha":  # A single image
            yield from single_file_inference(image_file=file_path, callback=save_prediction)
        elif file_path.suffix == ".tiff":  # A stack of images
            yield from stack_inference(stack=file_path, callback=save_prediction)

    write_referable_glaucoma_decision(is_referable_glaucoma_stacked)
    write_referable_glaucoma_decision_likelihood(
        is_referable_glaucoma_likelihood_stacked
    )
    write_glaucomatous_features(glaucomatous_features_stacked)


def single_file_inference(image_file, callback):
    with tempfile.TemporaryDirectory() as temp_dir:
        image = sitk.ReadImage(image_file)

        # Define the output file path
        output_path = Path(temp_dir) / "image.jpg"

        # Save the 2D slice as a JPG file
        sitk.WriteImage(image, str(output_path))

        # Call back that saves the result
        def save_prediction(
            is_referable_glaucoma,
            likelihood_referable_glaucoma,
            glaucomatous_features=None,
        ):
            glaucomatous_features = (
                glaucomatous_features or DEFAULT_GLAUCOMATOUS_FEATURES
            )
            write_referable_glaucoma_decision([is_referable_glaucoma])
            write_referable_glaucoma_decision_likelihood(
                [likelihood_referable_glaucoma]
            )
            write_glaucomatous_features(
                [{**DEFAULT_GLAUCOMATOUS_FEATURES, **glaucomatous_features}]
            )

        yield output_path, callback


def stack_inference(stack, callback):
    de_stacked_images = []

    # Unpack the stack
    with tempfile.TemporaryDirectory() as temp_dir:
        with Image.open(stack) as tiff_image:

            # Iterate through all pages
            for page_num in range(tiff_image.n_frames):
                # Select the current page
                tiff_image.seek(page_num)

                # Define the output file path
                output_path = Path(temp_dir) / f"image_{page_num + 1}.jpg"
                tiff_image.save(output_path, "JPEG")

                de_stacked_images.append(output_path)

                print(f"De-Stacked {output_path}")

        # Loop over the images, and generate the actual tasks
        for index, image in enumerate(de_stacked_images):
            # Call back that saves the result
            yield image, callback


def write_referable_glaucoma_decision(result):
    with open(f"/output/multiple-referable-glaucoma-binary.json", "w") as f:
        result = [bool(x) for x in result]
        f.write(json.dumps(result))


def write_referable_glaucoma_decision_likelihood(result):
    with open(f"/output/multiple-referable-glaucoma-likelihoods.json", "w") as f:
        result = [float(x) for x in result]
        f.write(json.dumps(result))


def write_glaucomatous_features(result):
    with open(f"/output/stacked-referable-glaucomatous-features.json", "w") as f:
        f.write(json.dumps(result))