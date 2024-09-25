# TensorFlow Object Detection Project

This project implements a custom object detection model using TensorFlow and TensorFlow Lite. It includes steps for training, converting, and testing the model on a custom dataset.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Project Structure](#project-structure)
3. [Setup](#setup)
4. [Training](#training)
5. [Model Conversion](#model-conversion)
6. [Testing](#testing)
7. [TensorFlow.js Conversion](#tensorflowjs-conversion)

## Prerequisites

- Python 3.x
- TensorFlow 2.x
- OpenCV
- Matplotlib
- TensorFlow Object Detection API
- Google Colab (optional, for GPU acceleration)

## Project Structure

```
project/
│
├── models/
│   └── research/
│       └── object_detection/
│
├── images/
│   ├── train/
│   ├── test/
│   └── valid/
│
├── TFModel/
│   └── 4plants/
│       └── vegs-fruits.tfrecord/
│
├── custom_model_lite/
│
└── mAP/
```

## Setup

1. Clone the TensorFlow models repository:
   ```
   !git clone --depth 1 https://github.com/tensorflow/models
   ```

2. Install the Object Detection API:
   ```
   !pip install /models/research/
   ```

3. Install additional dependencies:
   ```
   !pip install tensorflow==2.8.0
   !pip install tensorflow_io==0.23.1
   ```

## Training

1. Prepare your dataset in TFRecord format.
2. Set up the model configuration file.
3. Run the training script:
   ```
   !python model_main_tf2.py \
       --pipeline_config_path={pipeline_file} \
       --model_dir={model_dir} \
       --alsologtostderr \
       --num_train_steps={num_steps} \
       --sample_1_of_n_eval_examples=1
   ```

## Model Conversion

Convert the trained model to TensorFlow Lite format:

1. Export the inference graph:
   ```
   !python export_tflite_graph_tf2.py \
       --trained_checkpoint_dir {last_model_path} \
       --output_directory {output_directory} \
       --pipeline_config_path {pipeline_file}
   ```

2. Convert to TFLite:
   ```python
   converter = tf.lite.TFLiteConverter.from_saved_model('/content/custom_model_lite/saved_model')
   tflite_model = converter.convert()
   ```

## Testing

Test the TFLite model on sample images:

```python
tflite_detect_images(PATH_TO_MODEL, PATH_TO_IMAGES, PATH_TO_LABELS, min_conf_threshold, images_to_test)
```


## TensorFlow.js Conversion

Convert the model to TensorFlow.js format:

```
!tensorflowjs_converter \
    --input_format=tf_saved_model \
    --output_format=tfjs_graph_model \
    /content/drive/MyDrive/TFModel/july29/out_graph/saved_model \
    /content/drive/MyDrive/TFModel/july29/out_graph/graph_model
```

## Acknowledgments

- TensorFlow Object Detection API