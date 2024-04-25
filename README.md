# TensorFlow Object Detection API v2
## Setup TensorFlow Object Detection API v2


__[Follow these steps](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md)__


> [!NOTE]
> To use `protoc` command, [install it](https://github.com/protocolbuffers/protobuf/releases) - scroll down to assets

> [!IMPORTANT]
> Python 3.8 is strongly recommended, use conda or another tool to create a python environment for this

> [!TIP]
> Replace the last install command in the guide with `python -m pip install .`

## Prepare Data

* Choose dataset what model should recognize
* Make annotation (label), [recommend to use Label Studio](https://github.com/HumanSignal/label-studio)
* Export annotation in choosen format (xml, csv, tfrecord etc)

> [!NOTE]
> To convert images and annotation file into tfrecord use __[generate_tfrecords.py](https://github.com/MrZend/tensorflow-v2-API/blob/main/generate_tfrecords.py)__ script and command below
> ```
> python generate_tfrecords.py \
>     --path_to_images=${PATH_TO_IMAGES} \
>     --path_to_annot=${PATH_TO_ANNOTATION} \
>     --path_to_label_map=${PATH_TO_LABEL_MAP} \
>     --path_to_save_tfrecords=${PATH_TO_SAVE_TFRECORD}
> ```

## Pretrained model

* Choose and download [model](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)
* Set up pipeline configuration (has the same name as model)

## Training and Evaluating model

> [!IMPORTANT]
> Run commands into `$PATH/models/research`

### Training
```
python object_detection/model_main_tf2.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --alsologtostderr
```

### Evaluating
```
python object_detection/model_main_tf2.py \ 
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --checkpoint_dir=${CHECKPOINT_DIR} \
    --alsologtostderr
```
> [!TIP]
> `tensorboard` can be used to see graphics, metrics and more information about trainings

## Export and Convert the model

### Export
> [!NOTE]
> [problem](https://stackoverflow.com/questions/72201667/tensorflow-convert-from-pb-to-tflite-failes-due-to-ops-error)


```
python object_detection/exporter_main_v2.py \
    --input_type image_tensor \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_dir=${TRAINED_CHECKPOINT_DIR} \
    --output_directory=${OUTPUT_DIRECTORY}
```

### Convert

```
tflite_convert \
    --saved_model_dir=${OUTPUT_DIRECTORY} \
    --output_file=${NAME_OF_THE_FILE} \
    --input_shapes=1,320,320,3 \
    --input_arrays=normalized_input_image_tensor \
    --output_arrays='TFLite_Detection_PostProcess,TFLite_Detection_PostProcess:1,TFLite_Detection_PostProcess:2,TFLite_Detection_PostProcess:3' \
    --inference_type=QUANTIZED_UINT8 \
    --mean_values=128 \
    --std_dev_values=128 \
    --allow_custom_ops
```

## Other
__In this repo you can see the result of steps above. The *ssd_mobilenet_v2_fpnlite_320.tflite* can be used in mobile or low perfomance machines to recognise tanks__

## Recomandation
I Recommend course at __Udemy__ called Deep learning for object detection using Tensorflow 2
