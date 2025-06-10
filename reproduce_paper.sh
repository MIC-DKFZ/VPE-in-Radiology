#!/bin/bash

# Processing BiomedCLIPModel on PadChestGRTrain
python3 src/process.py --model_name BiomedCLIPModel --dataset_name PadChestGRTrain
python3 src/process.py --model_name BiomedCLIPModel --dataset_name PadChestGRTrain --image_annotation_type crop
python3 src/process.py --model_name BiomedCLIPModel --dataset_name PadChestGRTrain --image_annotation_type arrow --color red
python3 src/process.py --model_name BiomedCLIPModel --dataset_name PadChestGRTrain --image_annotation_type arrow --color red --text_annotation_suffix ' indicated by a red arrow'
python3 src/process.py --model_name BiomedCLIPModel --dataset_name PadChestGRTrain --image_annotation_type bbox --color red
python3 src/process.py --model_name BiomedCLIPModel --dataset_name PadChestGRTrain --image_annotation_type bbox --color red --text_annotation_suffix ' indicated by a red bounding box'
python3 src/process.py --model_name BiomedCLIPModel --dataset_name PadChestGRTrain --image_annotation_type circle --color red
python3 src/process.py --model_name BiomedCLIPModel --dataset_name PadChestGRTrain --image_annotation_type circle --color red --text_annotation_suffix ' indicated by a red circle'


# Processing BMC_CLIP_CF_Model on PadChestGRTrain
python3 src/process.py --model_name BMC_CLIP_CF_Model --dataset_name PadChestGRTrain
python3 src/process.py --model_name BMC_CLIP_CF_Model --dataset_name PadChestGRTrain --image_annotation_type crop
python3 src/process.py --model_name BMC_CLIP_CF_Model --dataset_name PadChestGRTrain --image_annotation_type arrow --color red
python3 src/process.py --model_name BMC_CLIP_CF_Model --dataset_name PadChestGRTrain --image_annotation_type arrow --color red --text_annotation_suffix ' indicated by a red arrow'
python3 src/process.py --model_name BMC_CLIP_CF_Model --dataset_name PadChestGRTrain --image_annotation_type bbox --color red
python3 src/process.py --model_name BMC_CLIP_CF_Model --dataset_name PadChestGRTrain --image_annotation_type bbox --color red --text_annotation_suffix ' indicated by a red bounding box'
python3 src/process.py --model_name BMC_CLIP_CF_Model --dataset_name PadChestGRTrain --image_annotation_type circle --color red
python3 src/process.py --model_name BMC_CLIP_CF_Model --dataset_name PadChestGRTrain --image_annotation_type circle --color red --text_annotation_suffix ' indicated by a red circle'


# Processing BiomedCLIPModel on PadChestGRVal
python3 src/process.py --model_name BiomedCLIPModel --dataset_name PadChestGRVal
python3 src/process.py --model_name BiomedCLIPModel --dataset_name PadChestGRVal --image_annotation_type crop
python3 src/process.py --model_name BiomedCLIPModel --dataset_name PadChestGRVal --image_annotation_type arrow --color red
python3 src/process.py --model_name BiomedCLIPModel --dataset_name PadChestGRVal --image_annotation_type arrow --color red --text_annotation_suffix ' indicated by a red arrow'
python3 src/process.py --model_name BiomedCLIPModel --dataset_name PadChestGRVal --image_annotation_type bbox --color red
python3 src/process.py --model_name BiomedCLIPModel --dataset_name PadChestGRVal --image_annotation_type bbox --color red --text_annotation_suffix ' indicated by a red bounding box'
python3 src/process.py --model_name BiomedCLIPModel --dataset_name PadChestGRVal --image_annotation_type circle --color red
python3 src/process.py --model_name BiomedCLIPModel --dataset_name PadChestGRVal --image_annotation_type circle --color red --text_annotation_suffix ' indicated by a red circle'


# Processing BMC_CLIP_CF_Model on PadChestGRVal
python3 src/process.py --model_name BMC_CLIP_CF_Model --dataset_name PadChestGRVal
python3 src/process.py --model_name BMC_CLIP_CF_Model --dataset_name PadChestGRVal --image_annotation_type crop
python3 src/process.py --model_name BMC_CLIP_CF_Model --dataset_name PadChestGRVal --image_annotation_type arrow --color red
python3 src/process.py --model_name BMC_CLIP_CF_Model --dataset_name PadChestGRVal --image_annotation_type arrow --color red --text_annotation_suffix ' indicated by a red arrow'
python3 src/process.py --model_name BMC_CLIP_CF_Model --dataset_name PadChestGRVal --image_annotation_type bbox --color red
python3 src/process.py --model_name BMC_CLIP_CF_Model --dataset_name PadChestGRVal --image_annotation_type bbox --color red --text_annotation_suffix ' indicated by a red bounding box'
python3 src/process.py --model_name BMC_CLIP_CF_Model --dataset_name PadChestGRVal --image_annotation_type circle --color red
python3 src/process.py --model_name BMC_CLIP_CF_Model --dataset_name PadChestGRVal --image_annotation_type circle --color red --text_annotation_suffix ' indicated by a red circle'


# Processing BiomedCLIPModel on PadChestGRTest
python3 src/process.py --model_name BiomedCLIPModel --dataset_name PadChestGRTest
python3 src/process.py --model_name BiomedCLIPModel --dataset_name PadChestGRTest --image_annotation_type crop
python3 src/process.py --model_name BiomedCLIPModel --dataset_name PadChestGRTest --image_annotation_type arrow --color red
python3 src/process.py --model_name BiomedCLIPModel --dataset_name PadChestGRTest --image_annotation_type arrow --color red --text_annotation_suffix ' indicated by a red arrow'
python3 src/process.py --model_name BiomedCLIPModel --dataset_name PadChestGRTest --image_annotation_type bbox --color red
python3 src/process.py --model_name BiomedCLIPModel --dataset_name PadChestGRTest --image_annotation_type bbox --color red --text_annotation_suffix ' indicated by a red bounding box'
python3 src/process.py --model_name BiomedCLIPModel --dataset_name PadChestGRTest --image_annotation_type circle --color red
python3 src/process.py --model_name BiomedCLIPModel --dataset_name PadChestGRTest --image_annotation_type circle --color red --text_annotation_suffix ' indicated by a red circle'


# Processing BMC_CLIP_CF_Model on PadChestGRTest
python3 src/process.py --model_name BMC_CLIP_CF_Model --dataset_name PadChestGRTest
python3 src/process.py --model_name BMC_CLIP_CF_Model --dataset_name PadChestGRTest --image_annotation_type crop
python3 src/process.py --model_name BMC_CLIP_CF_Model --dataset_name PadChestGRTest --image_annotation_type arrow --color red
python3 src/process.py --model_name BMC_CLIP_CF_Model --dataset_name PadChestGRTest --image_annotation_type arrow --color red --text_annotation_suffix ' indicated by a red arrow'
python3 src/process.py --model_name BMC_CLIP_CF_Model --dataset_name PadChestGRTest --image_annotation_type bbox --color red
python3 src/process.py --model_name BMC_CLIP_CF_Model --dataset_name PadChestGRTest --image_annotation_type bbox --color red --text_annotation_suffix ' indicated by a red bounding box'
python3 src/process.py --model_name BMC_CLIP_CF_Model --dataset_name PadChestGRTest --image_annotation_type circle --color red
python3 src/process.py --model_name BMC_CLIP_CF_Model --dataset_name PadChestGRTest --image_annotation_type circle --color red --text_annotation_suffix ' indicated by a red circle'


# Processing BiomedCLIPModel on VinDrCXRTrain
python3 src/process.py --model_name BiomedCLIPModel --dataset_name VinDrCXRTrain
python3 src/process.py --model_name BiomedCLIPModel --dataset_name VinDrCXRTrain --image_annotation_type crop
python3 src/process.py --model_name BiomedCLIPModel --dataset_name VinDrCXRTrain --image_annotation_type arrow --color red
python3 src/process.py --model_name BiomedCLIPModel --dataset_name VinDrCXRTrain --image_annotation_type arrow --color red --text_annotation_suffix ' indicated by a red arrow'
python3 src/process.py --model_name BiomedCLIPModel --dataset_name VinDrCXRTrain --image_annotation_type bbox --color red
python3 src/process.py --model_name BiomedCLIPModel --dataset_name VinDrCXRTrain --image_annotation_type bbox --color red --text_annotation_suffix ' indicated by a red bounding box'
python3 src/process.py --model_name BiomedCLIPModel --dataset_name VinDrCXRTrain --image_annotation_type circle --color red
python3 src/process.py --model_name BiomedCLIPModel --dataset_name VinDrCXRTrain --image_annotation_type circle --color red --text_annotation_suffix ' indicated by a red circle'


# Processing BMC_CLIP_CF_Model on VinDrCXRTrain
python3 src/process.py --model_name BMC_CLIP_CF_Model --dataset_name VinDrCXRTrain
python3 src/process.py --model_name BMC_CLIP_CF_Model --dataset_name VinDrCXRTrain --image_annotation_type crop
python3 src/process.py --model_name BMC_CLIP_CF_Model --dataset_name VinDrCXRTrain --image_annotation_type arrow --color red
python3 src/process.py --model_name BMC_CLIP_CF_Model --dataset_name VinDrCXRTrain --image_annotation_type arrow --color red --text_annotation_suffix ' indicated by a red arrow'
python3 src/process.py --model_name BMC_CLIP_CF_Model --dataset_name VinDrCXRTrain --image_annotation_type bbox --color red
python3 src/process.py --model_name BMC_CLIP_CF_Model --dataset_name VinDrCXRTrain --image_annotation_type bbox --color red --text_annotation_suffix ' indicated by a red bounding box'
python3 src/process.py --model_name BMC_CLIP_CF_Model --dataset_name VinDrCXRTrain --image_annotation_type circle --color red
python3 src/process.py --model_name BMC_CLIP_CF_Model --dataset_name VinDrCXRTrain --image_annotation_type circle --color red --text_annotation_suffix ' indicated by a red circle'


# Processing BiomedCLIPModel on VinDrCXRTest
python3 src/process.py --model_name BiomedCLIPModel --dataset_name VinDrCXRTest
python3 src/process.py --model_name BiomedCLIPModel --dataset_name VinDrCXRTest --image_annotation_type crop
python3 src/process.py --model_name BiomedCLIPModel --dataset_name VinDrCXRTest --image_annotation_type arrow --color red
python3 src/process.py --model_name BiomedCLIPModel --dataset_name VinDrCXRTest --image_annotation_type arrow --color red --text_annotation_suffix ' indicated by a red arrow'
python3 src/process.py --model_name BiomedCLIPModel --dataset_name VinDrCXRTest --image_annotation_type bbox --color red
python3 src/process.py --model_name BiomedCLIPModel --dataset_name VinDrCXRTest --image_annotation_type bbox --color red --text_annotation_suffix ' indicated by a red bounding box'
python3 src/process.py --model_name BiomedCLIPModel --dataset_name VinDrCXRTest --image_annotation_type circle --color red
python3 src/process.py --model_name BiomedCLIPModel --dataset_name VinDrCXRTest --image_annotation_type circle --color red --text_annotation_suffix ' indicated by a red circle'


# Processing BMC_CLIP_CF_Model on VinDrCXRTest
python3 src/process.py --model_name BMC_CLIP_CF_Model --dataset_name VinDrCXRTest
python3 src/process.py --model_name BMC_CLIP_CF_Model --dataset_name VinDrCXRTest --image_annotation_type crop
python3 src/process.py --model_name BMC_CLIP_CF_Model --dataset_name VinDrCXRTest --image_annotation_type arrow --color red
python3 src/process.py --model_name BMC_CLIP_CF_Model --dataset_name VinDrCXRTest --image_annotation_type arrow --color red --text_annotation_suffix ' indicated by a red arrow'
python3 src/process.py --model_name BMC_CLIP_CF_Model --dataset_name VinDrCXRTest --image_annotation_type bbox --color red
python3 src/process.py --model_name BMC_CLIP_CF_Model --dataset_name VinDrCXRTest --image_annotation_type bbox --color red --text_annotation_suffix ' indicated by a red bounding box'
python3 src/process.py --model_name BMC_CLIP_CF_Model --dataset_name VinDrCXRTest --image_annotation_type circle --color red
python3 src/process.py --model_name BMC_CLIP_CF_Model --dataset_name VinDrCXRTest --image_annotation_type circle --color red --text_annotation_suffix ' indicated by a red circle'


# Processing BiomedCLIPModel on NIH14
python3 src/process.py --model_name BiomedCLIPModel --dataset_name NIH14
python3 src/process.py --model_name BiomedCLIPModel --dataset_name NIH14 --image_annotation_type crop
python3 src/process.py --model_name BiomedCLIPModel --dataset_name NIH14 --image_annotation_type arrow --color red
python3 src/process.py --model_name BiomedCLIPModel --dataset_name NIH14 --image_annotation_type arrow --color red --text_annotation_suffix ' indicated by a red arrow'
python3 src/process.py --model_name BiomedCLIPModel --dataset_name NIH14 --image_annotation_type bbox --color red
python3 src/process.py --model_name BiomedCLIPModel --dataset_name NIH14 --image_annotation_type bbox --color red --text_annotation_suffix ' indicated by a red bounding box'
python3 src/process.py --model_name BiomedCLIPModel --dataset_name NIH14 --image_annotation_type circle --color red
python3 src/process.py --model_name BiomedCLIPModel --dataset_name NIH14 --image_annotation_type circle --color red --text_annotation_suffix ' indicated by a red circle'


# Processing BMC_CLIP_CF_Model on NIH14
python3 src/process.py --model_name BMC_CLIP_CF_Model --dataset_name NIH14
python3 src/process.py --model_name BMC_CLIP_CF_Model --dataset_name NIH14 --image_annotation_type crop
python3 src/process.py --model_name BMC_CLIP_CF_Model --dataset_name NIH14 --image_annotation_type arrow --color red
python3 src/process.py --model_name BMC_CLIP_CF_Model --dataset_name NIH14 --image_annotation_type arrow --color red --text_annotation_suffix ' indicated by a red arrow'
python3 src/process.py --model_name BMC_CLIP_CF_Model --dataset_name NIH14 --image_annotation_type bbox --color red
python3 src/process.py --model_name BMC_CLIP_CF_Model --dataset_name NIH14 --image_annotation_type bbox --color red --text_annotation_suffix ' indicated by a red bounding box'
python3 src/process.py --model_name BMC_CLIP_CF_Model --dataset_name NIH14 --image_annotation_type circle --color red
python3 src/process.py --model_name BMC_CLIP_CF_Model --dataset_name NIH14 --image_annotation_type circle --color red --text_annotation_suffix ' indicated by a red circle'


# Processing BiomedCLIPModel on JSRT
python3 src/process.py --model_name BiomedCLIPModel --dataset_name JSRT
python3 src/process.py --model_name BiomedCLIPModel --dataset_name JSRT --image_annotation_type crop
python3 src/process.py --model_name BiomedCLIPModel --dataset_name JSRT --image_annotation_type arrow --color red
python3 src/process.py --model_name BiomedCLIPModel --dataset_name JSRT --image_annotation_type arrow --color red --text_annotation_suffix ' indicated by a red arrow'
python3 src/process.py --model_name BiomedCLIPModel --dataset_name JSRT --image_annotation_type bbox --color red
python3 src/process.py --model_name BiomedCLIPModel --dataset_name JSRT --image_annotation_type bbox --color red --text_annotation_suffix ' indicated by a red bounding box'
python3 src/process.py --model_name BiomedCLIPModel --dataset_name JSRT --image_annotation_type circle --color red
python3 src/process.py --model_name BiomedCLIPModel --dataset_name JSRT --image_annotation_type circle --color red --text_annotation_suffix ' indicated by a red circle'


# Processing BMC_CLIP_CF_Model on JSRT
python3 src/process.py --model_name BMC_CLIP_CF_Model --dataset_name JSRT
python3 src/process.py --model_name BMC_CLIP_CF_Model --dataset_name JSRT --image_annotation_type crop
python3 src/process.py --model_name BMC_CLIP_CF_Model --dataset_name JSRT --image_annotation_type arrow --color red
python3 src/process.py --model_name BMC_CLIP_CF_Model --dataset_name JSRT --image_annotation_type arrow --color red --text_annotation_suffix ' indicated by a red arrow'
python3 src/process.py --model_name BMC_CLIP_CF_Model --dataset_name JSRT --image_annotation_type bbox --color red
python3 src/process.py --model_name BMC_CLIP_CF_Model --dataset_name JSRT --image_annotation_type bbox --color red --text_annotation_suffix ' indicated by a red bounding box'
python3 src/process.py --model_name BMC_CLIP_CF_Model --dataset_name JSRT --image_annotation_type circle --color red
python3 src/process.py --model_name BMC_CLIP_CF_Model --dataset_name JSRT --image_annotation_type circle --color red --text_annotation_suffix ' indicated by a red circle'

python3 src/utils/comb_results.py