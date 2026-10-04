# BDD100K-Toolkit

[![CI](https://github.com/dronefreak/bdd100k-toolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/dronefreak/bdd100k-toolkit/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)](pyproject.toml)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: BSD-3-Clause](https://img.shields.io/badge/license-BSD--3--Clause-green)](LICENSE)
[![Maintained](https://img.shields.io/badge/maintained-yes-brightgreen)](https://github.com/dronefreak/bdd100k-toolkit/commits)
[![Backends: Ultralytics, timm](https://img.shields.io/badge/backends-Ultralytics%20%7C%20timm-informational)](src/bdd100k_toolkit/classification)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-261230)](https://github.com/astral-sh/ruff)

![BDD100K detector and weather, period and scenario classifiers running on two city drives, side by side](assets/demo.gif)

An unofficial, modern toolkit for the [BDD100K dataset](https://arxiv.org/abs/1805.04687). It prepares the data, trains and evaluates models with current libraries (PyTorch, Ultralytics, timm, Hydra), and renders demos on images and videos. The official [toolkit](https://github.com/bdd100k/bdd100k) and [model zoo](https://github.com/SysCV/bdd100k-models) are unmaintained and no longer install cleanly, so this project rebuilds the parts that matter, one task at a time, each checked on real data. The goal is a set of reproducible baselines for others to build on.

This repository currently supports the tasks listed below. For more information about each task, please click on the individual task's name.

- [**Image Classification**](src/bdd100k_toolkit/classification): weather, time of day and scenario (unofficial tasks), 37 models available in the model zoo (1300+ architectures in `timm` and 12+ in `ultralytics`)
- [**Object Detection**](src/bdd100k_toolkit/detection): 10 classes, 10 models available in the model zoo (30+ architecures in `ultralytics` and 5+ in `rfdetr`)
- [**Semantic Segmentation**](src/bdd100k_toolkit/segmentation): in progress
- [**Object Tracking**](src/bdd100k_toolkit/tracking): planned

## Datasets

The official BDD100K website (bdd100k.com, bdd-data.berkeley.edu) is down and has been unreachable, so the datasets below are unofficial redistributions on Hugging Face, which the BDD100K license allows as long as the copyright notice is carried forward.

To use this toolkit, download the datasets from Hugging Face (all with the original 2018 labels):

| Task | Dataset |
|---|---|
| Detection | 🤗 [dronefreak/BDD100K](https://huggingface.co/datasets/dronefreak/BDD100K) |
| Weather | 🤗 [dronefreak/BDD100K-Weather-Classification](https://huggingface.co/datasets/dronefreak/BDD100K-Weather-Classification) |
| Period | 🤗 [dronefreak/BDD100K-Period-Classification](https://huggingface.co/datasets/dronefreak/BDD100K-Period-Classification) |
| Scenario | 🤗 [dronefreak/BDD100K-Scenario-Classification](https://huggingface.co/datasets/dronefreak/BDD100K-Scenario-Classification) |

## Roadmap

- [x] Classification: weather, period, scenario (with image and video demos)
- [x] Object detection data pipeline and baselines
- [ ] Semantic segmentation
- [ ] Instance segmentation
- [ ] Panoptic segmentation
- [ ] Object tracking (MOT)
- [ ] ONNX/TensorRT export

## License

BSD-3-Clause for the code. BDD100K data, including the Hugging Face copy, stays under the [BDD100K License](https://github.com/bdd100k/bdd100k/blob/master/doc/source/license.rst): free for educational, research and not-for-profit use with the UC Regents copyright notice carried forward; commercial use needs separate permission.

## Citation

To cite the BDD100K dataset in your paper,

```latex
@InProceedings{bdd100k,
    author = {Yu, Fisher and Chen, Haofeng and Wang, Xin and Xian, Wenqi and Chen,
              Yingying and Liu, Fangchen and Madhavan, Vashisht and Darrell, Trevor},
    title = {BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning},
    booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2020}
}
```

## References

- [BDD100K toolkit](https://github.com/bdd100k/bdd100k): the original toolkit
- [BDD100K model zoo](https://github.com/SysCV/bdd100k-models): the original model zoo
- [timm](https://github.com/huggingface/pytorch-image-models): classification backend
- [Ultralytics](https://github.com/ultralytics/ultralytics): YOLO classification and detection backend
- [RF-DETR](https://github.com/roboflow/rf-detr): transformer detection backend
