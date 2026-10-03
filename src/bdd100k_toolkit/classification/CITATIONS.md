# Citations for the classification models

If you use one of the classification models of this project, please cite its paper (or the
Ultralytics software entry where no paper exists), the `timm` library for the models trained
with the `timm` backend, and the BDD100K dataset paper (see the main README).

| Model | Cite |
|---|---|
| YOLOv8n-cls | Ultralytics YOLOv8 (software) |
| YOLO11n-cls | Ultralytics YOLO11 (software) |
| YOLO26n-cls | YOLO26 paper |
| MobileNetV4-Conv-Small | MobileNetV4 paper, plus timm |
| EfficientViT-B0 | EfficientViT paper, plus timm |
| ConvNeXt-Atto, ConvNeXt-Tiny | ConvNeXt paper, plus timm |
| ResNet-18 | ResNet paper, plus timm |

## YOLOv8

```bibtex
@software{yolov8_ultralytics,
  author = {Glenn Jocher and Ayush Chaurasia and Jing Qiu},
  title = {Ultralytics YOLOv8},
  version = {8.0.0},
  year = {2023},
  url = {https://github.com/ultralytics/ultralytics},
  orcid = {0000-0001-5950-6979, 0000-0002-7603-6750, 0000-0003-3783-7069},
  license = {AGPL-3.0}
}
```

## YOLO11

```bibtex
@software{yolo11_ultralytics,
  author = {Glenn Jocher and Jing Qiu},
  title = {Ultralytics YOLO11},
  version = {11.0.0},
  year = {2024},
  url = {https://github.com/ultralytics/ultralytics},
  orcid = {0000-0001-5950-6979, 0000-0003-3783-7069},
  license = {AGPL-3.0}
}
```

## YOLO26

```bibtex
@misc{jocher2026ultralyticsyolo26unifiedrealtime,
  title = {Ultralytics {YOLO26}: Unified Real-Time End-to-End Vision Models},
  author = {Glenn Jocher and Jing Qiu and Mengyu Liu and Shuai Lyu and Fatih Cagatay Akyon and Muhammet Esat Kalfaoglu},
  year = {2026},
  eprint = {2606.03748},
  archivePrefix = {arXiv},
  primaryClass = {cs.CV},
  doi = {10.48550/arXiv.2606.03748},
  url = {https://arxiv.org/abs/2606.03748}
}
```

## MobileNetV4

```bibtex
@article{qin2024mobilenetv4,
  title = {{MobileNetV4}: Universal Models for the Mobile Ecosystem},
  author = {Qin, Danfeng and Leichner, Chas and Delakis, Manolis and Fornoni, Marco and Luo, Shixin and Yang, Fan and Wang, Weijun and Banbury, Colby and Ye, Chengxi and Akin, Berkin and Aggarwal, Vaibhav and Zhu, Tenghui and Moro, Daniele and Howard, Andrew},
  journal = {arXiv preprint arXiv:2404.10518},
  year = {2024},
  url = {https://arxiv.org/abs/2404.10518}
}
```

## EfficientViT

```bibtex
@inproceedings{cai2023efficientvit,
  title = {{EfficientViT}: Multi-Scale Linear Attention for High-Resolution Dense Prediction},
  author = {Cai, Han and Li, Junyan and Hu, Muyan and Gan, Chuang and Han, Song},
  booktitle = {IEEE/CVF International Conference on Computer Vision (ICCV)},
  year = {2023},
  url = {https://arxiv.org/abs/2205.14756}
}
```

## ConvNeXt

```bibtex
@inproceedings{liu2022convnet,
  title = {A {ConvNet} for the 2020s},
  author = {Liu, Zhuang and Mao, Hanzi and Wu, Chao-Yuan and Feichtenhofer, Christoph and Darrell, Trevor and Xie, Saining},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2022},
  url = {https://arxiv.org/abs/2201.03545}
}
```

## ResNet

```bibtex
@inproceedings{he2016deep,
  title = {Deep Residual Learning for Image Recognition},
  author = {He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2016},
  url = {https://arxiv.org/abs/1512.03385}
}
```

## timm

```bibtex
@misc{rw2019timm,
  author = {Ross Wightman},
  title = {PyTorch Image Models},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  doi = {10.5281/zenodo.4414861},
  howpublished = {\url{https://github.com/rwightman/pytorch-image-models}}
}
```
