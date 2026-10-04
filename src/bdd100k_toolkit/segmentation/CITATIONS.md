# Citations for the segmentation models

If you use one of the pretrained segmentation models listed in this project, please cite its paper,
and the BDD100K dataset paper (see the main README). The models were trained on Cityscapes by their
authors and are evaluated here zero-shot on BDD100K, so the Cityscapes paper applies too.

| Model | Cite |
|---|---|
| SegFormer B0 to B5 | SegFormer paper |
| Mask2Former Swin-T, Swin-S, Swin-L | Mask2Former paper |

## SegFormer

```bibtex
@inproceedings{xie2021segformer,
  title = {{SegFormer}: Simple and Efficient Design for Semantic Segmentation with Transformers},
  author = {Xie, Enze and Wang, Wenhai and Yu, Zhiding and Anandkumar, Anima and Alvarez, Jose M. and Luo, Ping},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year = {2021},
  eprint = {2105.15203},
  archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/2105.15203}
}
```

## Mask2Former

```bibtex
@inproceedings{cheng2022mask2former,
  title = {Masked-attention Mask Transformer for Universal Image Segmentation},
  author = {Cheng, Bowen and Misra, Ishan and Schwing, Alexander G. and Kirillov, Alexander and Girdhar, Rohit},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2022},
  eprint = {2112.01527},
  archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/2112.01527}
}
```

## Cityscapes

```bibtex
@inproceedings{cordts2016cityscapes,
  title = {The Cityscapes Dataset for Semantic Urban Scene Understanding},
  author = {Cordts, Marius and Omran, Mohamed and Ramos, Sebastian and Rehfeld, Timo and Enzweiler, Markus and Benenson, Rodrigo and Franke, Uwe and Roth, Stefan and Schiele, Bernt},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2016}
}
```
