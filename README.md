## FlowNet2 (PyTorch v0.3.0)

Pytorch implementation of [FlowNet 2.0: Evolution of Optical Flow Estimation with Deep Networks](https://arxiv.org/abs/1612.01925). Most part are from this [repo](https://github.com/NVIDIA/flownet2-pytorch), we made it as a off-the-shelf package:
- After installation, just copy the whole folder FlowNet2_src to your codebase to use. See demo.py for details.

### Environment

This code has been test with Python3.6 and PyTorch0.3.0, with a Tesla K80 GPU. The system is Ubuntu 14.04.

### Installation 

    # install custom layers
    cd FlowNet2_src
    bash install.sh

Note: you might need to modify [here](https://github.com/vt-vl-lab/pytorch_flownet2/blob/master/FlowNet2_src/models/components/ops/channelnorm/make.sh#L10), [here](https://github.com/vt-vl-lab/pytorch_flownet2/blob/master/FlowNet2_src/models/components/ops/correlation/make.sh#L12), and [here](https://github.com/vt-vl-lab/pytorch_flownet2/blob/master/FlowNet2_src/models/components/ops/resample2d/make.sh#L10), according to the GPU you use.

### Converted Caffe Pre-trained Models
* [FlowNet2](https://drive.google.com/file/d/1hF8vS6YeHkx3j2pfCeQqqZGwA_PJq_Da/view?usp=sharing)[620MB]
* [FlowNet2-C](https://drive.google.com/file/d/1BFT6b7KgKJC8rA59RmOVAXRM_S7aSfKE/view?usp=sharing)[149MB]
* [FlowNet2-CS](https://drive.google.com/file/d/1iBJ1_o7PloaINpa8m7u_7TsLCX0Dt_jS/view?usp=sharing)[297MB]
* [FlowNet2-CSS](https://drive.google.com/file/d/157zuzVf4YMN6ABAQgZc8rRmR5cgWzSu8/view?usp=sharing)[445MB]
* [FlowNet2-CSS-ft-sd](https://drive.google.com/file/d/1R5xafCIzJCXc8ia4TGfC65irmTNiMg6u/view?usp=sharing)[445MB]
* [FlowNet2-S](https://drive.google.com/file/d/1V61dZjFomwlynwlYklJHC-TLfdFom3Lg/view?usp=sharing)[148MB]
* [FlowNet2-SD](https://drive.google.com/file/d/1QW03eyYG_vD-dT-Mx4wopYvtPu_msTKn/view?usp=sharing)[173MB]

### Inference mode
First download pre-trained models of FlowNet2 and modify the path, then

```
python demo.py
```    

If installation is sucessful, you should see the following:
![FlowNet2 Sample Prediction](/FlowNet2_src/example/flow0.png?raw=true)
   
### Reference 
If you find this implementation useful in your work, please acknowledge it appropriately and cite the paper using:
````
@InProceedings{IMKDB17,
  author       = "E. Ilg and N. Mayer and T. Saikia and M. Keuper and A. Dosovitskiy and T. Brox",
  title        = "FlowNet 2.0: Evolution of Optical Flow Estimation with Deep Networks",
  booktitle    = "IEEE Conference on Computer Vision and Pattern Recognition (CVPR)",
  month        = "Jul",
  year         = "2017",
  url          = "http://lmb.informatik.uni-freiburg.de//Publications/2017/IMKDB17"
}
````

### Acknowledgments
* [NVIDIA/flownet2-pytorch](https://github.com/NVIDIA/flownet2-pytorch): Most part
* [hellock/flownet2-pytorch](https://github.com/hellock/flownet2-pytorch): Python3.x and PyTorch0.3.0 support
