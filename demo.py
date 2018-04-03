import matplotlib
matplotlib.use('Agg')
import numpy as np
from scipy.misc import imread
import torch
from torch.autograd import Variable

from FlowNet2_src import FlowNet2
from FlowNet2_src import flow_to_image
import matplotlib.pyplot as plt


if __name__ == '__main__':
  # Prepare img pair
  im1 = imread('FlowNet2_src/example/0img0.ppm')
  im2 = imread('FlowNet2_src/example/0img1.ppm')
  # B x 3(RGB) x 2(pair) x H x W
  ims = np.array([[im1, im2]]).transpose((0, 4, 1, 2, 3)).astype(np.float32)
  ims = torch.from_numpy(ims)
  print(ims.size())
  ims_v = Variable(ims.cuda(), requires_grad=False)

  # Build model
  flownet2 = FlowNet2()
  path = 'FlowNet2_src/pretrained/FlowNet2_checkpoint.pth.tar'
  pretrained_dict = torch.load(path)['state_dict']
  model_dict = flownet2.state_dict()
  pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
  model_dict.update(pretrained_dict)
  flownet2.load_state_dict(model_dict)
  flownet2.cuda()

  pred_flow = flownet2(ims_v).cpu().data
  pred_flow = pred_flow[0].numpy().transpose((1,2,0))
  flow_im = flow_to_image(pred_flow)

  # Visualization
  plt.imshow(flow_im)
  plt.savefig('flow.png', bbox_inches='tight')
