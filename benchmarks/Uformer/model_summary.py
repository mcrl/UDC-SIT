import torch
import torch.nn as nn
import numpy as np
from torchsummary import summary
from model import Uformer
testBatch = 64
WARM_UP = 10
MEASURE = 35

model = Uformer(img_size=256,embed_dim=32,win_size=8,token_projection='linear',token_mlp='leff',
            depths=[1, 2, 8, 8, 2, 8, 8, 2, 1],modulator=True,dd_in=3)  
model = model.to('cuda')
model = torch.nn.DataParallel(model)

randIn = torch.rand(testBatch,3,256,256).to('cuda')

summary(model, (3,256,256))

starter, ender = torch.cuda.Event(enable_timing=True),   torch.cuda.Event(enable_timing=True)

with torch.no_grad():
  for cnt in range(WARM_UP):
      _ = model(randIn)

  torch.cuda.synchronize()
  starter.record()
  for cnt in range(MEASURE):
      _ = model(randIn)
  ender.record()
  torch.cuda.synchronize()

  curr_time = starter.elapsed_time(ender)/1000
  print('Final time:',(curr_time)/64)


