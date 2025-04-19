# 测试cuda是否可用
import torch
import os
print(torch.cuda.is_available())
print(os.environ.get('CUDA_VISIBLE_DEVICES'))

# 测试faiss是否可用
import faiss
print(faiss.get_num_gpus())
