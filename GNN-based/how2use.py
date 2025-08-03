import os
from config import *
from torch import nn
from scipy.ndimage.filters import gaussian_filter1d
from torch.autograd import Variable
import torch
import numpy as np
import eval_utils as utils


palmtree = utils.UsableTransformer(model_path="./palmtree/transformer.ep19", vocab_path="./palmtree/vocab")

# tokens has to be seperated by spaces.

text = ["mov rbp rdi", 
        "mov ebx 0x1", 
        "mov rdx rbx", 
        "call memcpy", 
        "mov [ rcx + rbx ] 0x0", 
        "mov rcx rax", 
        "mov [ rax ] 0x2e"]

text1 = ["mov rbp rdi", 
        "mov ebx 0x1", 
        "mov rdx rbx", 
        "call memcpy", 
        "mov [ rcx + rbx ] 0x0", 
        "mov rcx rax"]


# it is better to make batches as large as possible.
embeddings = palmtree.encode(text)
#print("usable embedding of this basicblock:", embeddings)
print("the shape of output tensor: ", embeddings.shape)
print("the len of embedding: ", len(embeddings))

emb_trn = []

i = 0
for emb in embeddings:
  #print(f"embedding[{i}]: {embeddings[i]}")
  print(f"embedding[{i}]/10: {emb[:10]}")
  emb_trn = emb_trn + list(emb[:10])
  i = i + 1

print (f"output: {emb_trn}")

#embeddings1 = palmtree.encode(text1)
#print("usable embedding of this basicblock:", embeddings1)
#print("the shape of output tensor: ", embeddings1.shape)
