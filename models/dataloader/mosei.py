# # 未测试.
# ######################################
# ## 数据文件夹下MiniImage文件夹的名字. ##
# ######################################
# MOSEI_DATA_PATH_stype = 'MOSEI'
# ######################################


# import torch
# import os.path as osp
# from PIL import Image

# from torch.utils.data import Dataset
# from torchvision import transforms
# import numpy as np
# from skimage import io
# import pandas as pd
# import os, sys

# sys.path.append("../../")
# sys.path.extend([os.path.join(root, stype) for root, dirs, _ in os.walk("../") for stype in dirs])

# from models.metrics import ROOT_PATH

# from __future__ import print_function
# import pickle
# from models.mosei_tokenize import tokenize, create_dict, sent_to_ix, cmumosei_2, cmumosei_7, pad_feature


# class MOSEI(Dataset):
#     def __init__(self, stype, args, token_to_ix=None):
#         super(MOSEI, self).__init__()
#         if stype not in ('train', 'val', 'test'):
#             raise(ValueError, 'stype must be one of (train, val, test)')
#         self.args = args

#         data_root_path = osp.join(args.data_path, f'{MOSEI_DATA_PATH_stype}')

#         word_file = os.path.join(data_root_path, stype + "_sentences.p")
#         audio_file = os.path.join(data_root_path, stype + "_mels.p")
#         # video_file = os.path.join(data_root_path, stype + "_r21d.p") # Didnt improve our results
#         video_file = os.path.join(data_root_path, stype + "_mels.p") # Dummy

#         y_s_file = os.path.join(data_root_path, stype + "_sentiment.p")
#         y_e_file = os.path.join(data_root_path, stype + "_emotion.p")

#         self.set = eval(stype.upper()+"_SET")
#         self.key_to_word = pickle.load(open(word_file, "rb"))
#         self.key_to_audio = pickle.load(open(audio_file, "rb"))
#         self.key_to_video = pickle.load(open(video_file, "rb"))

#         if args.task == "emotion":
#             self.key_to_label = pickle.load(open(y_e_file, "rb"))
#         if args.task == "sentiment":
#             self.key_to_label = pickle.load(open(y_s_file, "rb"))

#         for key in self.set:
#             if not (key in self.key_to_word and
#                     key in self.key_to_audio and
#                     key in self.key_to_video and
#                     key in self.key_to_label):
#                 self.set.remove(key)

#         # Creating embeddings and word indexes
#         self.key_to_sentence = tokenize(self.key_to_word)
#         if token_to_ix is not None:
#             self.token_to_ix = token_to_ix
#         else: # Train
#             self.token_to_ix, self.pretrained_emb = create_dict(self.key_to_sentence, data_root_path)
#         self.vocab_size = len(self.token_to_ix)

#         self.l_max_len = args.lang_seq_len
#         self.a_max_len = args.audio_seq_len
#         self.v_max_len = args.video_seq_len

#     def __getitem__(self, idx):
#         key = self.set[idx]
#         L = sent_to_ix(self.key_to_sentence[key], self.token_to_ix, max_token=self.l_max_len)
#         A = pad_feature(self.key_to_audio[key], self.a_max_len)
#         V = pad_feature(self.key_to_video[key], self.v_max_len)

#         y = np.array([])
#         Y = self.key_to_label[key]
#         if self.args.mosei_task == "sentiment_2":
#             c = cmumosei_2(Y)
#             y = np.array(c)
#         if self.args.mosei_task == "sentiment_7":
#             c = cmumosei_7(Y)
#             y = np.array(c)
#         if self.args.mosei_task == "emotion":
#             Y[Y > 0] = 1
#             y = Y

#         return key, torch.from_numpy(L), torch.from_numpy(A), torch.from_numpy(V).float(), torch.from_numpy(y)

#     def __len__(self):
#         return len(self.set)