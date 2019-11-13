import torch
import librosa
from torchnmf import NMF
from torchnmf.metrics import KL_divergence

device = torch.device('cuda:1')
y, sr = librosa.load(librosa.util.example_audio_file())
y = torch.from_numpy(y).to(device)
windowsize = 2048
S = torch.stft(y, windowsize, window=torch.hann_window(windowsize, device=device)).pow(2).sum(2).sqrt().cuda(1)

R = 8   # number of components

from sklearn.decomposition import NMF
model = NMF(n_components=R, init='random', random_state=0)
W = model.fit_transform(S)
H = model.components_
print(KL_divergence(torch.mm(W,H), S))        # KL divergence to S

# net = NMF(S.shape, rank=R).cuda(1)
# # run extremely fast on gpu
# _, V = net.fit_transform(S)      # fit to target matrix S
# print(KL_divergence(V, S))        # KL divergence to S