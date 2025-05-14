import torch
import numpy as np
import torch.nn.functional as F
from torch import Tensor
from pytorch_wavelets import *
from typing import List, Tuple
import math
from layers.utils import get_filter
from torch import nn, einsum, diagonal


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class WaveletF1DModule(nn.Module):
    def __init__(self,N,M, wavelet='sym2',L=1):
        super(WaveletF1DModule, self).__init__()
        self.DWT = DWT1D(wave=wavelet,mode='reflect')
        self.IDWT = IDWT1D(wave=wavelet,mode='reflect')
        self.L=L
        self.ns = min(math.floor(np.log2(N)),L)
        self.nl = pow(2, math.ceil(np.log2(N)))
        self.model = nn.ModuleList()
        for i in range(self.ns):
            model_in = nn.ModuleList()
            model_in.append(MsparseKernelFT1d(M, self.nl // (pow(2, i)+2)+1))
            model_in.append(MsparseKernelFT1d(M, self.nl // (pow(2, i)+2)+1))
            model_in.append(MsparseKernelFT1d(M, self.nl // (pow(2, i)+2)+1))#self.nl // pow(2, i)+5
            self.model.append(model_in)
        self.D = MsparseKernelFT1d(M,self.nl//pow(2, self.ns-1)+2*self.ns+3)
            #model_in = nn.ModuleList()
            #model_in.append(sparseKernelWa1d(M, self.nl // pow(2, i)+2*i+5))
            #model_in.append(sparseKernelWa1d(M, self.nl // pow(2, i)+2*i+5))
            #model_in.append(sparseKernelWa1d(M, self.nl // pow(2, i)+2*i+5))
            #self.model.append(model_in)
        #self.D = sparseKernelWa1d(M,self.nl)
    def forward(self,x):
        B, N, M = x.shape  # (B, N, k)
        ns= min(math.floor(np.log2(N)),self.L)
        nl = pow(2, math.ceil(np.log2(N)))
        extra_x = x[:, 0:nl - N, :]
        x = (torch.cat([x, extra_x], 1)).permute(0, 2, 1)
        Ud = torch.jit.annotate(List[Tensor], [])
        Us = torch.jit.annotate(List[Tensor], [])
        Us += [x]
        for i in range(ns):
            x,d = self.DWT(x)
            Ud += [self.model[i][0](d[0]) + self.model[i][1](x)]
            Us += [self.model[i][2](x)]
        x = self.D(x)  # coarsest scale transform
        for i in range(ns - 1 , -1, -1):
            x =x[:,:,:Us[i+1].shape[-1]]
            x = x + Us[i+1]
            x = self.IDWT((x,[Ud[i]]))
        x = x.permute(0,2,1)[:, :N, :]
        return x

class MultiWaveletTransform(nn.Module):
    """
    1D multiwavelet block.
    """

    def __init__(self,N, M=1,Tr_model=1024,
                 nCZ=1, wavelet='sym2'):
        super(MultiWaveletTransform, self).__init__()
        print('base', wavelet)
        self.nCZ = nCZ
        self.Lk0 = nn.Linear(M, Tr_model)
        self.Lk1 = nn.Linear(Tr_model, M)
        self.MWT_CZ = nn.ModuleList(WaveletF1DModule(N,Tr_model,wavelet=wavelet) for i in range(nCZ))

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        if L > S:
            zeros = torch.zeros_like(queries[:, :(L - S), :]).float()
            values = torch.cat([values, zeros], dim=1)
            keys = torch.cat([keys, zeros], dim=1)
        else:
            values = values[:, :L, :, :]
            keys = keys[:, :L, :, :]
        V = values.view(B, L, -1)

        V = self.Lk0(V)
        for i in range(self.nCZ):
            V = self.MWT_CZ[i](V)
            if i < self.nCZ - 1:
                V = F.relu(V)

        V = self.Lk1(V.view(B, L, -1))
        V = V.view(B, L, -1, D)
        return (V.contiguous(), None)

class MsparseKernelFT1d(nn.Module):
    def __init__(self,M,N,model=32,
                 **kwargs):
        super(MsparseKernelFT1d, self).__init__()
        self.scale = (1 / (N * N))
        self.modes1 = model
        self.weights1 = nn.Parameter(self.scale * torch.rand(M, N//2+1, N//2+1, dtype=torch.cfloat))
        self.weights1.requires_grad = True

    def compl_mul1d(self, x, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bmn,mnl->bml", x, weights)

    def forward(self, x):
        B, M, N = x.shape  #
        x_fft = torch.fft.rfft(x)
        # Multiply relevant Fourier modes
        l = min(self.modes1, N // 2 +1)
        out_ft = torch.zeros(B, M, N // 2 +1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :l] = self.compl_mul1d(x_fft[:, :, :l], self.weights1[:, :l, :l])
        x = torch.fft.irfft(out_ft, n=N)
        return x


class MultiWaveletCross(nn.Module):
    """
    1D Multiwavelet Cross Attention layer.
    """

    def __init__(self, in_channels, out_channels, seq_len_q, seq_len_kv, modes, c=64,
                 k=8, ich=512,
                 L=0,
                 base='legendre',
                 mode_select_method='random',
                 initializer=None, activation='tanh',
                 **kwargs):
        super(MultiWaveletCross, self).__init__()
        print('base', base)

        self.c = c
        self.k = k
        self.L = L
        H0, H1, G0, G1, PHI0, PHI1 = get_filter(base, k)
        H0r = H0 @ PHI0
        G0r = G0 @ PHI0
        H1r = H1 @ PHI1
        G1r = G1 @ PHI1

        H0r[np.abs(H0r) < 1e-8] = 0
        H1r[np.abs(H1r) < 1e-8] = 0
        G0r[np.abs(G0r) < 1e-8] = 0
        G1r[np.abs(G1r) < 1e-8] = 0
        self.max_item = 3

        self.attn1 = FourierCrossAttentionW(in_channels=in_channels, out_channels=out_channels, seq_len_q=seq_len_q,
                                            seq_len_kv=seq_len_kv, modes=modes, activation=activation,
                                            mode_select_method=mode_select_method)
        self.attn2 = FourierCrossAttentionW(in_channels=in_channels, out_channels=out_channels, seq_len_q=seq_len_q,
                                            seq_len_kv=seq_len_kv, modes=modes, activation=activation,
                                            mode_select_method=mode_select_method)
        self.attn3 = FourierCrossAttentionW(in_channels=in_channels, out_channels=out_channels, seq_len_q=seq_len_q,
                                            seq_len_kv=seq_len_kv, modes=modes, activation=activation,
                                            mode_select_method=mode_select_method)
        self.attn4 = FourierCrossAttentionW(in_channels=in_channels, out_channels=out_channels, seq_len_q=seq_len_q,
                                            seq_len_kv=seq_len_kv, modes=modes, activation=activation,
                                            mode_select_method=mode_select_method)
        self.T0 = nn.Linear(k, k)
        self.register_buffer('ec_s', torch.Tensor(
            np.concatenate((H0.T, H1.T), axis=0)))
        self.register_buffer('ec_d', torch.Tensor(
            np.concatenate((G0.T, G1.T), axis=0)))

        self.register_buffer('rc_e', torch.Tensor(
            np.concatenate((H0r, G0r), axis=0)))
        self.register_buffer('rc_o', torch.Tensor(
            np.concatenate((H1r, G1r), axis=0)))

        self.Lk = nn.Linear(ich, c * k)
        self.Lq = nn.Linear(ich, c * k)
        self.Lv = nn.Linear(ich, c * k)
        self.out = nn.Linear(c * k, ich)
        self.modes1 = modes

    def forward(self, q, k, v, mask=None):
        B, N, H, E = q.shape  # (B, N, H, E) torch.Size([3, 768, 8, 2])
        _, S, _, _ = k.shape  # (B, S, H, E) torch.Size([3, 96, 8, 2])

        q = q.view(q.shape[0], q.shape[1], -1)
        k = k.view(k.shape[0], k.shape[1], -1)
        v = v.view(v.shape[0], v.shape[1], -1)
        q = self.Lq(q)
        q = q.view(q.shape[0], q.shape[1], self.c, self.k)
        k = self.Lk(k)
        k = k.view(k.shape[0], k.shape[1], self.c, self.k)
        v = self.Lv(v)
        v = v.view(v.shape[0], v.shape[1], self.c, self.k)

        if N > S:
            zeros = torch.zeros_like(q[:, :(N - S), :]).float()
            v = torch.cat([v, zeros], dim=1)
            k = torch.cat([k, zeros], dim=1)
        else:
            v = v[:, :N, :, :]
            k = k[:, :N, :, :]

        ns = math.floor(np.log2(N))
        nl = pow(2, math.ceil(np.log2(N)))
        extra_q = q[:, 0:nl - N, :, :]
        extra_k = k[:, 0:nl - N, :, :]
        extra_v = v[:, 0:nl - N, :, :]
        q = torch.cat([q, extra_q], 1)
        k = torch.cat([k, extra_k], 1)
        v = torch.cat([v, extra_v], 1)

        Ud_q = torch.jit.annotate(List[Tuple[Tensor]], [])
        Ud_k = torch.jit.annotate(List[Tuple[Tensor]], [])
        Ud_v = torch.jit.annotate(List[Tuple[Tensor]], [])

        Us_q = torch.jit.annotate(List[Tensor], [])
        Us_k = torch.jit.annotate(List[Tensor], [])
        Us_v = torch.jit.annotate(List[Tensor], [])

        Ud = torch.jit.annotate(List[Tensor], [])
        Us = torch.jit.annotate(List[Tensor], [])

        # decompose
        for i in range(ns - self.L):
            # print('q shape',q.shape)
            d, q = self.wavelet_transform(q)
            Ud_q += [tuple([d, q])]
            Us_q += [d]
        for i in range(ns - self.L):
            d, k = self.wavelet_transform(k)
            Ud_k += [tuple([d, k])]
            Us_k += [d]
        for i in range(ns - self.L):
            d, v = self.wavelet_transform(v)
            Ud_v += [tuple([d, v])]
            Us_v += [d]
        for i in range(ns - self.L):
            dk, sk = Ud_k[i], Us_k[i]
            dq, sq = Ud_q[i], Us_q[i]
            dv, sv = Ud_v[i], Us_v[i]
            Ud += [self.attn1(dq[0], dk[0], dv[0], mask)[0] + self.attn2(dq[1], dk[1], dv[1], mask)[0]]
            Us += [self.attn3(sq, sk, sv, mask)[0]]
        v = self.attn4(q, k, v, mask)[0]

        # reconstruct
        for i in range(ns - 1 - self.L, -1, -1):
            v = v + Us[i]
            v = torch.cat((v, Ud[i]), -1)
            v = self.evenOdd(v)
        v = self.out(v[:, :N, :, :].contiguous().view(B, N, -1))
        return (v.contiguous(), None)

    def wavelet_transform(self, x):
        xa = torch.cat([x[:, ::2, :, :],
                        x[:, 1::2, :, :],
                        ], -1)
        d = torch.matmul(xa, self.ec_d)
        s = torch.matmul(xa, self.ec_s)
        return d, s

    def evenOdd(self, x):
        B, N, c, ich = x.shape  # (B, N, c, k)
        assert ich == 2 * self.k
        x_e = torch.matmul(x, self.rc_e)
        x_o = torch.matmul(x, self.rc_o)

        x = torch.zeros(B, N * 2, c, self.k,
                        device=x.device)
        x[..., ::2, :, :] = x_e
        x[..., 1::2, :, :] = x_o
        return x


class FourierCrossAttentionW(nn.Module):
    def __init__(self, in_channels, out_channels, seq_len_q, seq_len_kv, modes=16, activation='tanh',
                 mode_select_method='random'):
        super(FourierCrossAttentionW, self).__init__()
        print('corss fourier correlation used!')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes
        self.activation = activation

    def forward(self, q, k, v, mask):
        B, L, E, H = q.shape

        xq = q.permute(0, 3, 2, 1)  # size = [B, H, E, L] torch.Size([3, 8, 64, 512])
        xk = k.permute(0, 3, 2, 1)
        xv = v.permute(0, 3, 2, 1)
        self.index_q = list(range(0, min(int(L // 2), self.modes1)))
        self.index_k_v = list(range(0, min(int(xv.shape[3] // 2), self.modes1)))

        # Compute Fourier coefficients
        xq_ft_ = torch.zeros(B, H, E, len(self.index_q), device=xq.device, dtype=torch.cfloat)
        xq_ft = torch.fft.rfft(xq, dim=-1)
        for i, j in enumerate(self.index_q):
            xq_ft_[:, :, :, i] = xq_ft[:, :, :, j]

        xk_ft_ = torch.zeros(B, H, E, len(self.index_k_v), device=xq.device, dtype=torch.cfloat)
        xk_ft = torch.fft.rfft(xk, dim=-1)
        for i, j in enumerate(self.index_k_v):
            xk_ft_[:, :, :, i] = xk_ft[:, :, :, j]

        xqk_ft = (torch.einsum("bhex,bhey->bhxy", xq_ft_, xk_ft_))
        if self.activation == 'tanh':
            xqk_ft = xqk_ft.tanh()
        elif self.activation == 'softmax':
            xqk_ft = torch.softmax(abs(xqk_ft), dim=-1)
            xqk_ft = torch.complex(xqk_ft, torch.zeros_like(xqk_ft))
        else:
            raise Exception('{} actiation function is not implemented'.format(self.activation))

        xqkv_ft = torch.einsum("bhxy,bhey->bhex", xqk_ft, xk_ft_)

        xqkvw = xqkv_ft
        out_ft = torch.zeros(B, H, E, L // 2 + 1, device=xq.device, dtype=torch.cfloat)
        for i, j in enumerate(self.index_q):
            out_ft[:, :, :, j] = xqkvw[:, :, :, i]

        out = torch.fft.irfft(out_ft / self.in_channels / self.out_channels, n=xq.size(-1)).permute(0, 3, 2, 1)
        # size = [B, L, H, E]
        return (out, None)


class sparseKernelFT1d(nn.Module):
    def __init__(self,
                 k, alpha, c=1,
                 nl=1,
                 initializer=None,
                 **kwargs):
        super(sparseKernelFT1d, self).__init__()

        self.modes1 = alpha
        self.scale = (1 / (c * k * c * k))
        self.weights1 = nn.Parameter(self.scale * torch.rand(c * k, c * k, self.modes1, dtype=torch.cfloat))
        self.weights1.requires_grad = True
        self.k = k

    def compl_mul1d(self, x, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", x, weights)

    def forward(self, x):
        B, N, c, k = x.shape  # (B, N, c, k)

        x = x.view(B, N, -1)
        x = x.permute(0, 2, 1)#(B, c * k, N)
        x_fft = torch.fft.rfft(x)
        # Multiply relevant Fourier modes
        l = min(self.modes1, N // 2 + 1)
        # l = N//2+1
        out_ft = torch.zeros(B, c * k, N // 2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :l] = self.compl_mul1d(x_fft[:, :, :l], self.weights1[:, :, :l])
        x = torch.fft.irfft(out_ft, n=N)
        x = x.permute(0, 2, 1).view(B, N, c, k)
        return x

class sparseKernelWa1d(nn.Module):
    def __init__(self, M, L, modes1=10000,wavelet='bior2.2',
                 **kwargs):
        super(sparseKernelWa1d, self).__init__()
        '''
        给小波系数加权重
        '''
        self.modes1 = modes1
        self.L = L
        self.scale = (1 / (L*L))
        self.weights1 = nn.Parameter(self.scale * torch.rand(M, L//2+1, L//2+1, dtype=torch.float32,device=device))
        self.weights1.requires_grad = True
        self.weights2 = nn.Parameter(self.scale * torch.rand(M, L//2+1, L//2+1, dtype=torch.float32,device=device))
        self.weights2.requires_grad = True

        self.DWT=DWT1D(wave=wavelet,mode='periodization')
        self.IDWT=IDWT1D(wave=wavelet,mode='periodization')

    def compl_mul1d(self, x, weights):
        return torch.einsum('bnm,nml->bnl', x, weights)

    def forward(self, x):
        B, M, N = x.shape
        x,d = self.DWT(x)
        d = d[0]
        l = x.shape[-1]
        out_ft_x = torch.zeros(B, M, l, device=x.device, dtype=torch.float32)
        out_ft_x[:, :, :l] = self.compl_mul1d(x[:, :, :l], self.weights1[:, :l, :l])
        out_ft_d = torch.zeros(B, M, l, device=x.device, dtype=torch.float32)
        out_ft_d[:, :, :l] = self.compl_mul1d(d[:, :, :l], self.weights2[:, :l, :l])
        out_ft = self.IDWT((out_ft_x,[out_ft_d]))[:,:,:N]
        return out_ft

