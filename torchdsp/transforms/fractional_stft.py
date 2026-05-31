from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from torchdsp.functional.overlap_add import overlap_add


def fractional_stft(
    x: Tensor, 
    n_fft: int, 
    hop_length: int, 
    r: int, 
    window: Tensor
) -> Tensor:
    r"""Compute fractional Short-time Fourier Transform (STFT).

    b: batch_size
    l: audio_length
    t: n_frames
    n: frame_length
    f: freq_bins
    r: fractions
    
    Args:
        x: (b, l)

    Returns:
        out: (b, t, f*r)
    """

    # Enframe    
    N = n_fft
    x = F.pad(x, (N // 2, N // 2), mode="reflect")  # (b, l)
    x = x.unfold(dimension=-1, size=N, step=hop_length).contiguous()  # (b, t, n)
    x.mul_(window)  # (b, t, n)
    
    # Reserve space
    x = x.to(torch.complex64)  # (b, t, n)
    n = torch.arange(0, N, device=x.device)  # (n,)
    a = torch.exp(-1.j * 2 * math.pi / N * n / r)  # (n,)
    B, T, N = x.shape
    out = torch.zeros((B, T, N // 2 + 1, r), dtype=torch.complex64, device=x.device)  # (b, t, f, r)

    # Compute STFT for each fraction
    for i in range(r):    
        y = torch.fft.fft(x, dim=-1, norm="ortho") / math.sqrt(r)  # (b, t, f)
        out[:, :, :, i] = y[:, :, 0 : N // 2 + 1]  # (b, t, f, r)
        x.mul_(a)  # (b, t, n)

    out = rearrange(out, 'b t f r -> b t (f r)')  # (b, t, f*r)
    out = out[..., 0 : N * r // 2 + 1]  # (b, t, f*r)
    return out


def fractional_istft(
    x: Tensor, 
    n_fft: int, 
    hop_length: int, 
    r: int, 
    window: Tensor, 
    length=None
) -> Tensor:
    r"""Compute fractional inverse Short-time Fourier Transform (iSTFT).

    b: batch_size
    t: n_frames
    f: freq_bins
    r: fractions
    l: audio_length
    n: frame_length

    Args:
        x: (b, t, f*r)

    Returns:
        out: (b, l)
    """

    N = n_fft
    n = torch.arange(0, N, device=x.device)  # (n,)
    
    # Reserve space
    B, T = x.shape[0 : 2]
    out = torch.zeros((B, T, N), device=x.device)
    
    # Compute iSTFT for each fraction
    for i in range(r):
        y = x[:, :, i :: r]
        if i == 0:
            y_flip = torch.flip(y[..., 1 : -1], dims=[-1]).conj()
        else:
            y_flip = torch.flip(x[:, :, r - i :: r], dims=[-1]).conj()

        y = torch.cat([y, y_flip], dim=-1)  # (b, f)
        a = torch.exp(1.j * 2 * math.pi / N * n / r * i)  # (n, r)
        y = torch.fft.ifft(y, dim=-1, norm="ortho") / math.sqrt(r)  # (b, t, n, r)
        out.add_((y * a).real)
    
    # Overlap add
    out = overlap_add(
        x=out, 
        hop_length=hop_length, 
        window=window
    )  # (b, l)
    
    out = out[:, n_fft // 2 :]  # (b, l)
    
    if length is not None:
        out = out[:, 0 : length]  # (b, l)

    return out


if __name__ == '__main__':
    r"""
    b: batch_size
    l: audio_samples
    t: frames_num
    f: freq_bins
    """

    seed = 1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    n_fft = 2048
    hop_length = 512
    r = 16
    window = torch.hann_window(n_fft)

    # Data
    x = torch.randn(4, 48000)  # (b, l)

    # Analysis & synthesis
    h = fractional_stft(x, n_fft, hop_length, r, window)  # (b, t, f)
    x_hat = fractional_istft(h, n_fft, hop_length, r, window, x.shape[-1])  # (b, l)

    print(f"x (B, L): {x.shape}")
    print(f"h (B, T, F): {h.shape}")
    print(f"x_hat (B, L): {x_hat.shape}")
    print("Error: {}".format((x - x_hat).abs().mean()))