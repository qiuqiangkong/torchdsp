class LowbandFractionalStft(nn.Module):
    def __init__(self, sr: float, half_bandwidths: list[float], n_fft: int, hop_length: int):
        super().__init__()
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer("window", torch.hann_window(n_fft))

        N = n_fft
        M = len(half_bandwidths)
        r = (sr / 2) / Tensor(half_bandwidths)
        r = torch.clamp(torch.floor(r * 0.5), 1.)
        m = 1. / r

        k = torch.arange(0, N // 2 + 1)  # (n,)
        n = torch.arange(0, N)  # (n,)
        mkn = torch.einsum('m,k,n->mkn', m, k, n)  # (m, k/2, n)

        mkn0 = mkn[:, 0:1, :]
        mkn1 = mkn[:, 1:-1, :]
        mkn2 = mkn[:, -1:, :]
        mkn3 = - torch.flip(mkn1, dims=[1])
        mkn = torch.cat([mkn0, mkn1, mkn2, mkn3], dim=1)  # (m, k, n)
        
        w = torch.exp(-1.j * 2 * math.pi / N * mkn) / math.sqrt(N)  # (m, f, n)
        w /= torch.sqrt(r[:, None, None])

        self.register_buffer("w", w)

    def analysis(self, x: Tensor) -> Tensor:
        r"""
        b: batch_size
        k: n_bands
        l: audio_samples
        t: n_frames
        n: frame_samples

        Args:
            x: (b, m, l)

        Returns:
            out: (b, m, t, n)
        """
        N = self.n_fft
        x = F.pad(x, (N // 2, N // 2), mode="reflect")  # (b, m, l)
        x = x.unfold(dimension=-1, size=N, step=self.hop_length).contiguous()  # (b, m, t, n)
        x *= self.window
        out = torch.einsum('bmtn,mfn->bmtf', x, self.w)  # (b, m, t, f)
        return out

    def synthesis(self, x: Tensor, length: int) -> Tensor:
        r"""

        Args:
            x: (b, m, t, f)

        Returns:
            x: (b, m, l)
        """
        x = torch.einsum('bmtf,mfn->bmtn', x, self.w.conj())

        # Overlap add
        B = x.shape[0]
        x = rearrange(x, 'b k t n -> (b k) t n')
        out = overlap_add(x=x, hop_length=self.hop_length, window=self.window)  # (b, l)
        out = rearrange(out, '(b k) l -> b k l', b=B)
        out = out[..., self.n_fft // 2 :]  # (b, k, l)
        
        if length is not None:
            out = out[..., 0 : length]  # (b, k, l)

        return out


if __name__ == '__main__':

    sr = 48000
    n_bands = 112
    max_bandwidth = 390
    factor = sr // 400
    chunk_size = 16
    device = "cuda"

    n_fft = 16
    hop_length = 4

    banks = erb_linear_banks(sr=sr, n_bands=n_bands, max_bandwidth=max_bandwidth)
    sb_filter = SubbandFilter(sr, banks, factor, chunk_size=chunk_size).to(device)
    # print(banks)

    rs = np.random.RandomState(1234)
    audio = rs.uniform(low=-1, high=1, size=(4, 2, sr * 2))
    audio = Tensor(audio).to(device)  # (c, l)
            
    # Analysis
    x = sb_filter.analysis(audio)  # (b, c, k, l)
    pred_audio = sb_filter.synthesis(x)
    
    banks = [0, ]
    half_bandwidths = [(bank[1] - bank[0]) / 2 for bank in banks]
    part_stft = PartStft(400, half_bandwidths, n_fft, hop_length).to(device)
    
    tmp = rearrange(x, 'b c k l -> (b c) k l')
    tmp = part_stft.analysis(tmp)
    feat3 = rearrange(tmp, '(b c) k t f -> b c k t f', b=B)

    tmp = rearrange(feat3, 'b c k t f -> (b c) k t f')
    tmp = part_stft.synthesis(tmp, L)
    y3 = rearrange(tmp, '(b c) k l -> b c k l', b=B)
    pred_audio = sb_filter.synthesis(y3)
    sdr = fast_sdr(audio.cpu().numpy(), pred_audio.cpu().numpy())
    print((y3 - x).abs().mean(), sdr)