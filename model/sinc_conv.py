import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.fft
import sys
from torch.autograd import Variable
import math

class GlobalLayerNorm(nn.Module):
    '''
       Calculate Global Layer Normalization
       dim: (int or list or torch.Size) –
            input shape from an expected input of size
       eps: a value added to the denominator for numerical stability.
       elementwise_affine: a boolean value that when set to True, 
           this module has learnable per-element affine parameters 
           initialized to ones (for weights) and zeros (for biases).
    '''

    def __init__(self, dim, eps=1e-05, elementwise_affine=True):
        super(GlobalLayerNorm, self).__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.dim, 1))
            self.bias = nn.Parameter(torch.zeros(self.dim, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        # x = N x C x L
        # N x 1 x 1
        # cln: mean,var N x 1 x L
        # gln: mean,var N x 1 x 1
        if x.dim() != 3:
            raise RuntimeError("{} accept 3D tensor as input".format(
                self.__name__))

        mean = torch.mean(x, (1, 2), keepdim=True)
        var = torch.mean((x-mean)**2, (1, 2), keepdim=True)
        # N x C x L
        if self.elementwise_affine:
            x = self.weight*(x-mean)/torch.sqrt(var+self.eps)+self.bias
        else:
            x = (x-mean)/torch.sqrt(var+self.eps)
        return x

    
class TimeSincExtractor(nn.Module):
    """Sinc-based convolution
    Parameters
    ----------
    in_channels : `int`
        Number of input channels. Must be 1.
    out_channels : `int`
        Number of filters.
    kernel_size : `int`
        Filter length.
    sample_rate : `int`, optional
        Sample rate. Defaults to 16000.
    triangular : `bool`
        Squared sinc -> Triangular filter.
    freq_nml : `bool`
        Normalized to gain of 1 in frequency.
    range_constraint : `bool`
        Project the learned band within nyquist freq manually.
    Usage
    -----
    See `torch.nn.Conv1d`
    """

    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)
    
    def swap_(self, x, y, sort=False):
        mini = torch.minimum(x, y)
        maxi = torch.maximum(x, y)
        
        if sort:
            mini, idx = torch.sort(mini)
            maxi = maxi[idx].view(mini.shape)
        
        return mini, maxi

    def __init__(self, out_channels, kernel_size, triangular=False, 
                 freq_nml=False, range_constraint=False, freq_init='uniform', norm_after=True, sample_rate=16000, in_channels=1,
                 stride=1, padding=0, dilation=1, bias=False, groups=1, min_low_hz=50, min_band_hz=50, bi_factor=False, frame_length=400, hop_length=160):

        super(TimeSincExtractor,self).__init__()

        if in_channels != 1:
            # msg = (f'SincConv only support one input channel '
            #       f'(here, in_channels = {in_channels:d}).')
            msg = "SincConv only support one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.triangular = False
        self.freq_nml = False
        
        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size%2 == 0:
            self.kernel_size = self.kernel_size+1
            
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        self.frame_length = frame_length
        self.hop_length = hop_length

        if bias:
            raise ValueError('SincConv does not support bias.')
        if groups > 1:
            raise ValueError('SincConv does not support groups.')

        self.sample_rate = sample_rate
        self.nyquist_rate = sample_rate/2
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz
        self.range_constraint = range_constraint
        self.bi_factor = bi_factor

        if self.range_constraint:
            # msg = "Range constraint in learned frequency is not supported yet."
            # raise ValueError(msg)
            if freq_init == "uniform":
                low_freq, high_freq = torch.rand(out_channels*2).chunk(2)
            elif freq_init == "formant":
                # raise NotImplementedError('Formant distribution hasn\'t been implemented yet.')
                p = np.load('/share/nas165/Jasonho610/SincNet/exp/formant_distribution.npy')
                low_freq, high_freq = torch.from_numpy(np.random.choice(8000, out_channels*2, p=p)).chunk(2)
                low_freq = low_freq / self.nyquist_rate
                high_freq = high_freq / self.nyquist_rate
            elif freq_init == "mel":
                # raise NotImplementedError('Mel distribution hasn\'t been implemented yet.')
                low_hz = 30
                high_hz = self.nyquist_rate - (self.min_low_hz + self.min_band_hz)
                mel = np.linspace(self.to_mel(low_hz),
                                  self.to_mel(high_hz),
                                  self.out_channels + 1)
                hz = self.to_hz(mel)
                low_freq = torch.Tensor(hz[:-1]) / self.nyquist_rate
                high_freq = torch.Tensor(hz[1:]) / self.nyquist_rate
            else: 
                raise ValueError('SincConv must specify the freq initialization methods.')
                
            low_freq, high_freq = self.swap_(low_freq, high_freq)
            
            if self.bi_factor:
                self.band_imp = nn.Parameter(torch.ones(out_channels))
            self.low_f_ = nn.Parameter(low_freq.view(-1, 1))
            self.high_f_ = nn.Parameter(high_freq.view(-1, 1))            
        else:
            # initialize filterbanks such that they are equally spaced in Mel scale
            low_hz = 30
            high_hz = self.nyquist_rate - (self.min_low_hz + self.min_band_hz)
            mel = np.linspace(self.to_mel(low_hz),
                              self.to_mel(high_hz),
                              self.out_channels + 1)
            hz = self.to_hz(mel)
            # filter lower frequency (out_channels, 1)
            self.low_hz_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))

            # filter frequency band (out_channels, 1)
            self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))

        # Hamming window
        # self.window_ = torch.hamming_window(self.kernel_size)
        n_lin = torch.linspace(0, (self.kernel_size/2)-1, steps=int((self.kernel_size/2))) # computing only half of the window
        self.window_ = 0.54-0.46*torch.cos(2*math.pi*n_lin/self.kernel_size);

        # (1, kernel_size/2)
        n = (self.kernel_size - 1) / 2.0
        self.n_ = 2*math.pi*torch.arange(-n, 0).view(1, -1) / self.sample_rate # Due to symmetry, I only need half of the time axes
        
        self.norm_after = norm_after
        if self.norm_after:
            self.ln = GlobalLayerNorm(out_channels)


    def forward(self, waveforms, embedding):
        """
        Parameters
        ----------
        waveforms : `torch.Tensor` (batch_size, 1, n_samples)
            Batch of waveforms.
        Returns
        -------
        features : `torch.Tensor` (batch_size, out_channels, n_samples_out)
            Batch of sinc filters activations.
        """

        self.n_ = self.n_.to(waveforms.device)
        self.window_ = self.window_.to(waveforms.device)
        # waveforms = waveforms.unsqueeze(1)
        # print("Waveforms:", waveforms.shape)
        
        framing_padding = self.frame_length - (waveforms.shape[-1] % self.hop_length)
        waveforms = F.pad(waveforms, (0, framing_padding))
        frames = waveforms.unfold(-1, self.frame_length, self.hop_length)
        
        batch_size = frames.shape[0]
        n_frames = frames.shape[2]
        
        if self.range_constraint:
            low_f_, high_f_ = self.swap_(torch.abs(self.low_f_), torch.abs(self.high_f_))
            
            low  = self.min_low_hz + low_f_*self.nyquist_rate
            high = torch.clamp(self.min_band_hz + high_f_*self.nyquist_rate, self.min_low_hz, self.nyquist_rate)
            band = (high-low)[:,0]
        else:
            low  = self.min_low_hz + torch.abs(self.low_hz_)
            high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_), self.min_low_hz, self.nyquist_rate)
            band = (high-low)[:,0]
            
        self.low = low
        self.high = high
        self.band = band
        
        f_times_t_low = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)

        band_pass_left = ((torch.sin(f_times_t_high)-torch.sin(f_times_t_low))/(self.n_/2))*self.window_ # Equivalent of Eq.4 of the reference paper (SPEAKER RECOGNITION FROM RAW WAVEFORM WITH SINCNET). I just have expanded the sinc and simplified the terms. This way I avoid several useless computations. 
        band_pass_center = 2*band.view(-1,1)
        band_pass_right = torch.flip(band_pass_left,dims=[1])
        
        band_pass = torch.cat([band_pass_left,band_pass_center,band_pass_right],dim=1)
        
        band_pass = band_pass / (2*band[:,None])
        
        if self.triangular:
            band_pass = band_pass**2
            
        if self.freq_nml:
            mag_resp = torch.fft.rfft(band_pass).abs()
            mag_max = torch.max(mag_resp, dim=-1)[0]
            band_pass = band_pass / mag_max.unsqueeze(-1)
            
        if self.bi_factor:
            band_imp = F.relu(self.band_imp)
            band_pass = band_pass * band_imp.unsqueeze(-1)
            
        
        self.filters = (band_pass).view(
            self.out_channels, 1, self.kernel_size)
        
        # print("Filters:", self.filters.shape)
        # print("Frames:", frames.shape)
        
        rs_frames = frames.reshape(batch_size*n_frames, 1, self.frame_length)
        # print("Reshaped frames:", rs_frames.shape)
        
        filtered = F.conv1d(rs_frames, self.filters, stride=self.stride,
                        padding=self.padding, dilation=self.dilation,
                        bias=None, groups=1)
        # print('Pass conv1d')
        # print("Filtered:", filtered.shape)
        if self.norm_after:
            filtered = self.ln(filtered)
            
        # print("Normed filtered:", filtered.shape)
        
        filtered = filtered.reshape(batch_size, n_frames, self.out_channels , -1)
        
        # print("Final filtered:", filtered.shape)
         
        energy = torch.mean(filtered**2, dim=-1)
        log_filtered_energy = torch.log10(energy + 1e-6)
        # print("Log filtered energy:", log_filtered_energy.shape)  # (batch_size, n_samples_out(time), out_channels(frequency))

        log_filtered_energy = log_filtered_energy.unsqueeze(1)
        # print("Unsqueezed log filtered energy:", log_filtered_energy.shape)  # (batch_size, channels, n_samples_out(time), out_channels(frequency))

        log_filtered_energy = log_filtered_energy.permute(0, 1, 3, 2)
        # print("Permuted log filtered energy:", log_filtered_energy.shape)  # (batch_size, channels, out_channels(frequency), n_samples_out(time))
    
        return log_filtered_energy, self.filters, self.stride, self.padding


class FreqSincExtractor(nn.Module):
    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)
    
    def swap_(self, x, y, sort=False):
        mini = torch.minimum(x, y)
        maxi = torch.maximum(x, y)
        if sort:
            mini, idx = torch.sort(mini)
            maxi = maxi[idx].view(mini.shape)
        return mini, maxi

    def __init__(self, out_channels, kernel_size, triangular=False, 
                 freq_nml=False, range_constraint=False, freq_init='uniform',
                 norm_after=True, sample_rate=16000, in_channels=1,
                 stride=1, padding=0, dilation=1, bias=False, groups=1,
                 min_low_hz=50, min_band_hz=50, bi_factor=False,
                 frame_length=400, hop_length=160, n_fft=400):
        super(FreqSincExtractor, self).__init__()
        
        if in_channels != 1:
            msg = "FreqSincExtractor only supports one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.triangular = triangular
        self.freq_nml = freq_nml
        self.sample_rate = sample_rate
        self.nyquist_rate = sample_rate/2
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz
        self.range_constraint = range_constraint
        self.bi_factor = bi_factor
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.stride = stride
        self.padding = padding
        self.output_size = 64

        # Initialize frequency bands
        if self.range_constraint:
            if freq_init == "uniform":
                low_freq, high_freq = torch.rand(out_channels*2).chunk(2)
            elif freq_init == "mel":
                low_hz = 30
                high_hz = self.nyquist_rate - (self.min_low_hz + self.min_band_hz)
                mel = np.linspace(self.to_mel(low_hz),
                                self.to_mel(high_hz),
                                self.out_channels + 1)
                hz = self.to_hz(mel)
                low_freq = torch.Tensor(hz[:-1]) / self.nyquist_rate
                high_freq = torch.Tensor(hz[1:]) / self.nyquist_rate
            else:
                raise ValueError('FreqSincExtractor must specify the freq initialization methods.')
                
            low_freq, high_freq = self.swap_(low_freq, high_freq)
            
            if self.bi_factor:
                self.band_imp = nn.Parameter(torch.ones(out_channels))
            self.low_f_ = nn.Parameter(low_freq.view(-1, 1))
            self.high_f_ = nn.Parameter(high_freq.view(-1, 1))
        else:
            low_hz = 30
            high_hz = self.nyquist_rate - (self.min_low_hz + self.min_band_hz)
            mel = np.linspace(self.to_mel(low_hz),
                            self.to_mel(high_hz),
                            self.out_channels + 1)
            hz = self.to_hz(mel)
            self.low_hz_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))
            self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))

        # Frequency axis for STFT
        self.freq_axis = torch.linspace(0, self.nyquist_rate, self.n_fft//2 + 1)
        
        self.norm_after = norm_after
        if self.norm_after:
            self.ln = GlobalLayerNorm(out_channels)

    def get_filters(self):
        if self.range_constraint:
            low_f_, high_f_ = self.swap_(torch.abs(self.low_f_), torch.abs(self.high_f_))
            low = self.min_low_hz + low_f_ * self.nyquist_rate
            high = torch.clamp(self.min_low_hz + high_f_ * self.nyquist_rate,
                             self.min_low_hz, self.nyquist_rate)
        else:
            low = self.min_low_hz + torch.abs(self.low_hz_)
            high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_),
                             self.min_low_hz, self.nyquist_rate)

        # Create frequency domain filters
        freq_axis = self.freq_axis.to(low.device)
        filters = torch.zeros((self.out_channels, len(freq_axis))).to(low.device)
        
        for i in range(self.out_channels):
            mask = (freq_axis >= low[i]) & (freq_axis <= high[i])
            filters[i, mask] = 1.0
            
            if self.triangular:
                center_freq = (low[i] + high[i]) / 2
                bandwidth = high[i] - low[i]
                mask = (freq_axis >= low[i]) & (freq_axis <= high[i])
                freq_response = 1.0 - torch.abs(freq_axis[mask] - center_freq) / (bandwidth/2)
                filters[i, mask] = freq_response

        if self.freq_nml:
            filters = F.normalize(filters, p=2, dim=1)
            
        if self.bi_factor:
            band_imp = F.relu(self.band_imp)
            filters = filters * band_imp.unsqueeze(-1)
            
        return filters

    def forward(self, waveforms, embedding=None):
        batch_size = waveforms.shape[0]
        
        # Calculate necessary padding to achieve the correct output size
        target_length = self.hop_length * (self.output_size - 1) + self.frame_length
        current_length = waveforms.shape[-1]
        padding_needed = target_length - current_length
        
        # Pad the input if necessary
        if padding_needed > 0:
            waveforms = F.pad(waveforms, (0, padding_needed))
        
        # Compute STFT
        stft = torch.stft(waveforms.squeeze(1), 
                        n_fft=self.n_fft,
                        hop_length=self.hop_length,
                        win_length=self.frame_length,
                        window=torch.hann_window(self.frame_length).to(waveforms.device),
                        return_complex=True)
        
        # Get magnitude spectrogram
        mag_spec = torch.abs(stft)  # (batch_size, freq_bins, time_frames)
        
        # Get and apply filters
        filters = self.get_filters()  # (out_channels, freq_bins)
        filtered = torch.matmul(filters, mag_spec)  # (batch_size, out_channels, time_frames)
        
        if self.norm_after:
            filtered = self.ln(filtered)
        
        # Compute log energy
        energy = filtered ** 2
        log_energy = torch.log10(energy + 1e-6)
        
        # Ensure correct time dimension
        if log_energy.shape[-1] != self.output_size:
            log_energy = F.interpolate(
                log_energy,
                size=self.output_size,
                mode='linear',
                align_corners=False
            )
        
        # Reshape to the desired output format
        log_energy = log_energy.unsqueeze(1)  # Add channel dimension
        log_energy = log_energy.permute(0, 1, 3, 2)  # Rearrange to (batch, channel, freq, time)
        
        return log_energy, filters, self.stride, self.padding
