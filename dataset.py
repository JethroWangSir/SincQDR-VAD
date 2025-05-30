import json
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import random
import torch
import os
import pandas as pd
from torch.utils.data import DataLoader

from function.preprocessing import shift_perturbation, white_noise_perturbation, spec_augment, spec_cutout

class SCF(Dataset):
    def __init__(self, manifest_files, sample_duration=0.63, sample_rate=16000, n_fft=400, n_mels=64, win_length=400, hop_length=160, 
                augment=False, feature_extraction=True, add_noise=False, noise_csv=None, noise_audio_dir=None, noise_config_path=None):
        self.sample_duration = sample_duration
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.win_length = win_length
        self.hop_length = hop_length
        self.augment = augment
        self.feature_extraction = feature_extraction
        self.add_noise = add_noise
        self.noise_zero_power_count = 0

        # Load JSON files
        self.data = []
        for manifest_file in manifest_files:
            with open(manifest_file, 'r') as f:
                for line in f:
                    try:
                        manifest_entry = json.loads(line.strip())
                        audio_path = manifest_entry['audio_filepath']
                        duration = manifest_entry['duration']
                        offset = manifest_entry['offset']
                        label = 1 if manifest_entry['label'] == 'speech' else 0
                        self.data.append((audio_path, label, duration, offset))
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON: {e}")

        # Initialize noise-related components if noise addition is enabled
        if self.add_noise:
            if not all([noise_csv, noise_audio_dir, noise_config_path]):
                raise ValueError("noise_csv, noise_audio_dir, and noise_config_path must be provided when add_noise is True")
            
            self.noise_audio_dir = noise_audio_dir
            self.category = [
                'cat', 'water_drops', 'footsteps', 'washing_machine', 'train', 'hen', 'wind', 
                'laughing', 'vacuum_cleaner', 'church_bells', 'insects', 'pouring_water', 
                'brushing_teeth', 'clock_alarm', 'airplane', 'sheep', 'toilet_flush', 'snoring', 
                'clock_tick', 'fireworks', 'crow', 'thunderstorm', 'drinking_sipping', 
                'glass_breaking', 'hand_saw'
            ]
            
            # Load and filter noise data
            self.noise_data = pd.read_csv(noise_csv)
            self.noise_data = self.noise_data[self.noise_data['category'].isin(self.category)]

            # Load or initialize noise SNR map
            if os.path.exists(noise_config_path):
                with open(noise_config_path, 'r') as f:
                    self.noise_snr_map = json.load(f)
            else:
                self.noise_snr_map = self._initialize_noise_snr_map()
                with open(noise_config_path, 'w') as f:
                    json.dump(self.noise_snr_map, f)

        # Initialize audio processing components
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            win_length=self.win_length,
            hop_length=self.hop_length
        )
        self.log_mel_spectrogram = T.AmplitudeToDB()

        # Set time mask parameter based on sample duration
        if self.sample_duration == 0.63:
            self.time_mask_param = 25
            self.shift_range = 5.0
        elif self.sample_duration == 0.16:
            self.time_mask_param = 7
            self.shift_range = 2.0
        elif self.sample_duration == 0.032:
            self.time_mask_param = 2
            self.shift_range = 1.0

        self.timemask = T.TimeMasking(time_mask_param=self.time_mask_param)
        self.freqmask = T.FrequencyMasking(freq_mask_param=15)

    def _initialize_noise_snr_map(self):
        noise_snr_map = {}
        snr_values = [-10, -5, 0, 5, 10]
        for idx, (audio_path, _, _, _) in enumerate(self.data):
            noise_row = self.noise_data.sample(n=1).iloc[0]
            noise_filename = noise_row['filename']
            snr = random.choice(snr_values)
            noise_snr_map[idx] = {"noise_filename": noise_filename, "snr": snr}
        return noise_snr_map

    def _load_noise(self, noise_filename):
        noise_path = os.path.join(self.noise_audio_dir, noise_filename)
        noise_waveform, _ = torchaudio.load(noise_path)
        return noise_waveform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get noise information if noise addition is enabled
        if self.add_noise:
            noise_info = self.noise_snr_map.get(idx)
            noise_filename = noise_info["noise_filename"]
            snr = noise_info["snr"]

        audio_path, label, duration, offset = self.data[idx]
        audio_path = f'data/{audio_path}'
        frame_offset = int(offset * self.sample_rate)
        num_frames = int(duration * self.sample_rate)
        waveform, _ = torchaudio.load(audio_path, frame_offset=frame_offset, num_frames=num_frames)

        # Add noise if enabled
        if self.add_noise:
            noise_waveform = self._load_noise(noise_filename)
                
            # Ensure noise and clean speech have the same length
            if noise_waveform.size(1) < waveform.size(1):
                repeat_times = (waveform.size(1) // noise_waveform.size(1)) + 1
                noise_waveform = noise_waveform.repeat(1, repeat_times)[:, :waveform.size(1)]
            else:
                start = random.randint(0, noise_waveform.size(1) - waveform.size(1))
                noise_waveform = noise_waveform[:, start:start + waveform.size(1)]

            # Calculate SNR and mix
            clean_power = torch.norm(waveform) ** 2
            noise_power = torch.norm(noise_waveform) ** 2
            desired_noise_power = clean_power / (10 ** (snr / 10))
                
            if noise_power > 0:
                noise_scaling_factor = (desired_noise_power / noise_power) ** 0.5
                noise_waveform = noise_waveform * noise_scaling_factor
            else:
                self.noise_zero_power_count += 1
                noise_waveform = torch.zeros_like(waveform)

            waveform = waveform + noise_waveform

        # Apply augmentation if enabled
        if self.augment:
            if random.random() < 1.0:
                # Time shift perturbation
                waveform = shift_perturbation(waveform, -self.shift_range, self.shift_range)
                # Add white noise ranging from -90dB to -46dB
                waveform = white_noise_perturbation(waveform, -46, 46)
            
        # Convert to spectrogram if feature extraction is enabled
        if self.feature_extraction:
            mel_spec = self.mel_spectrogram(waveform)
            log_mel_spec = self.log_mel_spectrogram(mel_spec)

            if self.augment:
                length = torch.tensor([64] * 256)
                log_mel_spec = spec_augment(log_mel_spec, length, 1, 1, 15, self.time_mask_param, 0.0)
                log_mel_spec = spec_cutout(log_mel_spec, 1, self.time_mask_param, 15)

            return log_mel_spec, label
        else:
            return waveform, label

    def get_noise_zero_power_count(self):
        """
        Return the count of zero-power noise instances.
        """
        return self.noise_zero_power_count


class AVA(Dataset):
    def __init__(self, root_dir, max_duration=300.0, sample_duration=0.63, overlap=0.875, sample_rate=16000,
                n_fft=400, n_mels=64, win_length=400, hop_length=160, feature_extraction=True):
        self.root_dir = root_dir
        self.max_duration = max_duration
        self.sample_duration = sample_duration
        self.overlap = overlap
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.win_length = win_length
        self.hop_length = hop_length
        self.feature_extraction = feature_extraction
        self.audio_paths = []
        self.labels = []
        self.min_duration_samples = int(self.sample_duration * sample_rate)
        self.step_size = int(self.min_duration_samples * (1 - self.overlap))
        self.max_duration_samples = int(max_duration * sample_rate)

        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            win_length=self.win_length,
            hop_length=self.hop_length
        )
        self.log_mel_spectrogram = T.AmplitudeToDB()
        self._prepare_dataset()

    def _prepare_dataset(self):
        label_mapping = {
            'NO_SPEECH': 0,
            'CLEAN_SPEECH': 1,
            'SPEECH_WITH_MUSIC': 1,
            'SPEECH_WITH_NOISE': 1
        }

        for folder_name in os.listdir(self.root_dir):
            folder_path = os.path.join(self.root_dir, folder_name)
            if os.path.isdir(folder_path):
                label = label_mapping.get(folder_name, -1)
                for file_name in os.listdir(folder_path):
                    if file_name.endswith('.wav'):
                        audio_path = os.path.join(folder_path, file_name)
                        waveform, _ = torchaudio.load(audio_path)
                        if waveform.size(1) > self.max_duration_samples:
                            continue
                        self.audio_paths.append(os.path.join(folder_path, file_name))
                        self.labels.append(label)
    
    def _pad_audio(self, waveform, target_length):
        """Pad waveform to the target length with zeros."""
        pad_length = target_length - waveform.size(1)
        return torch.nn.functional.pad(waveform, (0, pad_length), mode='constant', value=0)

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        label = self.labels[idx]

        waveform, _ = torchaudio.load(audio_path)

        if waveform.size(1) < self.min_duration_samples:
            waveform = self._pad_audio(waveform, self.min_duration_samples)

        num_samples = waveform.size(1)
        segments = []

        start = 0
        while start + self.min_duration_samples <= num_samples:
            segment = waveform[:, start:start + self.min_duration_samples]

            if segment.size(1) < self.min_duration_samples:
                segment = self._pad_audio(segment, self.min_duration_samples)

            mel_spec = self.mel_spectrogram(segment)
            log_mel_spec = self.log_mel_spectrogram(mel_spec)

            segments.append((segment, log_mel_spec, label))

            start += self.step_size

        if start < num_samples:
            remainder = waveform[:, start:]

            if remainder.size(1) < self.min_duration_samples:
                remainder = self._pad_audio(remainder, self.min_duration_samples)
                
            mel_spec = self.mel_spectrogram(remainder)
            log_mel_spec = self.log_mel_spectrogram(mel_spec)

            segments.append((remainder, log_mel_spec, label))

        return tuple(segments)
