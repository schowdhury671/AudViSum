# -*- coding: utf-8 -*-
"""FeatureExtraction.ipynb

Automatically generated by Colaboratory.

"""

import numpy as np

import vggish_params
import mel_features
import librosa



# ==============================================================================

"""Compute input examples for VGGish from audio waveform."""

# Modification: Return torch tensors rather than numpy arrays
import torch

import numpy as np
import resampy

import soundfile as sf


def waveform_to_examples(data, sample_rate, return_tensor=True):
    """Converts audio waveform into an array of examples for VGGish.

  Args:
    data: np.array of either one dimension (mono) or two dimensions
      (multi-channel, with the outer dimension representing channels).
      Each sample is generally expected to lie in the range [-1.0, +1.0],
      although this is not required.
    sample_rate: Sample rate of data.
    return_tensor: Return data as a Pytorch tensor ready for VGGish

  Returns:
    3-D np.array of shape [num_examples, num_frames, num_bands] which represents
    a sequence of examples, each of which contains a patch of log mel
    spectrogram, covering num_frames frames of audio and num_bands mel frequency
    bands, where the frame length is vggish_params.STFT_HOP_LENGTH_SECONDS.

  """
    # Convert to mono.
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
    # Resample to the rate assumed by VGGish.
    if sample_rate != vggish_params.SAMPLE_RATE:
        print("ora",sample_rate,vggish_params.SAMPLE_RATE)
        data = resampy.resample(data, sample_rate, vggish_params.SAMPLE_RATE)

    # Compute log mel spectrogram features.
    log_mel = mel_features.log_mel_spectrogram(
        data,
        audio_sample_rate=vggish_params.SAMPLE_RATE,
        log_offset=vggish_params.LOG_OFFSET,
        window_length_secs=vggish_params.STFT_WINDOW_LENGTH_SECONDS,
        hop_length_secs=vggish_params.STFT_HOP_LENGTH_SECONDS,
        num_mel_bins=vggish_params.NUM_MEL_BINS,
        lower_edge_hertz=vggish_params.MEL_MIN_HZ,
        upper_edge_hertz=vggish_params.MEL_MAX_HZ)
    

    # Frame features into examples.
    features_sample_rate = 1.0 / vggish_params.STFT_HOP_LENGTH_SECONDS
    example_window_length = int(round(
        vggish_params.EXAMPLE_WINDOW_SECONDS * features_sample_rate))
    example_hop_length = int(round(
        vggish_params.EXAMPLE_HOP_SECONDS * features_sample_rate))
    log_mel_examples = mel_features.frame(
        log_mel,
        window_length=example_window_length,
        hop_length=example_hop_length)

    if return_tensor:
        log_mel_examples = torch.tensor(
            log_mel_examples, requires_grad=True)[:, None, :, :].float()

    return log_mel_examples


def wavfile_to_examples(wav_file, return_tensor=True):
    """Convenience wrapper around waveform_to_examples() for a common WAV format.

  Args:
    wav_file: String path to a file, or a file-like object. The file
    is assumed to contain WAV audio data with signed 16-bit PCM samples.
    torch: Return data as a Pytorch tensor ready for VGGish

  Returns:
    See waveform_to_examples.
  """
    wav_data, sr = sf.read(wav_file, dtype='int16')
    assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype
    samples = wav_data / 32768.0  # Convert to [-1.0, +1.0]
    return waveform_to_examples(samples, sr, return_tensor)

import numpy as np
import torch
import torch.nn as nn
class VGG(nn.Module):
    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self.embeddings = nn.Sequential(
            nn.Linear(512 * 4 * 6, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 128),
            nn.ReLU(True))

    def forward(self, x):
        x = self.features(x)

        # Transpose the output from features to
        # remain compatible with vggish embeddings
        x = torch.transpose(x, 1, 3)
        x = torch.transpose(x, 1, 2)
        x = x.contiguous()
        x = x.view(x.size(0), -1)

        return self.embeddings(x)


class Postprocessor(nn.Module):
    """Post-processes VGGish embeddings. Returns a torch.Tensor instead of a
    numpy array in order to preserve the gradient.

    "The initial release of AudioSet included 128-D VGGish embeddings for each
    segment of AudioSet. These released embeddings were produced by applying
    a PCA transformation (technically, a whitening transform is included as well)
    and 8-bit quantization to the raw embedding output from VGGish, in order to
    stay compatible with the YouTube-8M project which provides visual embeddings
    in the same format for a large set of YouTube videos. This class implements
    the same PCA (with whitening) and quantization transformations."
    """

    def __init__(self):
        """Constructs a postprocessor."""
        super(Postprocessor, self).__init__()
        # Create empty matrix, for user's state_dict to load
        self.pca_eigen_vectors = torch.empty(
            (vggish_params.EMBEDDING_SIZE, vggish_params.EMBEDDING_SIZE,),
            dtype=torch.float,
        )
        self.pca_means = torch.empty(
            (vggish_params.EMBEDDING_SIZE, 1), dtype=torch.float
        )

        self.pca_eigen_vectors = nn.Parameter(self.pca_eigen_vectors, requires_grad=False)
        self.pca_means = nn.Parameter(self.pca_means, requires_grad=False)

    def postprocess(self, embeddings_batch):
        """Applies tensor postprocessing to a batch of embeddings.

        Args:
          embeddings_batch: An tensor of shape [batch_size, embedding_size]
            containing output from the embedding layer of VGGish.

        Returns:
          A tensor of the same shape as the input, containing the PCA-transformed,
          quantized, and clipped version of the input.
        """
        assert len(embeddings_batch.shape) == 2, "Expected 2-d batch, got %r" % (
            embeddings_batch.shape,
        )
        assert (
            embeddings_batch.shape[1] == vggish_params.EMBEDDING_SIZE
        ), "Bad batch shape: %r" % (embeddings_batch.shape,)

        # Apply PCA.
        # - Embeddings come in as [batch_size, embedding_size].
        # - Transpose to [embedding_size, batch_size].
        # - Subtract pca_means column vector from each column.
        # - Premultiply by PCA matrix of shape [output_dims, input_dims]
        #   where both are are equal to embedding_size in our case.
        # - Transpose result back to [batch_size, embedding_size].
        pca_applied = torch.mm(self.pca_eigen_vectors, (embeddings_batch.t() - self.pca_means)).t()

        # Quantize by:
        # - clipping to [min, max] range
        clipped_embeddings = torch.clamp(
            pca_applied, vggish_params.QUANTIZE_MIN_VAL, vggish_params.QUANTIZE_MAX_VAL
        )
        # - convert to 8-bit in range [0.0, 255.0]
        quantized_embeddings = torch.round(
            (clipped_embeddings - vggish_params.QUANTIZE_MIN_VAL)
            * (
                255.0
                / (vggish_params.QUANTIZE_MAX_VAL - vggish_params.QUANTIZE_MIN_VAL)
            )
        )
        return torch.squeeze(quantized_embeddings)

    def forward(self, x):
        return self.postprocess(x)


def make_layers():
    layers = []
    in_channels = 1
    for v in [64, "M", 128, "M", 256, 256, "M", 512, 512, "M"]:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def _vgg():
    return VGG(make_layers())


urls= {'vggish': 'https://github.com/harritaylor/torchvggish/'
              'releases/download/v0.1/vggish-10086976.pth',}

class VGGish(VGG):
    def __init__(self, urls, pretrained=True, preprocess=True, postprocess=False, progress=True):
        super().__init__(make_layers())
        
        if pretrained:
            print("yes")
            state_dict = torch.hub.load_state_dict_from_url(urls['vggish'], progress=progress)
            super().load_state_dict(state_dict)

        self.preprocess = preprocess

    def forward(self, x, fs=16000):
        if self.preprocess:
            x = self._preprocess(x, fs)
        
        x = VGG.forward(self, x)

        return x

    def _preprocess(self, x, fs):
        if isinstance(x, np.ndarray):
            x = waveform_to_examples(x, fs)
        elif isinstance(x, str):
            x = wavfile_to_examples(x)
        else:
            raise AttributeError
        return x
    def _postprocess(self, x):
        return self.pproc(x)



model=VGGish(urls,)
u=model('/DATA/lost+found/lost/lost/dataset/ydata-tvsum50-v1_1/audio/Hl-__g2gn_A.wav')
u.shape


