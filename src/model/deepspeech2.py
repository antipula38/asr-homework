from torch import nn
from torch.nn import Sequential
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class DeepSpeech2(nn.Module):

    def __init__(self, n_feats, n_tokens, conv_channels=(32, 32), conv_kernel_sizes=((41, 11), (21, 11)), conv_strides=((2, 2), (2, 1)), hidden_size=512, num_layers=3, bidirectional=True, dropout=0.0):
        super().__init__()
        padding_list = [((kernel[0] - 1) // 2, (kernel[1] - 1) // 2) for kernel in conv_kernel_sizes]

        self.conv_kernel_sizes = conv_kernel_sizes
        self.conv_strides = conv_strides
        self.conv_paddings = padding_list

        in_channels = 1
        layers = []
        freq_in = n_feats

        assert len(conv_channels) == len(conv_kernel_sizes) == len(conv_strides)
        
        for (out_channels, kernel_size, stride, padding) in zip(conv_channels, conv_kernel_sizes, conv_strides, padding_list):
            layers.extend([
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.BatchNorm2d(num_features=out_channels),
                nn.ReLU()
            ])
            in_channels = out_channels
            freq_in = self.freq_size(freq_in, kernel_size[0], padding[0], stride[0])
        self.conv = nn.Sequential(*layers)

        rnn_input_size = in_channels * freq_in

        if not 0.0 <= dropout <= 1.0:
            raise ValueError("dropout must be between 0 and 1")

        self.gru_bi = nn.GRU(
            input_size = rnn_input_size,
            hidden_size = hidden_size,
            num_layers = num_layers,
            bidirectional = bidirectional,
            batch_first = True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        lin_in_features = hidden_size * (int(bidirectional) + 1)
        self.linear = nn.Linear(in_features=lin_in_features, out_features=n_tokens)


    def forward(self, spectrogram, spectrogram_length, **batch):

        spectrogram = spectrogram.unsqueeze(1)

        output = self.conv(spectrogram).permute(0, 3, 1, 2).flatten(2)

        lengths = self.transform_input_lengths(spectrogram_length)
        packed = pack_padded_sequence(
            output,
            lengths,
            batch_first=True,
            enforce_sorted=False
        )

        output, _ = self.gru_bi(packed)
        padded, _ = pad_packed_sequence(
                    output,
                    batch_first=True
                )
        output = self.linear(padded)
        log_probs = nn.functional.log_softmax(output, dim=-1)

        return {"log_probs": log_probs, "log_probs_length": lengths}

    def transform_input_lengths(self, input_lengths):
            """
            As the network may compress the Time dimension, we need to know
            what are the new temporal lengths after compression.
    
            Args:
                input_lengths (Tensor): old input lengths
            Returns:
                output_lengths (Tensor): new temporal lengths
            """
            length = input_lengths 
            for (kernel_size, stride, padding) in zip(self.conv_kernel_sizes, self.conv_strides, self.conv_paddings):
                length = (length + 2 * padding[1] - kernel_size[1]) // stride[1] + 1
            return length

    def freq_size(self, f_in, k, p, s):
         return (f_in + 2 * p - k) // s + 1
