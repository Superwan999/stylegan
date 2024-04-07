import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Blur2d(nn.Module):
    def __init__(self, f=[1, 2, 1], normalize=True, flip=False, stride=1):
        super(Blur2d, self).__init__()
        assert isinstance(stride, int) and stride >= 1

        if f is not None:

            f = torch.FloatTensor(f)
            if f.ndim == 1:
                f = f[: None] * f[None, :]

            assert f.ndim == 2
            if normalize:
                f = f / torch.sum(f)

            if flip:
                f = torch.flip(f, dims=[0, 1])

            f = f[:, :, None, None]

        self.f = f
        self.stride = stride

    def forward(self, x):
        if self.f is not None:
            filters = self.f.expand(x.size(1), -1, -1, -1).to(x.device)
            x = F.conv2d(
                x,
                filters,
                stride=self.stride,
                padding=int((self.f.size(2) - 1) / 2),
                groups=x.size(1)
            )

        return x


class EqualizedConv2d(nn.Module):
    def __init__(self,
                 in_c,
                 out_c,
                 kernel_size,
                 gain=2 ** (0.5),
                 use_wscale=False,
                 lrmul=1,
                 bias=True):
        super(EqualizedConv2d, self).__init__()

        self.kernel_size = kernel_size

        fan_in = in_c * out_c * (kernel_size ** 2)
        he_std = gain * fan_in ** (-0.5)
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_coef = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_coef = lrmul

        self.weight = nn.Parameter(torch.randn(out_c, in_c, kernel_size, kernel_size) * init_std)

        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_c))
            self.b_coef = lrmul
        else:
            self.bias = None

    def forward(self, x):
        if self.bias is not None:
            y = F.conv2d(x, weight=self.weight * self.w_coef,
                         bias=self.bias * self.b_coef,
                         padding=self.kernel_size // 2)
        else:
            y = F.conv2d(x, weight=self.weight * self.w_coef,
                         padding=self.kernel_size // 2)

        return y


class EqualizedFC(nn.Module):
    def __init__(self,
                 in_c,
                 out_c,
                 gain=2 ** (0.5),
                 use_wscale=False,
                 lrmul=1.0,
                 bias=True):
        super(EqualizedFC, self).__init__()

        fan_in = in_c * out_c
        he_std = gain / fan_in ** (0.5)
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_coef = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_coef = lrmul

        self.weight = nn.Parameter(torch.randn(out_c, in_c) * init_std)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_c))
            self.b_coef = lrmul
        else:
            self.bias = None

    def forward(self, x):
        if self.bias is not None:
            y = F.linear(x, weight=self.weight * self.w_coef, bias=self.bias * self.b_coef)
        else:
            y = F.linear(x, weight=self.weight * self.w_coef)

        y = F.leaky_relu(y, negative_slope=0.2, inplace=True)

        return y


class Upscale2d(nn.Module):
    def __init__(self, factor=2, gain=1):
        super(Upscale2d, self).__init__()

        self.gain = gain
        self.factor = factor

    def forward(self, x):
        if self.gain != 1:
            x = x * self.gain

        if self.factor > 1:
            shape = x.shape
            x = x.view(shape[0], shape[1], shape[2], 1, shape[3], 1).expand(-1, -1, -1, self.factor, -1, self.factor)
            x = x.contiguous().view(shape[0], shape[1], self.factor * shape[2], self.factor * shape[3])
        return x


class PixelNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(PixelNorm, self).__init__()

        self.epsilon = epsilon

    def forward(self, x):
        tmp = torch.mul(x, x)
        tmp1 = torch.rsqrt(torch.mean(tmp, dim=1, keepdim=True) + self.epsilon)
        return x * tmp1


class InstanceNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(InstanceNorm, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        x = x - torch.mean(x, dim=(2, 3), keepdim=True)
        tmp = torch.mul(x, x)
        tmp = torch.rsqrt(torch.mean(tmp, dim=(2, 3), keepdim=True) + self.epsilon)
        return x * tmp


class G_mapping(nn.Module):
    def __init__(self,
                 mapping_fmaps=512,
                 dlatent_size=512,
                 resolution=1024,
                 normalize_latents=True,
                 use_wscale=True,
                 lrmul=0.01,
                 gain=2 ** (0.5)):
        super(G_mapping, self).__init__()

        self.mapping_fmaps = mapping_fmaps
        self.layers = nn.Sequential(
            EqualizedFC(in_c=mapping_fmaps, out_c=dlatent_size, gain=gain, lrmul=lrmul, use_wscale=use_wscale),
            *[EqualizedFC(in_c=dlatent_size, out_c=dlatent_size, gain=gain, lrmul=lrmul, use_wscale=use_wscale) for _ in range(7)]
        )

        self.normalize_latents = normalize_latents
        self.res_log2 = int(np.log2(resolution))
        self.num_layers = self.res_log2 * 2 - 2
        self.pixel_norm = PixelNorm()

    def forward(self, x):
        if self.normalize_latents:
            x = self.pixel_norm(x)
        out = self.layers(x)
        return out, self.num_layers


class ApplyNoise(nn.Module):
    def __init__(self, channels):
        super(ApplyNoise, self).__init__()
        self.weight = nn.Parameter(torch.zeros(channels))

    def forward(self, x, noise):
        if noise is not None:
            noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
        return x + self.weight.view(1, -1, 1, 1) * noise.to(x.device)


class ApplyStyle(nn.Module):
    def __init__(self, channels, dlatent_size, use_wscale):
        super(ApplyStyle, self).__init__()

        self.fc = EqualizedFC(dlatent_size, channels * 2, gain=1.0, use_wscale=use_wscale)

    def forward(self,x, latent):
        style = self.fc(latent)
        style = style.view(-1, 2, x.size(1), 1, 1)
        out = x * (style[:, 0] + 1.0) + style[:, 1]

        return out


class LayerEpilogue(nn.Module):
    def __init__(self,
                 channels,
                 dlatent_size,
                 use_wscale,
                 use_noise,
                 use_pixel_norm,
                 use_instance_norm,
                 use_style):
        super(LayerEpilogue, self).__init__()

        self.apply_noise = ApplyNoise(channels)
        self.pixel_norm = PixelNorm()
        self.instance_norm = InstanceNorm()
        self.apply_style = ApplyStyle(channels, dlatent_size, use_wscale)
        self.activation = nn.LeakyReLU(negative_slope=0.2)

        self.use_noise = use_noise
        self.use_pixel_norm = use_pixel_norm
        self.use_instance_norm = use_instance_norm
        self.use_style = use_style

    def forward(self, x, noise, dlatent=None):
        if self.use_noise:
            x = self.apply_noise(x, noise)

        x = self.activation(x)

        if self.use_pixel_norm:
            x = self.instance_norm(x)

        if self.use_style:
            x = self.apply_style(x, dlatent)

        return x


class GBlock(nn.Module):
    def __init__(self,
                 res,
                 use_wscale,
                 use_noise,
                 use_pixel_norm,
                 use_instance_norm,
                 noise_input,
                 dlatent_size=512,
                 use_style=True,
                 f=None,
                 factor=2,
                 fmap_base=8192,
                 fmap_decay=1.0,
                 fmap_max=512):
        super(GBlock, self).__init__()

        self.nf = lambda stage: min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)

        self.res = res

        self.blur = Blur2d(f)
        self.noise_input = noise_input

        if res < 6:
            self.up_sample = Upscale2d(factor)
        else:
            self.up_sample = nn.ConvTranspose2d(self.nf(res - 3), self.nf(res - 2), 4, stride=2, padding=1)

        self.adaIn1 = LayerEpilogue(self.nf(res - 2),
                                    dlatent_size=dlatent_size,
                                    use_wscale=use_wscale,
                                    use_noise=use_noise,
                                    use_pixel_norm=use_pixel_norm,
                                    use_instance_norm=use_instance_norm,
                                    use_style=use_style)

        self.conv1 = EqualizedConv2d(in_c=self.nf(res - 2), out_c=self.nf(res - 2), kernel_size=3, use_wscale=use_wscale)

        self.adaIn2 = LayerEpilogue(self.nf(res - 2),
                                    dlatent_size=dlatent_size,
                                    use_wscale=use_wscale,
                                    use_noise=use_noise,
                                    use_pixel_norm=use_pixel_norm,
                                    use_instance_norm=use_instance_norm,
                                    use_style=use_style)

    def forward(self, x, dlatent):
        x = self.up_sample(x)
        x = self.adaIn1(x, self.noise_input[self.res * 2 - 4], dlatent[:, self.res * 2 - 4])
        x = self.conv1(x)
        x = self.adaIn2(x, self.noise_input[self.res * 2 - 3], dlatent[:, self.res * 2 - 3])
        return x


class G_synthesis(nn.Module):
    def __init__(self,
                 dlatent_size=512,
                 num_channels=3,
                 resolution=1024,
                 fmap_base=8192,
                 fmap_decay=1.0,
                 fmap_max=512,
                 f=[1, 2, 1],
                 use_pixel_norm=False,
                 use_instance_norm=True,
                 use_wscale=True,
                 use_noise=True,
                 use_style=True
                 ):
        super(G_synthesis, self).__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.nf = lambda stage: min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)

        self.res_log2 = int(np.log2(resolution))
        num_layers = 2 * (self.res_log2 - 1)
        self.num_layers = num_layers

        self.noise_inputs = []
        for layer_idx in range(num_layers):
            res = layer_idx // 2 + 2
            shape = [1, 1, 2 ** res, 2 ** res]
            self.noise_inputs.append(torch.randn(*shape).to(self.device))

        self.blur = Blur2d(f)

        # torgb: fixed mode
        self.channel_shrinkage = EqualizedConv2d(in_c=self.nf(self.res_log2 - 2),
                                                 out_c=self.nf(self.res_log2),
                                                 kernel_size=3,
                                                 use_wscale=use_wscale)
        self.torgb = EqualizedConv2d(self.nf(self.res_log2), num_channels, kernel_size=1, gain=1, use_wscale=use_wscale)

        # initial Input Block
        self.const_input = nn.Parameter(torch.ones(1, self.nf(1), 4, 4))
        self.bias = nn.Parameter(torch.ones(self.nf(1)))

        self.adaIn1 = LayerEpilogue(channels=self.nf(1),
                                    dlatent_size=dlatent_size,
                                    use_wscale=use_wscale,
                                    use_noise=use_noise,
                                    use_pixel_norm=use_pixel_norm,
                                    use_instance_norm=use_instance_norm,
                                    use_style=use_style)

        self.conv1 = EqualizedConv2d(in_c=self.nf(1), out_c=self.nf(1), kernel_size=3, use_wscale=use_wscale)
        self.adaIn2 = LayerEpilogue(channels=self.nf(1),
                                    dlatent_size=dlatent_size,
                                    use_wscale=use_wscale,
                                    use_noise=use_noise,
                                    use_pixel_norm=use_pixel_norm,
                                    use_instance_norm=use_instance_norm,
                                    use_style=use_style)


        self.blocks = nn.ModuleList([GBlock(res, use_wscale=use_wscale,
                                            use_noise=use_noise,
                                            use_pixel_norm=use_pixel_norm,
                                            use_instance_norm=use_instance_norm,
                                            noise_input=self.noise_inputs,
                                            dlatent_size=dlatent_size,
                                            use_style=True,
                                            f=None,
                                            factor=2,
                                            fmap_base=fmap_base,
                                            fmap_decay=fmap_decay,
                                            fmap_max=fmap_max
                                            ) for res in range(3, self.res_log2 + 1)])




    def forward(self, dlatent):
        images_out = None

        x = self.const_input.expand(dlatent.size(0), -1, -1, -1)

        x = x + self.bias.view(1, -1, 1, 1)

        x = self.adaIn1(x, self.noise_inputs[0], dlatent[:, 0])

        x = self.conv1(x)
        x = self.adaIn2(x, self.noise_inputs[1], dlatent[:, 1])

        for i, block in enumerate(self.blocks):
            # print(f"{i}: x shape: {x.shape}")
            x = block(x, dlatent)


        x = self.channel_shrinkage(x)
        images_out = self.torgb(x)

        return images_out


class StyleGenerator(nn.Module):
    def __init__(self,
                 resolution=1024,
                 mapping_fmaps=512,
                 fmap_base=8192,
                 fmap_max=512,
                 style_mixing_prob=0.9,
                 truncation_psi=0.7,
                 truncation_cutoff=8):
        super(StyleGenerator, self).__init__()

        self.mapping_fmaps = mapping_fmaps
        self.style_mixing_prob = style_mixing_prob
        self.truncation_psi = truncation_psi
        self.truncation_cutoff = truncation_cutoff

        self.mapping = G_mapping(self.mapping_fmaps, dlatent_size=mapping_fmaps, resolution=resolution)
        self.synthesis = G_synthesis(dlatent_size=self.mapping_fmaps, resolution=resolution, fmap_base=fmap_base, fmap_max=fmap_max)

    def forward(self, latents1):
        dlatents1, num_layers = self.mapping(latents1)

        dlatents1 = dlatents1.unsqueeze(1)
        dlatents1 = dlatents1.expand(-1, int(num_layers), -1)

        # Apply truncation trick
        if self.truncation_psi and self.truncation_cutoff:
            coefs = np.ones([1, num_layers, 1], dtype=np.float32)

            for i in range(num_layers):
                if i < self.truncation_cutoff:
                    coefs[:, i, :] *= self.truncation_psi

            dlatents1 = dlatents1 * torch.Tensor(coefs).to(dlatents1.device)

        img = self.synthesis(dlatents1)

        return img


class DownBlock(nn.Module):
    def __init__(self, f, in_c, out_c, type='avg_pool'):
        super(DownBlock, self).__init__()
        if type == 'avg_pool':
            self.down = nn.AvgPool2d(2)
        else:
            self.down = nn.Conv2d(out_c, out_c, kernel_size=2, stride=2)
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, stride=1)
        self.blur2d = Blur2d(f)
        self.activation1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.activation2 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation1(x)
        x = self.blur2d(x)
        x = self.down(x)
        x = self.activation2(x)
        return x


class StyleDiscriminator(nn.Module):
    def __init__(self,
                 resolution=1024,
                 fmap_base=8192,
                 num_channels=3,
                 fmap_max=512,
                 fmap_decay=1.0,
                 f=None,
                 ):
        super(StyleDiscriminator, self).__init__()
        self.res_log2 = int(np.log2(resolution))
        assert resolution == 2 ** self.res_log2 and resolution >= 4

        self.nf = lambda stage: min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)

        self.fromrgb = nn.Conv2d(num_channels, self.nf(self.res_log2 - 1), kernel_size=1)

        # down_sample
        self.block0 = DownBlock(f, in_c=self.nf(self.res_log2 - 1), out_c=self.nf(self.res_log2 - 1), type='avg_pool')
        #
        self.blocks = nn.Sequential(*[DownBlock(f, in_c=self.nf(self.res_log2 - i),
                                                out_c=self.nf(self.res_log2 - i - 1),
                                                type='avg_pool' if i < 4 else 'conv')
                                      for i in range(1, self.res_log2 - 2)])


        # fc head
        self.final_conv = nn.Conv2d(self.nf(2), self.nf(1), kernel_size=3, padding=1)
        self.dense0 = nn.Linear(fmap_base, self.nf(0))
        self.dense1 = nn.Linear(self.nf(0), 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fromrgb(x)
        x = F.leaky_relu(x, negative_slope=0.2, inplace=True)

        x = self.block0(x)

        x = self.blocks(x)

        x = self.final_conv(x)

        x = F.leaky_relu(x, negative_slope=0.2, inplace=True)

        x = x.view(x.size(0), -1)


        x = F.leaky_relu(self.dense0(x), 0.2, inplace=True)

        x = F.leaky_relu(self.dense1(x), 0.2, inplace=True)

        return x


if __name__ == "__main__":
    G = StyleGenerator(resolution=256, fmap_base=2048, mapping_fmaps=128, fmap_max=128).cuda()
    D = StyleDiscriminator(resolution=256, fmap_base=2048, fmap_max=128).cuda()

    z = torch.randn(8, 128).cuda()

    y = G(z)
    print(f"y.shape: {y.shape}")
    w = D(y)
    print(f"w shape: {w.shape}")
