import torch.nn as nn
from torchvision import models

class Encoder(nn.Module):
    channels_in = 3
    channels_code = 512

    def __init__(
        self,
        pretrained: bool = True
    ):
        super(Encoder, self).__init__()

        vgg = models.vgg16_bn(pretrained=pretrained)
        del vgg.classifier
        del vgg.avgpool

        self.encoder = self._encodify_(vgg)

    def forward(self, x):
        pool_indices = []
        x_current = x
        for module_encode in self.encoder:
            output = module_encode(x_current)
            
            # If the module is pooling, there are two outputs, the second the pool indices
            if isinstance(output, tuple) and len(output) == 2:
                x_current = output[0]
                pool_indices.append(output[1])
            else:
                x_current = output

        return x_current, pool_indices

    def _encodify_(self, encoder):
        """
        Args:
            encoder: the template VGG model
        Returns:
            modules: the list of modules that define the encoder corresponding to the VGG model 
        """
        modules = nn.ModuleList()
        for module in encoder.features:
            if isinstance(module, nn.MaxPool2d):
                module_modify = nn.MaxPool2d(kernel_size=module.kernel_size,
                                        stride=module.stride,
                                        padding=module.padding,
                                        return_indices=True)
                modules.append(module_modify)
            else:
                modules.append(module)

        return modules

class Decoder(nn.Module):
    channels_in = Encoder.channels_code
    channels_out = 3

    def __init__(self, encoder):
        super(Decoder, self).__init__()
        self.decoder = self._invert_(encoder)

    def forward(self, x, pool_indices):
        x_current = x

        k_pool = 0
        reversed_pool_indices = list(reversed(pool_indices))
        for module_decode in self.decoder:

            # If the module is unpooling, collect the appropriate pooling indices
            if isinstance(module_decode, nn.MaxUnpool2d):
                x_current = module_decode(x_current, indices=reversed_pool_indices[k_pool])
                k_pool += 1
            else:
                x_current = module_decode(x_current)

        return x_current

    def _invert_(self, encoder):
        """
        Args:
            encoder (ModuleList): the encoder
        Returns:
            decoder (ModuleList): the decoder obtained by "inversion" of encoder
        """
        modules_transpose = []
        for module in reversed(encoder):

            if isinstance(module, nn.Conv2d):
                kwargs = {'in_channels' : module.out_channels, 'out_channels' : module.in_channels,
                          'kernel_size' : module.kernel_size, 'stride' : module.stride,
                          'padding' : module.padding}
                module_transpose = nn.ConvTranspose2d(**kwargs)
                module_norm = nn.BatchNorm2d(module.in_channels)
                module_act = nn.ReLU(inplace=True)
                modules_transpose += [module_transpose, module_norm, module_act]

            elif isinstance(module, nn.MaxPool2d):
                kwargs = {'kernel_size' : module.kernel_size, 'stride' : module.stride,
                          'padding' : module.padding}
                module_transpose = nn.MaxUnpool2d(**kwargs)
                modules_transpose += [module_transpose]

        # Discard the final normalization and activation, so final module is convolution with bias
        modules_transpose = modules_transpose[:-2]

        return nn.ModuleList(modules_transpose)

class AutoEncoder(nn.Module):
    channels_in = Encoder.channels_in
    channels_code = Encoder.channels_code
    channels_out = Decoder.channels_out

    def __init__(self, pretrained=True):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(pretrained=pretrained)
        self.decoder = Decoder(self.encoder.encoder)

    def forward(self, x):
        code, pool_indices = self.encoder(x)
        x_prime = self.decoder(code, pool_indices)

        return x_prime

class EncoderMerged(Encoder):
    def __init__(
        self,
        merger_type=None,
        pretrained=True
    ):
        super(EncoderMerged, self).__init__(pretrained=pretrained)
        if merger_type is None:
            self.code_post_process = lambda x: x
            self.code_post_process_kwargs = {}
        elif merger_type == 'mean':
            self.code_post_process = torch.mean
            self.code_post_process_kwargs = {'dim': (-2, -1)}
        elif merger_type == 'flatten':
            self.code_post_process = torch.flatten
            self.code_post_process_kwargs = {'start_dim': 1, 'end_dim': -1}
        else:
            raise ValueError('Unknown merger type for the encoder code: {}'. format(merger_type))

    def forward(self, x):
        x_current = super().forward(x)
        x_code = self.code_post_process(x_current, **self.code_post_process_kwargs)

        return x_code        

if __name__ == '__main__':
    import torch
    inp = torch.rand(1,3,224,224)
    model = AutoEncoder(pretrained=True)
    res = model(inp)
    print(model)
