import h5py
import torch
from torch import nn, optim
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt

from functools import partial
from einops import rearrange, repeat, reduce
from src.models.sequence import SequenceModule
from src.models.sequence.kernels import registry as kernel_registry
from src.models.nn import LinearActivation, Activation, DropoutNd
import src.utils as utils
contract = torch.einsum

device = "cuda" if torch.cuda.is_available() else "cpu"


###################################
# Neural Network Architecture Here
###################################
def multiple_axis_slice(x, L):
    """
    x: (..., L1, L2, .., Lk)
    L: list of length k [l1, l2, .., lk]
    returns: x[..., :l1, :l2, .., :lk]
    """
    # TODO I don't see a way to do this programmatically in Pytorch without sacrificing speed so...
    assert len(L) > 0
    if len(L) == 1:
        return x[..., :L[0]]
    elif len(L) == 2:
        return x[..., :L[0], :L[1]]
    elif len(L) == 3:
        return x[..., :L[0], :L[1], :L[2]]
    elif len(L) == 4:
        return x[..., :L[0], :L[1], :L[2], :L[3]]
    else: raise NotImplementedError("lol")


class S4ND(SequenceModule):
    requires_length = True

    def __init__(
        self,
        d_model,
        d_state=64,
        l_max=None, # Maximum length of sequence (list or tuple). None for unbounded
        dim=3, # Dimension of data, e.g. 2 for images and 3 for video
        out_channels=None, # Do depthwise-separable or not
        channels=1, # maps 1-dim to C-dim
        bidirectional=True,
        # Arguments for FF
        activation='gelu', # activation in between SS and FF
        ln=False, # Extra normalization
        final_act=None, # activation after FF
        initializer=None, # initializer on FF
        weight_norm=False, # weight normalization on FF
        hyper_act=None, # Use a "hypernetwork" multiplication
        dropout=0.0, tie_dropout=False,
        transposed=True, # axis ordering (B, L, D) or (B, D, L)
        verbose=False,
        trank=1, # tensor rank of C projection tensor
        linear=True,
        return_state=True,
        contract_version=0,
        # SSM Kernel arguments
        kernel=None,  # New option
        mode='dplr',  # Old option
        k_arg=[],
        #**kernel_args,
    ):
        """
        d_state: the dimension of the state, also denoted by N
        l_max: the maximum sequence length, also denoted by L
          if this is not known at model creation, or inconvenient to pass in,
          set l_max=None and length_correction=True
        dropout: standard dropout argument
        transposed: choose backbone axis ordering of (B, L, D) or (B, D, L) [B=batch size, L=sequence length, D=feature dimension]

        Other options are all experimental and should not need to be configured
        """

        super().__init__()
        print(f"Constructing S4ND (H, N, L) = ({d_model}, {d_state}, {l_max})")

        self.h = d_model
        self.n = d_state
        self.bidirectional = bidirectional
        self.ln = ln
        self.channels = channels
        self.transposed = transposed
        self.linear = linear
        self.return_state = return_state
        self.contract_version = contract_version
        self.out_channels = out_channels
        self.verbose = verbose
        #self.kernel_args = kernel_args
        self.kernel_args = k_arg
        self.kernel_pass = kernel
        self.mode_pass = mode

        self.D = nn.Parameter(torch.randn(self.channels, self.h)) # TODO if self.out_channels

        self.trank = trank

        if self.out_channels is not None:
            channels *= self.out_channels

            assert self.linear # TODO change name of linear_output

        channels *= self.trank

        if self.bidirectional:
            channels *= 2

        # Check dimensions and kernel sizes
        if dim is None:
            assert utils.is_list(l_max)

        # assert l_max is not None # TODO implement auto-sizing functionality for the kernel
        if l_max is None:
            self.l_max = [None] * dim
        elif isinstance(l_max, int):
            self.l_max = [l_max] * dim
        else:
            assert l_max is None or utils.is_list(l_max)
            self.l_max = l_max

        # SSM Kernel
        if kernel is None and mode is not None: kernel = mode
        self._kernel_channels = channels
        #self.kernel = nn.ModuleList([
        #    # SSKernel(self.h, N=self.n, L=L, channels=channels, verbose=verbose, **kernel_args)
        #    kernel_registry[kernel](d_model=self.h, d_state=self.n, l_max=L, channels=channels, verbose=verbose, **kernel_args)
        #    for L in self.l_max
        #])
        self.kernel = nn.ModuleList([
            # SSKernel(self.h, N=self.n, L=L, channels=channels, verbose=verbose, **kernel_args)
            #kernel_registry[kernel](init='fourier',d_model=self.h, d_state=self.n, l_max=L, channels=channels, verbose=verbose,
            #                        **k_arg[idx])
            kernel_registry[kernel](d_model=self.h, d_state=self.n, l_max=L, channels=channels,
                                    verbose=verbose, **k_arg[idx])
            for idx, L in enumerate(self.l_max)
        ])

        if not self.linear:

            self.activation = Activation(activation)
            dropout_fn = partial(DropoutNd, transposed=self.transposed) if tie_dropout else nn.Dropout
            self.dropout = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()


            # position-wise output transform to mix features
            self.output_linear = LinearActivation(
                self.h*self.channels,
                self.h,
                transposed=self.transposed,
                initializer=initializer,
                activation=final_act,
                activate=True,
                weight_norm=weight_norm,
            )

        ## To handle some operations with unspecified number of dims, we're going to define the einsum/einops contractions programmatically

        # Outer product function for the convolution kernel taking arbitary number of dims
        contract_str = ', '.join([f'... {chr(i+97)}' for i in range(len(self.l_max))]) \
            + ' -> ... ' \
            + ' '.join([f'{chr(i+97)}' for i in range(len(self.l_max))])


    def reinit(self, k_arg):
        if k_arg[0]['dt_min'].get_device() == 0:
            k_arg_rec = [{k: v.cpu() for k, v in _data.items()} for _data in k_arg]
            #self.kernel = self.kernel.cpu()
        else:
            k_arg_rec = k_arg
        if self.kernel_pass is None and self.mode_pass is not None:
            kernel = self.mode_pass
        else:
            kernel = self.kernel_pass
        self.kernel = nn.ModuleList([
            kernel_registry[kernel](d_model=self.h, d_state=self.n, l_max=L, channels=self._kernel_channels, verbose=self.verbose,
                                    **k_arg_rec[idx])
            for idx, L in enumerate(self.l_max)
        ])
        #if k_arg[0]['dt_min'].get_device() == 0:
         #   self.kernel = self.kernel.to(device)

    def forward(self, u, rate=1.0, state=None, **kwargs):  # absorbs return_output and transformer src mask
        """
        u: (B H L) if self.transposed else (B L H)
        state: (H N) never needed unless you know what you're doing

        Returns: same shape as u
        """

        half_precision = False

        # fft can only handle float32
        if u.dtype == torch.float16:
            half_precision = True
            u = u.to(torch.float32)

        assert state is None, f"state not currently supported in S4ND"

        # ensure shape is B, C, L (L can be multi-axis)
        if not self.transposed:
            u = rearrange(u, "b ... h -> b h ...")

        L_input = u.shape[2:]
        # print(L_input)

        L_kernel = [
            l_i if l_k is None else min(l_i, round(l_k / rate)) for l_i, l_k in zip(L_input, self.l_max)
        ]
        # print(L_kernel)
        # Compute SS Kernel
        # 1 kernel for each axis in L
        # for us length 3, each kernel 8x7x16 for example
        #print(self.kernel[0](L=L_kernel[0], rate=rate)[0])
        k = [kernel(L=l, rate=rate)[0] for kernel, l in zip(self.kernel, L_kernel)]
        if k[0].get_device() != 0:
            k = [entry.to(device) for entry in k]
        # print(k)
        # print(self.kernel)

        if self.bidirectional:  # halves channels
            k = [torch.chunk(_k, 2, dim=-3) for _k in k]  # (C H L)
            k = [
                F.pad(k0, (0, l)) + F.pad(k1.flip(-1), (l, 0))
                # for l, (k0, k1) in zip(L_kernel, k) # TODO bug??
                for l, (k0, k1) in zip(L_input, k)
            ]

        # fft can only handle float32
        if u.dtype == torch.float16:
            half_precision = True
            # cast to fp32
            k.dtype = torch.float32

        L_padded = [l_input + l_kernel for l_input, l_kernel in zip(L_input, L_kernel)]
        # print(L_padded)
        # print(u.shape)
        # since real input, can omit negative frequencies in last dimension that's why (rfftn is used)
        u_f = torch.fft.rfftn(u, s=tuple([l for l in L_padded]))  # (B H L)
        # print(u_f)
        # print(u_f.shape)
        # print(k[0].shape)
        # print(k[1].shape)
        # print(k[2].shape)
        # print(k)
        k_f = [torch.fft.fft(_k, n=l) for _k, l in zip(k[:-1], L_padded[:-1])] + [
            torch.fft.rfft(k[-1], n=L_padded[-1])]  # (C H L)

        # print(k_f[0].shape)
        # print(k_f[1].shape)
        # print(k_f[2].shape)
        # Take outer products

        if self.contract_version == 0:  # TODO set this automatically if l_max is provided
            k_f = contract('... c h m, ... c h n -> ... c h m n', k_f[0], k_f[1])  # (H L1 L2) # 2D case of next line
            # k_f = self.nd_outer(*k_f)
            # sum over tensor rank
            k_f = reduce(k_f, '(r c) h ... -> c h ...', 'sum',
                         r=self.trank) / self.trank  # reduce_mean not available for complex... # TODO does it matter if (r c) or (c r)?
            y_f = contract('bh...,ch...->bch...', u_f, k_f)  # k_f.unsqueeze(-4) * u_f.unsqueeze(-3) # (B C H L)

        else:
            contract_str_l = [f'{chr(i + 100)}' for i in range(len(L_input))]
            contract_str = 'b ... ' + ' '.join(contract_str_l) + ', ' \
                           + ', '.join(['... ' + l for l in contract_str_l]) \
                           + ' -> b ... ' \
                           + ' '.join(contract_str_l)

            # print(contract_str)
            # print(u_f.shape)
            y_f = contract(contract_str, u_f, *k_f)
            # print(y_f.shape)
            k_f = reduce(y_f, 'b (r c) h ... -> b c h ...', 'sum',
                         r=self.trank) / self.trank  # reduce_mean not available for complex... # TODO does it matter if (r c) or (c r)?

        # print(y_f)
        # Contract over channels if not depthwise separable
        if self.out_channels is not None:
            y_f = reduce(y_f, 'b (i c) h ... -> b c i ...', 'sum',
                         i=self.out_channels)  # TODO normalization might not be right

        # print(y_f)
        y = torch.fft.irfftn(y_f, s=tuple([l for l in L_padded]))

        # need to cast back to half if used
        if half_precision:
            y = y.to(torch.float16)

        y = multiple_axis_slice(y, L_input)

        # Compute D term in state space equation - essentially a skip connection
        # B, C, H, L (not flat)
        if not self.out_channels:
            y = y + contract('bh...,ch->bch...', u, self.D)  # u.unsqueeze(-3) * self.D.unsqueeze(-1)

        # Reshape to flatten channels
        # B, H, L (not flat)
        y = rearrange(y, 'b c h ... -> b (c h) ...')

        if not self.linear:
            y = self.dropout(self.activation(y))

        # ensure output and input shape are the same
        if not self.transposed:
            # B, H, L -> B, H, C
            y = rearrange(y, "b h ... -> b ... h")

        if not self.linear:
            y = self.output_linear(y)

        if self.return_state:
            return y, None
        else:
            return y

    def default_state(self, *batch_shape, device=None):
        return self._initial_state.repeat(*batch_shape, 1, 1)

    @property
    def d_output(self):
        return self.h
        # return self.h if self.out_channels is None else self.out_channels

    @property
    def d_state(self):
        raise NotImplementedError

    @property
    def state_to_tensor(self):
        raise NotImplementedError

# Now with the S4ND code complete (with some modifications made, we can declare our TBNN)

class UnetTBNN(nn.Module):
    def __init__(self, k_args, k_args2):
        super(UnetTBNN, self).__init__()
        self.S4ND1 = S4ND(d_model=11, d_state=8, dim=3, l_max=(None, None, None), contract_version=1,
                          linear=True, bidirectional=False, transposed=True,
                          return_state=False, mode='s4', k_arg=k_args)
        self.S4ND15 = S4ND(d_model=11, d_state=8, dim=3, l_max=(None, None, None), contract_version=1,
                           linear=True, bidirectional=False, transposed=True,
                           return_state=False, mode='s4', k_arg=k_args)
        self.S4ND2 = S4ND(d_model=11, d_state=8, dim=3, l_max=(None, None, None), contract_version=1,
                          linear=True, bidirectional=False, transposed=True,
                          return_state=False, mode='s4', k_arg=k_args2)
        self.S4ND25 = S4ND(d_model=11, d_state=8, dim=3, l_max=(None, None, None), contract_version=1,
                           linear=True, bidirectional=False, transposed=True,
                           return_state=False, mode='s4', k_arg=k_args2)
        self.S4ND3 = S4ND(d_model=22, d_state=8, dim=3, l_max=(None, None, None), contract_version=1,
                          linear=True, bidirectional=False, transposed=True,
                          return_state=False, mode='s4', k_arg=k_args)

        self.activation_leaky = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(p=0.1)
        self.avgpool = nn.AvgPool3d(kernel_size=2, stride=2)
        self.conv8 = nn.Conv3d(22, 16, kernel_size=1, stride=1, padding=0, padding_mode='zeros')
        self.conv9 = nn.Conv3d(16, 16, kernel_size=1, stride=1, padding=0, padding_mode='zeros')
        self.conv10 = nn.Conv3d(16, 8, kernel_size=1, stride=1, padding=0, padding_mode='zeros')
        self.conv11 = nn.Conv3d(8, 16, kernel_size=1, stride=1, padding=0, padding_mode='zeros')
        self.conv12 = nn.Conv3d(16, 16, kernel_size=1, stride=1, padding=0, padding_mode='zeros')
        self.conv13 = nn.Conv3d(16, 8, kernel_size=1, stride=1, padding=0, padding_mode='zeros')
        self.conv8m = nn.Conv3d(22, 16, kernel_size=1, stride=1, padding=0, padding_mode='zeros')
        self.conv9m = nn.Conv3d(16, 16, kernel_size=1, stride=1, padding=0, padding_mode='zeros')
        self.conv10m = nn.Conv3d(16, 8, kernel_size=1, stride=1, padding=0, padding_mode='zeros')
        self.conv11m = nn.Conv3d(8, 16, kernel_size=1, stride=1, padding=0, padding_mode='zeros')
        self.conv12m = nn.Conv3d(16, 16, kernel_size=1, stride=1, padding=0, padding_mode='zeros')
        self.conv13m = nn.Conv3d(16, 8, kernel_size=1, stride=1, padding=0, padding_mode='zeros')
        self.instancenorm3 = nn.InstanceNorm3d(16)

    def forward(self, invar_in, strain2_in, strain2_m_in, rot2_in, strainrot_m_in, rotstrainrot_m_in, strain2rot_m_in, rotstrain2_m_in, delta_in, rot_in, k_arg1, k_arg2):
        self.S4ND1.reinit(k_arg1)
        self.S4ND15.reinit(k_arg1)
        self.S4ND2.reinit(k_arg2)
        self.S4ND25.reinit(k_arg2)
        self.S4ND3.reinit(k_arg1)

        x = self.activation_leaky(self.S4ND1(invar_in))
        x = self.activation_leaky(self.S4ND15(x))
        x_2 = self.avgpool(x)
        x_2 = self.activation_leaky(self.S4ND2(x_2))
        x_2 = self.activation_leaky(self.S4ND25(x_2))
        x_1 = self.activation_leaky(F.interpolate(x_2, scale_factor=(2, 2, 2), mode="trilinear"))
        merge = torch.cat((x, x_1), 1)
        S4out = self.activation_leaky((self.S4ND3(merge)))
        conv_out = self.activation_leaky(self.instancenorm3(self.conv8(S4out)))
        conv_out = self.activation_leaky(self.conv9(conv_out))
        outlayer = self.activation_leaky(self.conv10(conv_out))
        outlayer = self.activation_leaky(self.conv11(outlayer))
        outlayer = self.activation_leaky(self.conv12(outlayer))
        outlayer = self.conv13(outlayer)

        conv_outm = self.activation_leaky(self.conv8m(S4out))
        conv_outm = self.activation_leaky(self.conv9m(conv_outm))
        outlayerm = self.activation_leaky(self.conv10m(conv_outm))
        outlayerm = self.activation_leaky(self.conv11m(outlayerm))
        outlayerm = self.activation_leaky(self.conv12m(outlayerm))
        outlayerm = self.conv13m(outlayerm)

        x1_1 = torch.mul(outlayer[::1, 1, ::1, ::1, ::1], strain2_in[::1, ::1, ::1, ::1, 0, 0])
        x1_2 = torch.mul(outlayer[::1, 2, ::1, ::1, ::1], strain2_m_in[::1, ::1, ::1, ::1, 0, 0])
        x1_3 = torch.mul(outlayer[::1, 3, ::1, ::1, ::1], rot2_in[::1, ::1, ::1, ::1, 0, 0])
        x1_4 = torch.mul(outlayer[::1, 4, ::1, ::1, ::1], strainrot_m_in[::1, ::1, ::1, ::1, 0, 0])
        x1_5 = torch.mul(outlayer[::1, 5, ::1, ::1, ::1], rotstrainrot_m_in[::1, ::1, ::1, ::1, 0, 0])
        x1_6 = torch.mul(outlayer[::1, 6, ::1, ::1, ::1], strain2rot_m_in[::1, ::1, ::1, ::1, 0, 0])
        x1_7 = torch.mul(outlayer[::1, 7, ::1, ::1, ::1], rotstrain2_m_in[::1, ::1, ::1, ::1, 0, 0])
        x_1 = torch.add(x1_1, x1_2)
        x_1 = torch.add(x_1, x1_3)
        x_1 = torch.add(x_1, x1_4)
        x_1 = torch.add(x_1, x1_5)
        x_1 = torch.add(x_1, x1_6)
        x_1 = torch.add(x_1, x1_7)
        x_1 = torch.add(x_1, outlayer[::1, 0, ::1, ::1, ::1])

        x2_1 = torch.mul(outlayer[::1, 1, ::1, ::1, ::1], strain2_in[::1, ::1, ::1, ::1, 0, 1])
        x2_2 = torch.mul(outlayer[::1, 2, ::1, ::1, ::1], strain2_m_in[::1, ::1, ::1, ::1, 0, 1])
        x2_3 = torch.mul(outlayer[::1, 3, ::1, ::1, ::1], rot2_in[::1, ::1, ::1, ::1, 0, 1])
        x2_4 = torch.mul(outlayer[::1, 4, ::1, ::1, ::1], strainrot_m_in[::1, ::1, ::1, ::1, 0, 1])
        x2_5 = torch.mul(outlayer[::1, 5, ::1, ::1, ::1], rotstrainrot_m_in[::1, ::1, ::1, ::1, 0, 1])
        x2_6 = torch.mul(outlayer[::1, 6, ::1, ::1, ::1], strain2rot_m_in[::1, ::1, ::1, ::1, 0, 1])
        x2_7 = torch.mul(outlayer[::1, 7, ::1, ::1, ::1], rotstrain2_m_in[::1, ::1, ::1, ::1, 0, 1])
        x_2 = torch.add(x2_1, x2_2)
        x_2 = torch.add(x_2, x2_3)
        x_2 = torch.add(x_2, x2_4)
        x_2 = torch.add(x_2, x2_5)
        x_2 = torch.add(x_2, x2_6)
        x_2 = torch.add(x_2, x2_7)

        x3_1 = torch.mul(outlayer[::1, 1, ::1, ::1, ::1], strain2_in[::1, ::1, ::1, ::1, 0, 2])
        x3_2 = torch.mul(outlayer[::1, 2, ::1, ::1, ::1], strain2_m_in[::1, ::1, ::1, ::1, 0, 2])
        x3_3 = torch.mul(outlayer[::1, 3, ::1, ::1, ::1], rot2_in[::1, ::1, ::1, ::1, 0, 2])
        x3_4 = torch.mul(outlayer[::1, 4, ::1, ::1, ::1], strainrot_m_in[::1, ::1, ::1, ::1, 0, 2])
        x3_5 = torch.mul(outlayer[::1, 5, ::1, ::1, ::1], rotstrainrot_m_in[::1, ::1, ::1, ::1, 0, 2])
        x3_6 = torch.mul(outlayer[::1, 6, ::1, ::1, ::1], strain2rot_m_in[::1, ::1, ::1, ::1, 0, 2])
        x3_7 = torch.mul(outlayer[::1, 7, ::1, ::1, ::1], rotstrain2_m_in[::1, ::1, ::1, ::1, 0, 2])
        x_3 = torch.add(x3_1, x3_2)
        x_3 = torch.add(x_3, x3_3)
        x_3 = torch.add(x_3, x3_4)
        x_3 = torch.add(x_3, x3_5)
        x_3 = torch.add(x_3, x3_6)
        x_3 = torch.add(x_3, x3_7)

        x4_1 = torch.mul(outlayer[::1, 1, ::1, ::1, ::1], strain2_in[::1, ::1, ::1, ::1, 1, 1])
        x4_2 = torch.mul(outlayer[::1, 2, ::1, ::1, ::1], strain2_m_in[::1, ::1, ::1, ::1, 1, 1])
        x4_3 = torch.mul(outlayer[::1, 3, ::1, ::1, ::1], rot2_in[::1, ::1, ::1, ::1, 1, 1])
        x4_4 = torch.mul(outlayer[::1, 4, ::1, ::1, ::1], strainrot_m_in[::1, ::1, ::1, ::1, 1, 1])
        x4_5 = torch.mul(outlayer[::1, 5, ::1, ::1, ::1], rotstrainrot_m_in[::1, ::1, ::1, ::1, 1, 1])
        x4_6 = torch.mul(outlayer[::1, 6, ::1, ::1, ::1], strain2rot_m_in[::1, ::1, ::1, ::1, 1, 1])
        x4_7 = torch.mul(outlayer[::1, 7, ::1, ::1, ::1], rotstrain2_m_in[::1, ::1, ::1, ::1, 1, 1])
        x_4 = torch.add(x4_1, x4_2)
        x_4 = torch.add(x_4, x4_3)
        x_4 = torch.add(x_4, x4_4)
        x_4 = torch.add(x_4, x4_5)
        x_4 = torch.add(x_4, x4_6)
        x_4 = torch.add(x_4, x4_7)
        x_4 = torch.add(x_4, outlayer[::1, 0, ::1, ::1, ::1])

        x5_1 = torch.mul(outlayer[::1, 1, ::1, ::1, ::1], strain2_in[::1, ::1, ::1, ::1, 1, 2])
        x5_2 = torch.mul(outlayer[::1, 2, ::1, ::1, ::1], strain2_m_in[::1, ::1, ::1, ::1, 1, 2])
        x5_3 = torch.mul(outlayer[::1, 3, ::1, ::1, ::1], rot2_in[::1, ::1, ::1, ::1, 1, 2])
        x5_4 = torch.mul(outlayer[::1, 4, ::1, ::1, ::1], strainrot_m_in[::1, ::1, ::1, ::1, 1, 2])
        x5_5 = torch.mul(outlayer[::1, 5, ::1, ::1, ::1], rotstrainrot_m_in[::1, ::1, ::1, ::1, 1, 2])
        x5_6 = torch.mul(outlayer[::1, 6, ::1, ::1, ::1], strain2rot_m_in[::1, ::1, ::1, ::1, 1, 2])
        x5_7 = torch.mul(outlayer[::1, 7, ::1, ::1, ::1], rotstrain2_m_in[::1, ::1, ::1, ::1, 1, 2])
        x_5 = torch.add(x5_1, x5_2)
        x_5 = torch.add(x_5, x5_3)
        x_5 = torch.add(x_5, x5_4)
        x_5 = torch.add(x_5, x5_5)
        x_5 = torch.add(x_5, x5_6)
        x_5 = torch.add(x_5, x5_7)

        x6_1 = torch.mul(outlayer[::1, 1, ::1, ::1, ::1], strain2_in[::1, ::1, ::1, ::1, 2, 2])
        x6_2 = torch.mul(outlayer[::1, 2, ::1, ::1, ::1], strain2_m_in[::1, ::1, ::1, ::1, 2, 2])
        x6_3 = torch.mul(outlayer[::1, 3, ::1, ::1, ::1], rot2_in[::1, ::1, ::1, ::1, 2, 2])
        x6_4 = torch.mul(outlayer[::1, 4, ::1, ::1, ::1], strainrot_m_in[::1, ::1, ::1, ::1, 2, 2])
        x6_5 = torch.mul(outlayer[::1, 5, ::1, ::1, ::1], rotstrainrot_m_in[::1, ::1, ::1, ::1, 2, 2])
        x6_6 = torch.mul(outlayer[::1, 6, ::1, ::1, ::1], strain2rot_m_in[::1, ::1, ::1, ::1, 2, 2])
        x6_7 = torch.mul(outlayer[::1, 7, ::1, ::1, ::1], rotstrain2_m_in[::1, ::1, ::1, ::1, 2, 2])
        x_6 = torch.add(x6_1, x6_2)
        x_6 = torch.add(x_6, x6_3)
        x_6 = torch.add(x_6, x6_4)
        x_6 = torch.add(x_6, x6_5)
        x_6 = torch.add(x_6, x6_6)
        x_6 = torch.add(x_6, x6_7)
        x_6 = torch.add(x_6, outlayer[::1, 0, ::1, ::1, ::1])
        y = torch.stack((x_1, x_2, x_3, x_4, x_5, x_6), dim=1)

        x1_1 = torch.mul(outlayerm[::1, 1, ::1, ::1, ::1], strain2_in[::1, ::1, ::1, ::1, 0, 0])
        x1_2 = torch.mul(outlayerm[::1, 2, ::1, ::1, ::1], strain2_m_in[::1, ::1, ::1, ::1, 0, 0])
        x1_3 = torch.mul(outlayerm[::1, 3, ::1, ::1, ::1], rot2_in[::1, ::1, ::1, ::1, 0, 0])
        x1_4 = torch.mul(outlayerm[::1, 4, ::1, ::1, ::1], strainrot_m_in[::1, ::1, ::1, ::1, 0, 0])
        x1_5 = torch.mul(outlayerm[::1, 5, ::1, ::1, ::1], rotstrainrot_m_in[::1, ::1, ::1, ::1, 0, 0])
        x1_6 = torch.mul(outlayerm[::1, 6, ::1, ::1, ::1], strain2rot_m_in[::1, ::1, ::1, ::1, 0, 0])
        x1_7 = torch.mul(outlayerm[::1, 7, ::1, ::1, ::1], rotstrain2_m_in[::1, ::1, ::1, ::1, 0, 0])
        x_1 = torch.add(x1_1, x1_2)
        x_1 = torch.add(x_1, x1_3)
        x_1 = torch.add(x_1, x1_4)
        x_1 = torch.add(x_1, x1_5)
        x_1 = torch.add(x_1, x1_6)
        x_1 = torch.add(x_1, x1_7)
        x_1 = torch.add(x_1, outlayerm[::1, 0, ::1, ::1, ::1])

        x2_1 = torch.mul(outlayerm[::1, 1, ::1, ::1, ::1], strain2_in[::1, ::1, ::1, ::1, 0, 1])
        x2_2 = torch.mul(outlayerm[::1, 2, ::1, ::1, ::1], strain2_m_in[::1, ::1, ::1, ::1, 0, 1])
        x2_3 = torch.mul(outlayerm[::1, 3, ::1, ::1, ::1], rot2_in[::1, ::1, ::1, ::1, 0, 1])
        x2_4 = torch.mul(outlayerm[::1, 4, ::1, ::1, ::1], strainrot_m_in[::1, ::1, ::1, ::1, 0, 1])
        x2_5 = torch.mul(outlayerm[::1, 5, ::1, ::1, ::1], rotstrainrot_m_in[::1, ::1, ::1, ::1, 0, 1])
        x2_6 = torch.mul(outlayerm[::1, 6, ::1, ::1, ::1], strain2rot_m_in[::1, ::1, ::1, ::1, 0, 1])
        x2_7 = torch.mul(outlayerm[::1, 7, ::1, ::1, ::1], rotstrain2_m_in[::1, ::1, ::1, ::1, 0, 1])
        x_2 = torch.add(x2_1, x2_2)
        x_2 = torch.add(x_2, x2_3)
        x_2 = torch.add(x_2, x2_4)
        x_2 = torch.add(x_2, x2_5)
        x_2 = torch.add(x_2, x2_6)
        x_2 = torch.add(x_2, x2_7)

        x3_1 = torch.mul(outlayerm[::1, 1, ::1, ::1, ::1], strain2_in[::1, ::1, ::1, ::1, 0, 2])
        x3_2 = torch.mul(outlayerm[::1, 2, ::1, ::1, ::1], strain2_m_in[::1, ::1, ::1, ::1, 0, 2])
        x3_3 = torch.mul(outlayerm[::1, 3, ::1, ::1, ::1], rot2_in[::1, ::1, ::1, ::1, 0, 2])
        x3_4 = torch.mul(outlayerm[::1, 4, ::1, ::1, ::1], strainrot_m_in[::1, ::1, ::1, ::1, 0, 2])
        x3_5 = torch.mul(outlayerm[::1, 5, ::1, ::1, ::1], rotstrainrot_m_in[::1, ::1, ::1, ::1, 0, 2])
        x3_6 = torch.mul(outlayerm[::1, 6, ::1, ::1, ::1], strain2rot_m_in[::1, ::1, ::1, ::1, 0, 2])
        x3_7 = torch.mul(outlayerm[::1, 7, ::1, ::1, ::1], rotstrain2_m_in[::1, ::1, ::1, ::1, 0, 2])
        x_3 = torch.add(x3_1, x3_2)
        x_3 = torch.add(x_3, x3_3)
        x_3 = torch.add(x_3, x3_4)
        x_3 = torch.add(x_3, x3_5)
        x_3 = torch.add(x_3, x3_6)
        x_3 = torch.add(x_3, x3_7)

        x4_1 = torch.mul(outlayerm[::1, 1, ::1, ::1, ::1], strain2_in[::1, ::1, ::1, ::1, 1, 1])
        x4_2 = torch.mul(outlayerm[::1, 2, ::1, ::1, ::1], strain2_m_in[::1, ::1, ::1, ::1, 1, 1])
        x4_3 = torch.mul(outlayerm[::1, 3, ::1, ::1, ::1], rot2_in[::1, ::1, ::1, ::1, 1, 1])
        x4_4 = torch.mul(outlayerm[::1, 4, ::1, ::1, ::1], strainrot_m_in[::1, ::1, ::1, ::1, 1, 1])
        x4_5 = torch.mul(outlayerm[::1, 5, ::1, ::1, ::1], rotstrainrot_m_in[::1, ::1, ::1, ::1, 1, 1])
        x4_6 = torch.mul(outlayerm[::1, 6, ::1, ::1, ::1], strain2rot_m_in[::1, ::1, ::1, ::1, 1, 1])
        x4_7 = torch.mul(outlayerm[::1, 7, ::1, ::1, ::1], rotstrain2_m_in[::1, ::1, ::1, ::1, 1, 1])
        x_4 = torch.add(x4_1, x4_2)
        x_4 = torch.add(x_4, x4_3)
        x_4 = torch.add(x_4, x4_4)
        x_4 = torch.add(x_4, x4_5)
        x_4 = torch.add(x_4, x4_6)
        x_4 = torch.add(x_4, x4_7)
        x_4 = torch.add(x_4, outlayerm[::1, 0, ::1, ::1, ::1])

        x5_1 = torch.mul(outlayerm[::1, 1, ::1, ::1, ::1], strain2_in[::1, ::1, ::1, ::1, 1, 2])
        x5_2 = torch.mul(outlayerm[::1, 2, ::1, ::1, ::1], strain2_m_in[::1, ::1, ::1, ::1, 1, 2])
        x5_3 = torch.mul(outlayerm[::1, 3, ::1, ::1, ::1], rot2_in[::1, ::1, ::1, ::1, 1, 2])
        x5_4 = torch.mul(outlayerm[::1, 4, ::1, ::1, ::1], strainrot_m_in[::1, ::1, ::1, ::1, 1, 2])
        x5_5 = torch.mul(outlayerm[::1, 5, ::1, ::1, ::1], rotstrainrot_m_in[::1, ::1, ::1, ::1, 1, 2])
        x5_6 = torch.mul(outlayerm[::1, 6, ::1, ::1, ::1], strain2rot_m_in[::1, ::1, ::1, ::1, 1, 2])
        x5_7 = torch.mul(outlayerm[::1, 7, ::1, ::1, ::1], rotstrain2_m_in[::1, ::1, ::1, ::1, 1, 2])
        x_5 = torch.add(x5_1, x5_2)
        x_5 = torch.add(x_5, x5_3)
        x_5 = torch.add(x_5, x5_4)
        x_5 = torch.add(x_5, x5_5)
        x_5 = torch.add(x_5, x5_6)
        x_5 = torch.add(x_5, x5_7)

        x6_1 = torch.mul(outlayerm[::1, 1, ::1, ::1, ::1], strain2_in[::1, ::1, ::1, ::1, 2, 2])
        x6_2 = torch.mul(outlayerm[::1, 2, ::1, ::1, ::1], strain2_m_in[::1, ::1, ::1, ::1, 2, 2])
        x6_3 = torch.mul(outlayerm[::1, 3, ::1, ::1, ::1], rot2_in[::1, ::1, ::1, ::1, 2, 2])
        x6_4 = torch.mul(outlayerm[::1, 4, ::1, ::1, ::1], strainrot_m_in[::1, ::1, ::1, ::1, 2, 2])
        x6_5 = torch.mul(outlayerm[::1, 5, ::1, ::1, ::1], rotstrainrot_m_in[::1, ::1, ::1, ::1, 2, 2])
        x6_6 = torch.mul(outlayerm[::1, 6, ::1, ::1, ::1], strain2rot_m_in[::1, ::1, ::1, ::1, 2, 2])
        x6_7 = torch.mul(outlayerm[::1, 7, ::1, ::1, ::1], rotstrain2_m_in[::1, ::1, ::1, ::1, 2, 2])
        x_6 = torch.add(x6_1, x6_2)
        x_6 = torch.add(x_6, x6_3)
        x_6 = torch.add(x_6, x6_4)
        x_6 = torch.add(x_6, x6_5)
        x_6 = torch.add(x_6, x6_6)
        x_6 = torch.add(x_6, x6_7)
        x_6 = torch.add(x_6, outlayerm[::1, 0, ::1, ::1, ::1])
        ym = torch.sqrt(torch.square(x_1)+2*torch.square(x_2)+2*torch.square(x_3)+torch.square(x_4)+2*torch.square(x_5)+torch.square(x_6))
        ym = torch.unsqueeze(ym, 1)
        return y*ym


# now we can declare the inputs
datasize = 1
spacialdim = 16
delta = np.ones((datasize, 8, spacialdim, spacialdim, spacialdim))
delta2 = np.ones((datasize, 8, spacialdim, spacialdim, spacialdim))
delta = delta*(2*np.pi/1024*16)**2
train_load = np.zeros((datasize, 10, 11, spacialdim, spacialdim, spacialdim))
test_load = np.zeros((datasize, 1, 9, spacialdim, spacialdim, spacialdim))
test_load_new = np.zeros((datasize, 1, 6, spacialdim, spacialdim, spacialdim))
test_load_struc = np.zeros((datasize, 1, 9, spacialdim, spacialdim, spacialdim))
test_load_struc_new = np.zeros((datasize, 1, 6, spacialdim, spacialdim, spacialdim))
strain2 = np.zeros((datasize, spacialdim, spacialdim, spacialdim, 3, 3))
strain2nonorm = np.zeros((datasize, spacialdim, spacialdim, spacialdim, 3, 3))
strain2nonorm_l= np.zeros((datasize, spacialdim, spacialdim, spacialdim, 3, 3))
strain2_m = np.zeros((datasize, spacialdim, spacialdim, spacialdim, 3, 3))
rot2 = np.zeros((datasize, spacialdim, spacialdim, spacialdim, 3, 3))
rot = np.zeros((datasize, spacialdim, spacialdim, spacialdim, 3, 3))
strainrot_m = np.zeros((datasize, spacialdim, spacialdim, spacialdim, 3, 3))
rotstrainrot_m = np.zeros((datasize, spacialdim, spacialdim, spacialdim, 3, 3))
strain2rot_m = np.zeros((datasize, spacialdim, spacialdim, spacialdim, 3, 3))
rotstrain2_m = np.zeros((datasize, spacialdim, spacialdim, spacialdim, 3, 3))
final = np.zeros((datasize, 9, spacialdim, spacialdim, spacialdim))
scalef = np.zeros((datasize, spacialdim, spacialdim, spacialdim))

# load the inputs
i = 0
for j in range(datasize):
   file_name = 'train_' + str(j) + '.h5'
   f = h5py.File(file_name, 'r')
   train_load[i, ::1, ::1, ::1, ::1, ::1] = f['dataset_JHTB']
   file_name = 'gt_' + str(j) + '.h5'
   f = h5py.File(file_name, 'r')
   test_load[i, 0, ::1, ::1, ::1, ::1] = f['dataset_JHTB']
   file_name = 'scalef_' + str(j) + '.h5'
   f = h5py.File(file_name, 'r')
   scalef[i, ::1, ::1, ::1] = f['dataset_JHTB']
   file_name = 'strain' + str(j) + '.h5'
   f = h5py.File(file_name, 'r')
   strain2[i, ::1, ::1, ::1, ::1, ::1] = f['dataset_JHTB']
   file_name = 'strainnonorm' + str(j) + '.h5'
   f = h5py.File(file_name, 'r')
   strain2nonorm_l[i, ::1, ::1, ::1, ::1, ::1] = f['dataset_JHTB']
   file_name = 'strainsq' + str(j) + '.h5'
   f = h5py.File(file_name, 'r')
   strain2_m[i, ::1, ::1, ::1, ::1, ::1] = f['dataset_JHTB']
   file_name = 'rotsq' + str(j) + '.h5'
   f = h5py.File(file_name, 'r')
   rot2[i, ::1, ::1, ::1, ::1, ::1] = f['dataset_JHTB']
   file_name = 'strainrot' + str(j) + '.h5'
   f = h5py.File(file_name, 'r')
   strainrot_m[i, ::1, ::1, ::1, ::1, ::1] = f['dataset_JHTB']
   file_name = 'rotstrainrot' + str(j) + '.h5'
   f = h5py.File(file_name, 'r')
   rotstrainrot_m[i, ::1, ::1, ::1, ::1, ::1] = f['dataset_JHTB']
   file_name = 'strain2rot' + str(j) + '.h5'
   f = h5py.File(file_name, 'r')
   strain2rot_m[i, ::1, ::1, ::1, ::1, ::1] = f['dataset_JHTB']
   file_name = 'rotstrain2' + str(j) + '.h5'
   f = h5py.File(file_name, 'r')
   rotstrain2_m[i, ::1, ::1, ::1, ::1, ::1] = f['dataset_JHTB']
   file_name = 'rot' + str(j) + '.h5'
   f = h5py.File(file_name, 'r')
   rot[i, ::1, ::1, ::1, ::1, ::1] = f['dataset_JHTB']
   i = i + 1

# ensure trace is divergence free
train_load[::1, ::1, 0, ::1, ::1, ::1] = np.zeros((train_load.shape[0], 10, spacialdim, spacialdim, spacialdim))


test_load_new = np.zeros((datasize, 1, 6, spacialdim, spacialdim, spacialdim))
test_load_new[::1, 0, 0, ::1, ::1, ::1] = test_load[::1, 0, 0, ::1, ::1, ::1]
test_load_new[::1, 0, 1, ::1, ::1, ::1] = test_load[::1, 0, 1, ::1, ::1, ::1]
test_load_new[::1, 0, 2, ::1, ::1, ::1] = test_load[::1, 0, 2, ::1, ::1, ::1]
test_load_new[::1, 0, 3, ::1, ::1, ::1] = test_load[::1, 0, 4, ::1, ::1, ::1]
test_load_new[::1, 0, 4, ::1, ::1, ::1] = test_load[::1, 0, 5, ::1, ::1, ::1]
test_load_new[::1, 0, 5, ::1, ::1, ::1] = test_load[::1, 0, 8, ::1, ::1, ::1]

# deviatoric
trace = test_load_new[::1, 0, 0, ::1, ::1, ::1]+test_load_new[::1, 0, 3, ::1, ::1, ::1]+test_load_new[::1, 0, 5, ::1, ::1, ::1]
test_load_new[::1, 0, 0, ::1, ::1, ::1] = test_load_new[::1, 0, 0, ::1, ::1, ::1]-1/3*trace
test_load_new[::1, 0, 3, ::1, ::1, ::1] = test_load_new[::1, 0, 3, ::1, ::1, ::1]-1/3*trace
test_load_new[::1, 0, 5, ::1, ::1, ::1] = test_load_new[::1, 0, 5, ::1, ::1, ::1]-1/3*trace

# intialize the bandlimiting
k_args_x = {'dt_min': 2 * np.pi / 64, 'dt_max': 2 * np.pi / 64, 'deterministic': True, 'dt_tie': True, 'bandlimit': 0.1}
k_args_y = {'dt_min': 2 * np.pi / 64, 'dt_max': 2 * np.pi / 64, 'deterministic': True, 'dt_tie': True, 'bandlimit': 0.1}
k_args_z = {'dt_min': 2 * np.pi / 64, 'dt_max': 2 * np.pi / 64, 'deterministic': True, 'dt_tie': True, 'bandlimit': 0.1}
k_args = [k_args_x, k_args_y, k_args_z]

k_args2_x = {'dt_min': 2 * np.pi / 32, 'dt_max': 2 * np.pi / 32, 'deterministic': True, 'dt_tie': True, 'bandlimit': 0.1}
k_args2_y = {'dt_min': 2 * np.pi / 32, 'dt_max': 2 * np.pi / 32, 'deterministic': True, 'dt_tie': True, 'bandlimit': 0.1}
k_args2_z = {'dt_min': 2 * np.pi / 32, 'dt_max': 2 * np.pi / 32, 'deterministic': True, 'dt_tie': True, 'bandlimit': 0.1}
k_args2 = [k_args2_x, k_args2_y, k_args2_z]

# now create the model
model2 = UnetTBNN(k_args, k_args2).to(device)
model2.load_state_dict(torch.load('S4ND_model'))
model2.eval()

# all inputs to GPU
train_loadnorm = torch.from_numpy(train_load).to(device, dtype=torch.float)
strain2nonorm = torch.from_numpy(strain2nonorm).to(device, dtype=torch.float)
strain2 = torch.from_numpy(strain2).to(device, dtype=torch.float)
strain2_m = torch.from_numpy(strain2_m).to(device, dtype=torch.float)
rot2 = torch.from_numpy(rot2).to(device, dtype=torch.float)
rot = torch.from_numpy(rot).to(device, dtype=torch.float)
strainrot_m = torch.from_numpy(strainrot_m).to(device, dtype=torch.float)
rotstrainrot_m = torch.from_numpy(rotstrainrot_m).to(device, dtype=torch.float)
strain2rot_m = torch.from_numpy(strain2rot_m).to(device, dtype=torch.float)
rotstrain2_m = torch.from_numpy(rotstrain2_m).to(device, dtype=torch.float)
delta = torch.from_numpy(delta).to(device, dtype=torch.float)
delta2 = torch.from_numpy(delta2).to(device, dtype=torch.float)
test_final = np.zeros((1, 6, spacialdim, spacialdim, spacialdim))

k_args = [{k: torch.tensor(v).to(device=device, non_blocking=True) for k, v in _data.items()} for _data in k_args]
k_args2 = [{k: torch.tensor(v).to(device=device, non_blocking=True) for k, v in _data.items()} for _data in k_args2]

# multiply the non-dimensional output with the dimensional scaling
test = scalef[0:1:1, ::1, ::1, ::1]
test2 = model2(train_loadnorm[0:1:1, 9, 0:11, ::1, ::1, ::1], strain2[0:1:1, ::1, ::1, ::1, ::1, ::1],
               strain2_m[0:1:1, ::1, ::1, ::1, ::1, ::1], rot2[0:1:1, ::1, ::1, ::1, ::1, ::1],
               strainrot_m[0:1:1, ::1, ::1, ::1, ::1, ::1], rotstrainrot_m[0:1:1, ::1, ::1, ::1, ::1, ::1],
               strain2rot_m[0:1:1, ::1, ::1, ::1, ::1, ::1], rotstrain2_m[0:1:1, ::1, ::1, ::1, ::1, ::1],
               delta2[0:1:1, ::1, ::1, ::1, ::1], rot[0:1:1, ::1, ::1, ::1, ::1, ::1], k_args, k_args2)
test2 = test2.detach().cpu().numpy()
test = test*test2

# make deviatoric
trace = test[::1, 0, ::1, ::1, ::1] + test[::1, 3, ::1, ::1, ::1] + test[::1, 5, ::1, ::1, ::1]
test[::1, 0, ::1, ::1, ::1] = test[::1, 0, ::1, ::1, ::1] - 1 / 3 * trace
test[::1, 3, ::1, ::1, ::1] = test[::1, 3, ::1, ::1, ::1] - 1 / 3 * trace
test[::1, 5, ::1, ::1, ::1] = test[::1, 5, ::1, ::1, ::1] - 1 / 3 * trace

# the GNN prediction
from mpl_toolkits.axes_grid1 import make_axes_locatable
fig3 = plt.figure(figsize = (15, 15))
a = fig3.add_subplot(121)
im2=a.imshow(test[0,0,::1,::1,1],
         extent = [0, 3.14*2/4, 0, 3.14*2/4],
         interpolation=None,
         cmap='bwr',
         vmin=-0.04, vmax=0.04)
divider = make_axes_locatable(a)
cax = divider.append_axes('right', size='10%', pad=0.05)
fig3.colorbar(im2,cax=cax)

# The ground truth
from mpl_toolkits.axes_grid1 import make_axes_locatable
fig3 = plt.figure(figsize = (15, 15))
a = fig3.add_subplot(121)
im2=a.imshow(test_load_new[0,0,0,::1,::1,1],
         extent = [0, 3.14*2/4, 0, 3.14*2/4],
         interpolation=None,
         cmap='bwr',
         vmin=-0.04, vmax=0.04)
divider = make_axes_locatable(a)
cax = divider.append_axes('right', size='10%', pad=0.05)
fig3.colorbar(im2,cax=cax)
plt.show()

