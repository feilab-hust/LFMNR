"""
Create embedders.
    * positional encoder
    * MVSNeRF-like feature embedder
"""
import torch
from core.utils.LF import back_project
from core.utils.Coord import Coord
from glbSettings import *
import numpy as np

class Embedder:
    @staticmethod
    def get_by_name(embedder_type:str=None, *args, **kwargs):
        if embedder_type is None or embedder_type == 'None':
            embedder_type = "BasicEmbedder"
        try:
            embedder = eval(embedder_type)
        except Exception as e:
            print(e)
            raise ValueError(f"Type {embedder_type} not recognized!")
        embedder = embedder(*args, **kwargs)
        return embedder

class BasicEmbedder:
    """
    An embedder that do nothing to the input tensor.
    Used as a template.
    """
    def __init__(self, *args, **kwargs):
        """To be overwritten"""
        del args, kwargs
    
    def embed(self, inputs):
        """To be overwritten"""
        return inputs


class RadiaPosiEncoder(BasicEmbedder):
    def __init__(self, multires=0, multiresZ=None, input_dim=3, embed_bases=[torch.sin, torch.cos], include_input=True,
                 device=DEVICE, *args, **kwargs):
        del args, kwargs
        self.multires = multires
        if input_dim > 2 and multiresZ is not None:
            self.multiresZ = multiresZ
        self.in_dim = input_dim
        self.embed_bases = embed_bases
        self.include_input = include_input
        self.device = device
        self.embed_fns = []
        self.out_dim = 0
        self._create_embed_fn()
        self.embed = self._embed_iso if not hasattr(self, 'embed_fns_Z') else self._embed_aniso

    def _create_embed_fn(self, use_pi=False):
        embed_fns = []
        d = self.in_dim if not hasattr(self, 'multiresZ') else self.in_dim - 1
        out_dim = 0
        if self.include_input:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.multires - 1
        N_freqs = self.multires
        freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs, device=self.device)

        # radia pe
        dia_digree = 45

        s = np.sin(np.arange(0, 180, dia_digree) * np.pi / 180)[
            :, np.newaxis
            ]
        c = np.cos(np.arange(0, 180, dia_digree) * np.pi / 180)[
            :, np.newaxis
            ]


        fourier_mapping = np.concatenate((s, c), axis=1).T
        fourier_mapping = torch.from_numpy(np.float32(fourier_mapping)).to(self.device)
        radia_dim = len(s)
        # xy_freq = tf.matmul(in_node[:, :2], fourier_mapping)

        for freq in freq_bands:
            for p_fn in self.embed_bases:
                # embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * torch.pi*freq))
                embed_fns.append(
                    lambda x, p_fn=p_fn, freq=freq: p_fn(torch.matmul(x, fourier_mapping) * torch.pi * freq))
                # embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x*torch.pi* freq))
                out_dim += radia_dim

        if hasattr(self, 'multiresZ'):
            embed_fns_Z = []
            d = 1
            if self.include_input:
                embed_fns_Z.append(lambda x: x)
                out_dim += d

            max_freq_Z = self.multiresZ - 1
            N_freqs_Z = self.multiresZ
            freq_bands_Z = 2. ** torch.linspace(0., max_freq_Z, steps=N_freqs_Z, device=self.device)

            for freqZ in freq_bands_Z:
                for p_fn in self.embed_bases:
                    if use_pi:
                        embed_fns_Z.append(lambda x, p_fn=p_fn, freq=freqZ: p_fn(x * torch.pi * freq))
                    else:
                        embed_fns_Z.append(lambda x, p_fn=p_fn, freq=freqZ: p_fn(x * freq))
                    out_dim += d
            self.embed_fns_Z = embed_fns_Z

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def _embed_iso(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

    def _embed_aniso(self, inputs):
        embeded = self._embed_iso(inputs[..., :-1])
        embeded_Z = torch.cat([fn(inputs[..., -1:]) for fn in self.embed_fns_Z], -1)
        return torch.cat([embeded, embeded_Z], -1)




class PositionalEncoder(BasicEmbedder):
    def __init__(self, multires=0, multiresZ = None, input_dim=3, embed_bases=[torch.sin,torch.cos], include_input=True, device=DEVICE, *args, **kwargs):
        del args, kwargs
        self.multires = multires
        if input_dim > 2 and multiresZ is not None:
            self.multiresZ = multiresZ
        self.in_dim = input_dim
        self.embed_bases = embed_bases
        self.include_input = include_input
        self.device = device
        self.embed_fns = []
        self.out_dim = 0
        self._create_embed_fn()
        self.embed = self._embed_iso if not hasattr(self, 'embed_fns_Z') else self._embed_aniso

    def _create_embed_fn(self, use_pi=False):
        embed_fns = []
        d = self.in_dim if not hasattr(self, 'multiresZ') else self.in_dim-1
        out_dim = 0
        if self.include_input:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.multires-1
        N_freqs = self.multires
        freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs, device=self.device)
            
        for freq in freq_bands:
            for p_fn in self.embed_bases:
                if use_pi:
                    embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * torch.pi * freq))
                else:
                    embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        if hasattr(self, 'multiresZ'):
            embed_fns_Z = [] 
            d = 1
            if self.include_input:
                embed_fns_Z.append(lambda x : x)
                out_dim += d 
                
            max_freq_Z = self.multiresZ-1
            N_freqs_Z = self.multiresZ
            freq_bands_Z = 2.**torch.linspace(0., max_freq_Z, steps=N_freqs_Z, device=self.device)

            for freqZ in freq_bands_Z:
                for p_fn in self.embed_bases:
                    if use_pi:
                        embed_fns_Z.append(lambda x, p_fn=p_fn, freq=freqZ : p_fn(x * torch.pi * freq))
                    else:
                        embed_fns_Z.append(lambda x, p_fn=p_fn, freq=freqZ: p_fn(x * freq))
                    out_dim += d
            self.embed_fns_Z = embed_fns_Z
        
        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def _embed_iso(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

    def _embed_aniso(self, inputs):
        embeded = self._embed_iso(inputs[...,:-1])
        embeded_Z = torch.cat([fn(inputs[...,-1:]) for fn in self.embed_fns_Z], -1)
        return torch.cat([embeded, embeded_Z], -1)


