import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from core.models.Embedder import *
from core.utils.Coord import Coord
from core.utils.misc import strsort
from core.models.layers import *
from glbSettings import *


class Model:
    def __init__(self, model_type:str, compile=False, *args, **kwargs):
        self.type = model_type
        self.model = self.get_by_name(model_type, *args, **kwargs)
        # import torch
        if compile:
            try:
                self.model = torch.compile(self.model)
            except Exception as e:
                print(f"Unable to compile model, using uncompiled one.")
                compile = False
                print(e)
        self.compile = compile

    def get_by_name(self, model_type:str, *args, **kwargs):
        try:
            model = eval(model_type)
        except:
            raise ValueError(f"Type {model_type} not recognized!")
        model = model(*args, **kwargs)
        return model

    def run(self, x:torch.tensor, embedder, post, model_kwargs:dict) -> torch.tensor:
        """
        x: (...,3),  [x,y,z]
        model_kwargs: additional inputs for embedder and post processor
        """
        embedder_kwargs = model_kwargs['embedder']
        post_kwargs = model_kwargs['post']
        h = x
        h = embedder.embed(h, **embedder_kwargs)
        h = self.model(h)
        h = post(h, **post_kwargs)
        return h
    
    def get_model(self):
        return self.model

    def get_state(self):
        return self.model.get_state()

    def load(self, ckpt):
        self.model.load_params(ckpt)

class BasicModel(nn.Module):
    """Basic template for creating models."""
    def __init__(self):
        super().__init__()

    def forward(self):
        """To be overwritten"""
        raise NotImplementedError
    
    def load_params(self, path:str):
        """To be overwritten"""
        raise NotImplementedError

    def get_state(self):
        """To be overwritten"""
        raise NotImplementedError

class NeRF(BasicModel):
    """Standard NeRF model"""
    def __init__(self, D=8, W=256, input_ch=3, output_ch=1, skips=[4], negative_slope=0.0, device=DEVICE, *args, **kwargs):
        del args, kwargs
        super().__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.output_ch = output_ch
        self.skips = skips
        self.negative_slope = negative_slope

        layers = [nn.Linear(input_ch, W)]
        for i in range(D-1):
            in_channels = W
            if i in self.skips:
                print('MLP ResConnect: %d/%d'%(i,D))
                in_channels += input_ch
            layers.append(nn.Linear(in_channels, W))
        self.pts_linear = nn.ModuleList(layers)
        self.output_linear = nn.Linear(W, output_ch)

        # self.pts_linear.apply(lambda m:weights_init(m, a=self.negative_slope)

        self.to(device)
        
    def forward(self, x):
        h = x
        for i,l in enumerate(self.pts_linear):
            h = l(h)
            h = F.relu(h)
            # h = F.leaky_relu(h, negative_slope=self.negative_slope)
            if i in self.skips:
                h = torch.cat([x, h], dim=-1)
        outputs = self.output_linear(h)
        return outputs
    
    def load_params(self, ckpt:dict):
        # Load model
        self.load_state_dict(ckpt['network_fn_state_dict'])

    def get_state(self):
        return self.state_dict()

class LeakyReluLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, **kwargs):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.leaky_relu = partial(F.leaky_relu, inplace = True) # saves a lot of memory
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            nn.init.uniform_(self.linear.bias, -0.05, 0.05)
            nn.init.kaiming_uniform_(self.linear.weight, mode = 'fan_in', nonlinearity = 'leaky_relu')

    def eval_(self):
        self.linear.weight.requires_grad = False
        self.linear.bias.requires_grad = False

    def forward(self, input):
            return self.leaky_relu(self.linear(input))

class NeRF_ex(BasicModel):
    def __init__(self,  D=8, W=256, input_ch=3, output_ch=1, skips=[4], negative_slope=0.0, device=DEVICE, *args, **kwargs):
        super().__init__()
        hidden_c = 12
        self.input_features = 3
        self.skips = [2]
        hidden_l = 4
        self.out_c = 1

        self.hash_table_l = list(np.around(np.array([340, 340, 120]) * np.array([0.6, 0.6, 0.5])).astype(np.uint8))
        self.table = nn.Parameter(1e-4 * (torch.rand(self.hash_table_l + [3]) * 2), requires_grad = True) # [H, W, D, 2 or 3]

        # table_aberation_xy = list(np.array([340, 340]) // 2) # adjust spatial res for aberration table
        # self.table_aberration = nn.Parameter(1e-4 * (torch.rand(table_aberation_xy + [3]) * 2 - 1), requires_grad = True) # H,W,24

        self.net_layers_real = [LeakyReluLayer(self.input_features, hidden_c)] + \
                               [LeakyReluLayer(hidden_c, hidden_c)
                                if i not in self.skips else
                                LeakyReluLayer(hidden_c + self.input_features, hidden_c) for i in range(hidden_l - 1)]
        # self.net_layers_aberration = [LeakyReluLayer(self.input_features, hidden_c)] + \
        #                              [LeakyReluLayer(hidden_c, hidden_c)
        #                               if i not in self.skips else
        #                               LeakyReluLayer(hidden_c + self.input_features, hidden_c) for i in range(hidden_l - 1)]

        self.output_layer_real = nn.Linear(hidden_c, self.out_c)

        # self.output_layer_aberration = nn.Linear(hidden_c, 1)

        self.init_weight_zero_bias_constant(self.output_layer_real)
        # self.init_weight_zero(self.output_layer_aberration)

        self.net_real = nn.ModuleList(self.net_layers_real)
        # self.net_aberration = nn.ModuleList(self.net_layers_aberration)

    def init_weight_zero(self, sub_net):
        nn.init.zeros_(sub_net.weight)
        nn.init.zeros_(sub_net.bias)

    def init_weight_zero_bias_constant(self, sub_net):
        nn.init.zeros_(sub_net.weight)
        nn.init.zeros_(sub_net.bias)

    def load_params(self, ckpt:dict):
        # Load model
        self.load_state_dict(ckpt)

    def forward(self, coords):
        H, W, D = 340, 340, 120
        table = F.interpolate(self.table[..., None].permute(3, 4, 0, 1, 2), [H, W, D], mode='trilinear',
                              align_corners=True).permute(2, 3, 4, 0, 1)[..., 0]
        table = table.view(-1, self.input_features)

        x_real = table
        for i, l in enumerate(self.net_real):
            x_real = l(x_real)
            if i in self.skips:
                x_real = torch.cat([table, x_real], -1)

        output = self.output_layer_real(x_real)
        output = F.tanh(output).view([340, 340, 120])[20:-20, 20:-20, :]

        # ''' operation for aberration  '''
        # table_aberration = self.table_aberration.view(-1, self.input_features)
        # aberration = table_aberration
        # for i, l in enumerate(self.net_aberration):
        #     aberration = l(aberration)
        #     if i in self.skips:
        #         aberration = torch.cat([table_aberration, aberration], -1)
        #
        # output_aberration = self.output_layer_aberration(aberration)
        # output_aberration = F.tanh(output_aberration)

        return (output + 1.33).permute(2,0,1)


def create_model(Flags, shape, img_ch, model_type='NeRF', embedder_type='PositionalEncoder',
                 post_processor_type='relu', lr=1e-4,
                 weights_path=None,
                 create_optimizer=True,
                 ):
    ## embedder
    embedder_kwargs = {
        # general
        'device': DEVICE, 
        'compile': Flags.compile,
        # PositionalEmbedder
        'multires': Flags.multires,
        'multiresZ': Flags.multiresZ,
        'input_dim': 3,     # input pts dim
        }
    embedder = Embedder.get_by_name(embedder_type, **embedder_kwargs)
    ## model
    model_kwargs={
                # NeRF
                'D': Flags.netdepth,
                'W': Flags.netwidth,
                'input_ch':  embedder.out_dim if hasattr(embedder, 'out_dim') else 3,
                'output_ch': Flags.sigch,
                'skips': [eval(_s) if isinstance(_s,str) else _s for _s in Flags.skips],
                'compile': Flags.compile,
        }
    model = Model(model_type=model_type, **model_kwargs)
    if create_optimizer:
        grad_vars = list(model.get_model().parameters())
        if hasattr(embedder, 'parameters') and callable(embedder.parameters):
            grad_vars += list(embedder.parameters())
        optimizer = torch.optim.Adam(params=grad_vars, lr=lr, betas=(0.9,0.999), capturable=(DEVICE.type == 'cuda'))
    start = 0
    
    ## post processor
    if post_processor_type == 'linear':
        post_processor = lambda x:x
    elif post_processor_type == 'relu':
        post_processor = torch.relu
    elif post_processor_type == 'leakyrelu':
        post_processor = F.leaky_relu
    else:
        raise ValueError(f"Type {post_processor_type} not recognized!")
    
    ## Load checkpoint
    if weights_path != None:
        ckpts = {key:torch.load(path, map_location=DEVICE) for key,path in weights_path.items() if path is not None and path.lower() != 'none'}
        if 'model' in ckpts: model.load(ckpts['model'])
        if 'embedder' in ckpts and hasattr(embedder, 'load') and callable(embedder.load): embedder.load(ckpts['embedder'])
        if 'post' in ckpts and hasattr(post_processor, 'load') and callable(post_processor.load): post_processor.load(ckpts['post'])
        if create_optimizer:
            if 'model' in ckpts: optimizer.load_state_dict(ckpts['model']['optimizer_state_dict'])
        start = ckpts['model']['global_step'] if 'global_step' in ckpts['model'].keys() else 0

    embedder_args = embedder.list_args_input() if hasattr(embedder, 'list_args_input') else {}
    post_args = post_processor.list_args_input() if hasattr(post_processor, 'list_args_input') else {}

    if create_optimizer:
        out = model, embedder, post_processor, optimizer, start, embedder_args, post_args
    else:
        out = model, embedder, post_processor, start, embedder_args, post_args
    return out



def loading_4D_weights(Flags, model_type='NeRF', embedder_type='PositionalEncoder', post_processor_type='relu', weights_paras=None):
    embedder_kwargs = {
        # general
        'device': DEVICE,
        'compile': Flags.compile,
        # PositionalEmbedder
        'multires': Flags.multires,
        'multiresZ': Flags.multiresZ,
        'input_dim': 3,     # input pts dim
        }
    embedder = Embedder.get_by_name(embedder_type, **embedder_kwargs)
    ## model
    model_kwargs={
                # NeRF
                'D': Flags.netdepth,
                'W': Flags.netwidth,
                'input_ch':  embedder.out_dim if hasattr(embedder, 'out_dim') else 3,
                'output_ch': Flags.sigch,
                'skips': [eval(_s) if isinstance(_s,str) else _s for _s in Flags.skips],
        }

    model = Model(model_type=model_type, **model_kwargs)
    model.load(weights_paras)
    ## post processor
    if post_processor_type == 'linear':
        post_processor = lambda x:x
    elif post_processor_type == 'relu':
        post_processor = torch.relu
    elif post_processor_type == 'leakyrelu':
        post_processor = F.leaky_relu
    else:
        raise ValueError(f"Type {post_processor_type} not recognized!")

    return model, embedder, post_processor



def search_load_model(Flags, shape = None, create_ok:bool = True, create_optimizer:bool = True, initial_ckpt=False):
    if shape == None: shape = (0,0,0)
    C = Flags.sigch
    if Flags.weights_path is not None and Flags.weights_path != 'None' and initial_ckpt:
        keys = ['model']
        ckpts = {key: [val] for key, val in zip(keys, Flags.weights_path)}
        ckpts.update({'embedder':[]})
        ckpts.update({'post': []})

    else:
        ckpts = {
            'model': sorted([os.path.join(Flags.basedir, Flags.expname, f) for f in
                             os.listdir(os.path.join(Flags.basedir, Flags.expname)) if PFX_model in f],
                            key=strsort),
            'embedder': sorted([os.path.join(Flags.basedir, Flags.expname, f) for f in
                                os.listdir(os.path.join(Flags.basedir, Flags.expname)) if PFX_embedder in f],
                               key=strsort),
            'post': sorted([os.path.join(Flags.basedir, Flags.expname, f) for f in
                            os.listdir(os.path.join(Flags.basedir, Flags.expname)) if PFX_post in f], key=strsort)
        }
    print('Found ckpts', ckpts)

    ckpt_paths = {}
    for key, ckpt in ckpts.items():
        if len(ckpt) > 0 and not Flags.no_reload:
            ckpt_paths[key] = ckpt[-1]
    if len(ckpt_paths) == 0:
        if create_ok:
            ckpt_paths = None
            print("Creating new model...")
        else:
            print("No ckpts found. Do nothing.")
            return None
    else:
        print('Reloading from', ckpt_paths)

    models = create_model(Flags, shape, C, Flags.modeltype, Flags.embeddertype, Flags.postprocessortype,
                          lr=Flags.lrate, weights_path=ckpt_paths, create_optimizer=create_optimizer)
    return models

def weights_init(m, a=0.0):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data, a=a)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)



if __name__ == '__main__':
    model = NeRF()
    print(model.parameters)