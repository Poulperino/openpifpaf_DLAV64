import logging
import sys
from copy import deepcopy

sys.path.append('./')  # to run '$ python *.py' files in subdirectories
logger = logging.getLogger(__name__)
import torch
from .common import *
from .experimental import *
from ..utils.general import make_divisible
from ..utils.torch_utils import time_synchronized, model_info, scale_img, initialize_weights, \
    copy_attr
import openpifpaf

try:
    import thop  # for FLOPS computation
except ImportError:
    thop = None

class DictionarizeFPN(nn.Module):
    def __init__(self, ch=3):
        super(DictionarizeFPN, self).__init__()

    def forward(self, x):
        assert isinstance(x, list)
        dict_out = {}
        for i in range(len(x)):
            dict_out[i] = x[i]
        return dict_out

class Model(openpifpaf.network.BaseNetwork):
    def __init__(self, cfg='yolor-csp-c.yaml', ch=3, nc=None, anchors=None):  # model, input channels, number of classes
        if isinstance(cfg, dict):
            yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            yaml_file = Path(cfg).name
            with open(cfg) as f:
                yaml = yaml.load(f, Loader=yaml.SafeLoader)  # model dict
        super(Model, self).__init__(Path(cfg).stem, stride=32, out_features=yaml['out_features'])
        self.traced = False
        self.yaml = yaml

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        # if nc and nc != self.yaml['nc']:
        #     logger.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
        #     self.yaml['nc'] = nc  # override yaml value
        # if anchors:
        #     logger.info(f'Overriding model.yaml anchors with anchors={anchors}')
        #     self.yaml['anchors'] = round(anchors)  # override yaml value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        # self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        # print([x.shape for x in self.forward(torch.zeros(1, ch, 64, 64))])

        # # Build strides, anchors
        # m = self.model[-1]  # Detect()
        # if isinstance(m, Detect):
        #     s = 256  # 2x min stride
        #     m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
        #     check_anchor_order(m)
        #     m.anchors /= m.stride.view(-1, 1, 1)
        #     self.stride = m.stride
        #     self._initialize_biases()  # only run once
        #     # print('Strides: %s' % m.stride.tolist())
        # if isinstance(m, IDetect):
        #     s = 256  # 2x min stride
        #     m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
        #     check_anchor_order(m)
        #     m.anchors /= m.stride.view(-1, 1, 1)
        #     self.stride = m.stride
        #     self._initialize_biases()  # only run once
        #     # print('Strides: %s' % m.stride.tolist())
        # if isinstance(m, IAuxDetect):
        #     s = 256  # 2x min stride
        #     m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))[:4]])  # forward
        #     #print(m.stride)
        #     check_anchor_order(m)
        #     m.anchors /= m.stride.view(-1, 1, 1)
        #     self.stride = m.stride
        #     self._initialize_aux_biases()  # only run once
        #     # print('Strides: %s' % m.stride.tolist())
        # if isinstance(m, IBin):
        #     s = 256  # 2x min stride
        #     m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
        #     check_anchor_order(m)
        #     m.anchors /= m.stride.view(-1, 1, 1)
        #     self.stride = m.stride
        #     self._initialize_biases_bin()  # only run once
        #     # print('Strides: %s' % m.stride.tolist())
        # if isinstance(m, IKeypoint):
        #     s = 256  # 2x min stride
        #     m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
        #     check_anchor_order(m)
        #     m.anchors /= m.stride.view(-1, 1, 1)
        #     self.stride = m.stride
        #     self._initialize_biases_kpt()  # only run once
        #     # print('Strides: %s' % m.stride.tolist())

        # Init weights, biases
        initialize_weights(self)
        self.info()
        logger.info('')

    def forward(self, x, augment=False, profile=False):
        if augment:
            img_size = x.shape[-2:]  # height, width
            s = [1, 0.83, 0.67]  # scales
            f = [None, 3, None]  # flips (2-ud, 3-lr)
            y = []  # outputs
            for si, fi in zip(s, f):
                xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
                yi = self.forward_once(xi)[0]  # forward
                # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
                yi[..., :4] /= si  # de-scale
                if fi == 2:
                    yi[..., 1] = img_size[0] - yi[..., 1]  # de-flip ud
                elif fi == 3:
                    yi[..., 0] = img_size[1] - yi[..., 0]  # de-flip lr
                y.append(yi)
            return torch.cat(y, 1), None  # augmented inference, train
        else:
            return self.forward_once(x, profile)  # single-scale inference, train

    def forward_once(self, x, profile=False):
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            if not hasattr(self, 'traced'):
                self.traced=False

            if self.traced:
                if isinstance(m, Detect) or isinstance(m, IDetect) or isinstance(m, IAuxDetect) or isinstance(m, IKeypoint):
                    break

            if profile:
                c = isinstance(m, (Detect, IDetect, IAuxDetect, IBin))
                o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPS
                for _ in range(10):
                    m(x.copy() if c else x)
                t = time_synchronized()
                for _ in range(10):
                    m(x.copy() if c else x)
                dt.append((time_synchronized() - t) * 100)
                print('%10.1f%10.0f%10.1fms %-40s' % (o, m.np, dt[-1], m.type))

            x = m(x)  # run

            y.append(x if m.i in self.save else None)  # save output

        if profile:
            print('%.1fms total' % sum(dt))
        return x

    # def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
    #     # https://arxiv.org/abs/1708.02002 section 3.3
    #     # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
    #     m = self.model[-1]  # Detect() module
    #     for mi, s in zip(m.m, m.stride):  # from
    #         b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
    #         b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
    #         b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
    #         mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
    #
    # def _initialize_aux_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
    #     # https://arxiv.org/abs/1708.02002 section 3.3
    #     # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
    #     m = self.model[-1]  # Detect() module
    #     for mi, mi2, s in zip(m.m, m.m2, m.stride):  # from
    #         b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
    #         b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
    #         b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
    #         mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
    #         b2 = mi2.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
    #         b2.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
    #         b2.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
    #         mi2.bias = torch.nn.Parameter(b2.view(-1), requires_grad=True)
    #
    # def _initialize_biases_bin(self, cf=None):  # initialize biases into Detect(), cf is class frequency
    #     # https://arxiv.org/abs/1708.02002 section 3.3
    #     # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
    #     m = self.model[-1]  # Bin() module
    #     bc = m.bin_count
    #     for mi, s in zip(m.m, m.stride):  # from
    #         b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
    #         old = b[:, (0,1,2,bc+3)].data
    #         obj_idx = 2*bc+4
    #         b[:, :obj_idx].data += math.log(0.6 / (bc + 1 - 0.99))
    #         b[:, obj_idx].data += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
    #         b[:, (obj_idx+1):].data += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
    #         b[:, (0,1,2,bc+3)].data = old
    #         mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
    #
    # def _initialize_biases_kpt(self, cf=None):  # initialize biases into Detect(), cf is class frequency
    #     # https://arxiv.org/abs/1708.02002 section 3.3
    #     # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
    #     m = self.model[-1]  # Detect() module
    #     for mi, s in zip(m.m, m.stride):  # from
    #         b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
    #         b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
    #         b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
    #         mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
    #
    # def _print_biases(self):
    #     m = self.model[-1]  # Detect() module
    #     for mi in m.m:  # from
    #         b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
    #         print(('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             print('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    # def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
    #     print('Fusing layers... ')
    #     for m in self.model.modules():
    #         if isinstance(m, RepConv):
    #             #print(f" fuse_repvgg_block")
    #             m.fuse_repvgg_block()
    #         elif isinstance(m, RepConv_OREPA):
    #             #print(f" switch_to_deploy")
    #             m.switch_to_deploy()
    #         elif type(m) is Conv and hasattr(m, 'bn'):
    #             m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
    #             delattr(m, 'bn')  # remove batchnorm
    #             m.forward = m.fuseforward  # update forward
    #         elif isinstance(m, (IDetect, IAuxDetect)):
    #             m.fuse()
    #             m.forward = m.fuseforward
    #     self.info()
    #     return self

    def nms(self, mode=True):  # add or remove NMS module
        present = type(self.model[-1]) is NMS  # last layer is NMS
        if mode and not present:
            print('Adding NMS... ')
            m = NMS()  # module
            m.f = -1  # from
            m.i = self.model[-1].i + 1  # index
            self.model.add_module(name='%s' % m.i, module=m)  # add
            self.eval()
        elif not mode and present:
            print('Removing NMS... ')
            self.model = self.model[:-1]  # remove
        return self

    def autoshape(self):  # add autoShape module
        print('Adding autoShape... ')
        m = autoShape(self)  # wrap model
        copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())  # copy attributes
        return m

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)


def parse_model(d, ch):  # model_dict, input_channels(3)
    logger.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    # anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    gd, gw = d['depth_multiple'], d['width_multiple']
    # na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = -1#na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [nn.Conv2d, Conv, RobustConv, RobustConv2, DWConv, GhostConv, RepConv, RepConv_OREPA, DownC,
                 SPP, SPPF, SPPCSPC, GhostSPPCSPC, MixConv2d, Focus, Stem, GhostStem, CrossConv,
                 Bottleneck, BottleneckCSPA, BottleneckCSPB, BottleneckCSPC,
                 RepBottleneck, RepBottleneckCSPA, RepBottleneckCSPB, RepBottleneckCSPC,
                 Res, ResCSPA, ResCSPB, ResCSPC,
                 RepRes, RepResCSPA, RepResCSPB, RepResCSPC,
                 ResX, ResXCSPA, ResXCSPB, ResXCSPC,
                 RepResX, RepResXCSPA, RepResXCSPB, RepResXCSPC,
                 Ghost, GhostCSPA, GhostCSPB, GhostCSPC,
                 SwinTransformerBlock, STCSPA, STCSPB, STCSPC,
                 SwinTransformer2Block, ST2CSPA, ST2CSPB, ST2CSPC]:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in [DownC, SPPCSPC, GhostSPPCSPC,
                     BottleneckCSPA, BottleneckCSPB, BottleneckCSPC,
                     RepBottleneckCSPA, RepBottleneckCSPB, RepBottleneckCSPC,
                     ResCSPA, ResCSPB, ResCSPC,
                     RepResCSPA, RepResCSPB, RepResCSPC,
                     ResXCSPA, ResXCSPB, ResXCSPC,
                     RepResXCSPA, RepResXCSPB, RepResXCSPC,
                     GhostCSPA, GhostCSPB, GhostCSPC,
                     STCSPA, STCSPB, STCSPC,
                     ST2CSPA, ST2CSPB, ST2CSPC]:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[x] for x in f])
        elif m is Chuncat:
            c2 = sum([ch[x] for x in f])
        elif m is Shortcut:
            c2 = ch[f[0]]
        elif m is Foldcut:
            c2 = ch[f] // 2
            # elif m in [Detect, IDetect, IAuxDetect, IBin, IKeypoint]:
            #     args.append([ch[x] for x in f])
            #     if isinstance(args[1], int):  # number of anchors
            #         args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is ReOrg:
            c2 = ch[f] * 4
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        elif m is DictionarizeFPN:
            c2 = sum([ch[x] for x in f])
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        logger.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)
