"""Head networks."""

import logging
import re

import torch
from openpifpaf.network import HeadNetwork
import openpifpaf

LOG = logging.getLogger(__name__)

class CompositeField(HeadNetwork):
    dropout_p = 0.0
    quad = 0

    def __init__(self, meta: openpifpaf.headmeta.Base,
                 in_features, *,
                 kernel_size=1, padding=0, dilation=1):
        super().__init__(meta, in_features)

        n_fields = meta.n_fields
        n_confidences = meta.n_confidences
        n_vectors = meta.n_vectors
        n_scales = meta.n_scales

        LOG.debug('%s config: fields = %d, confidences = %d, vectors = %d, scales = %d '
                  'kernel = %d, padding = %d, dilation = %d',
                  meta.name, n_fields, n_confidences, n_vectors, n_scales,
                  kernel_size, padding, dilation)

        self.shortname = meta.name
        self.dilation = dilation

        self.dropout = torch.nn.Dropout2d(p=self.dropout_p)
        self._quad = self.quad

        # classification
        out_features = n_fields * (4 ** self._quad)
        self.class_convs = torch.nn.ModuleList([
            torch.nn.Conv2d(in_features, out_features,
                            kernel_size, padding=padding, dilation=dilation)
            for _ in range(n_confidences)
        ])

        # regression
        self.reg_convs = torch.nn.ModuleList([
            torch.nn.Conv2d(in_features, 2 * out_features,
                            kernel_size, padding=padding, dilation=dilation)
            for _ in range(n_vectors)
        ])
        self.reg_spreads = torch.nn.ModuleList([
            torch.nn.Conv2d(in_features, out_features,
                            kernel_size, padding=padding, dilation=dilation)
            for _ in self.reg_convs
        ])

        # scale
        self.scale_convs = torch.nn.ModuleList([
            torch.nn.Conv2d(in_features, out_features,
                            kernel_size, padding=padding, dilation=dilation)
            for _ in range(n_scales)
        ])

        # dequad
        self.dequad_op = torch.nn.PixelShuffle(2)

        #self.init_weights()

    def init_weights(self, pi_focal=0.01):
        import numpy as np
        m = list(self.class_convs.modules())[-1]
        if isinstance(m, torch.nn.Conv2d):
            # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            torch.nn.init.normal_(m.weight, std=0.01)
            for name, _ in m.named_parameters():
                if name in ['bias']:
                    torch.nn.init.constant_(m.bias, -np.log((1-pi_focal)/pi_focal))

    def forward(self, x):  # pylint: disable=arguments-differ
        x = self.dropout(x)

        # classification
        classes_x = [class_conv(x) for class_conv in self.class_convs]
        if not self.training:
            classes_x = [torch.sigmoid(class_x) for class_x in classes_x]

        # regressions
        regs_x = [reg_conv(x) * self.dilation for reg_conv in self.reg_convs]
        regs_x_spread = [reg_spread(x) for reg_spread in self.reg_spreads]
        regs_x_spread = [torch.nn.functional.leaky_relu(x + 2.0) - 2.0
                         for x in regs_x_spread]

        # scale
        scales_x = [scale_conv(x) for scale_conv in self.scale_convs]
        scales_x = [torch.nn.functional.relu(scale_x) for scale_x in scales_x]

        # upscale
        for _ in range(self._quad):
            classes_x = [self.dequad_op(class_x)[:, :, :-1, :-1]
                         for class_x in classes_x]
            regs_x = [self.dequad_op(reg_x)[:, :, :-1, :-1]
                      for reg_x in regs_x]
            regs_x_spread = [self.dequad_op(reg_x_spread)[:, :, :-1, :-1]
                             for reg_x_spread in regs_x_spread]
            scales_x = [self.dequad_op(scale_x)[:, :, :-1, :-1]
                        for scale_x in scales_x]
        # reshape regressions
        regs_x = [
            reg_x.reshape(reg_x.shape[0],
                          reg_x.shape[1] // 2,
                          2,
                          reg_x.shape[2],
                          reg_x.shape[3])
            for reg_x in regs_x
        ]

        return classes_x + regs_x + regs_x_spread + scales_x
