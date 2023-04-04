import argparse
import functools
import logging
import math

import torch

from . import headmeta
from openpifpaf.network import HeadNetwork

LOG = logging.getLogger(__name__)


class Butterfly2Head(HeadNetwork):
    dropout_p = 0.0
    inplace_ops = True
    deep = False
    deep_separate = False

    def __init__(self,
                 meta: headmeta.Butterfly2,
                 in_features, *,
                 kernel_size=1, padding=0, dilation=1):
        super().__init__(meta, in_features)

        LOG.debug('%s config: fields = %d, confidences = %d, vectors = %d, scales = %d '
                  'kernel = %d, padding = %d, dilation = %d',
                  meta.name, meta.n_fields, meta.n_confidences, meta.n_vectors, meta.n_scales,
                  kernel_size, padding, dilation)

        self.dropout = torch.nn.Dropout2d(p=self.dropout_p)

        # convolution
        self.n_components = meta.n_confidences + meta.n_vectors * 2 + meta.n_scales

        if self.deep:
            self.conv = make_deep_layer(3, in_features, in_features, meta.n_fields * self.n_components * (meta.upsample_stride ** 2) , padding=padding, dilation=dilation)
        elif self.deep_separate:
            self.conv_confidences = [make_deep_layer(3, in_features, in_features, meta.n_fields  * (meta.upsample_stride ** 2), padding=padding, dilation=dilation).cuda() for _ in range(meta.n_confidences)]
            self.conv_vectors = [make_deep_layer(3, in_features, in_features, meta.n_fields * 2 * (meta.upsample_stride ** 2), padding=padding, dilation=dilation).cuda() for _ in range(meta.n_vectors)]
            self.conv_scales = [make_deep_layer(3, in_features, in_features, meta.n_fields * (meta.upsample_stride ** 2), padding=padding, dilation=dilation).cuda() for _ in range(meta.n_scales)]
        else:
            self.conv = torch.nn.Conv2d(
                in_features, meta.n_fields * self.n_components * (meta.upsample_stride ** 2),
                kernel_size, padding=padding, dilation=dilation,
            )

        # upsample
        assert meta.upsample_stride >= 1
        self.upsample_op = None
        if meta.upsample_stride > 1:
            self.upsample_op = torch.nn.PixelShuffle(meta.upsample_stride)

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group('CompositeField4')
        group.add_argument('--bf2-dropout', default=cls.dropout_p, type=float,
                           help='[experimental] zeroing probability of feature in head input')
        assert cls.inplace_ops
        group.add_argument('--bf2-no-inplace-ops', dest='bf2_inplace_ops',
                           default=True, action='store_false',
                           help='alternative graph without inplace ops')

        group.add_argument('--bf2-deep', dest='bf2_deep',
                           default=False, action='store_true',
                           help='use deep heads')

        group.add_argument('--bf2-deep-separate', dest='bf2_deep_separate',
                           default=False, action='store_true',
                           help='use deep heads separate per type of field')

    @classmethod
    def configure(cls, args: argparse.Namespace):
        cls.dropout_p = args.bf2_dropout
        cls.inplace_ops = args.bf2_inplace_ops
        cls.deep = args.bf2_deep
        cls.deep_separate = args.bf2_deep_separate

    @property
    def sparse_task_parameters(self):
        return [self.conv.weight]

    def forward(self, x):  # pylint: disable=arguments-differ
        x = self.dropout(x)
        if self.deep_separate:
            x_toConcat = []

            x_toConcat.append(self.conv_uncertainties(x))

            for conv_conf in self.conv_confidences:
                x_toConcat.append(conv_conf(x))

            for conv_vec in self.conv_vectors:
                x_toConcat.append(conv_vec(x))

            for conv_scal in self.conv_scales:
                x_toConcat.append(conv_scal(x))

            x = torch.cat(x_toConcat, dim=1)
        else:
            x = self.conv(x)
        # upscale
        if self.upsample_op is not None:
            x = self.upsample_op(x)
            low_cut = (self.meta.upsample_stride - 1) // 2
            high_cut = math.ceil((self.meta.upsample_stride - 1) / 2.0)
            if self.training:
                # negative axes not supported by ONNX TensorRT
                x = x[:, :, low_cut:-high_cut, low_cut:-high_cut]
            else:
                # the int() forces the tracer to use static shape
                x = x[:, :, low_cut:int(x.shape[2]) - high_cut, low_cut:int(x.shape[3]) - high_cut]

        # Extract some shape parameters once.
        # Convert to int so that shape is constant in ONNX export.
        x_size = x.size()
        batch_size = x_size[0]
        feature_height = int(x_size[2])
        feature_width = int(x_size[3])

        x = x.view(
            batch_size,
            self.meta.n_fields,
            self.n_components,
            feature_height,
            feature_width
        )

        if not self.training and self.inplace_ops:
            # classification
            classes_x = x[:, :, 1:1 + self.meta.n_confidences]
            torch.sigmoid_(classes_x)

            # regressions x: add index
            if self.meta.n_vectors > 0:
                index_field = index_field_torch((feature_height, feature_width), device=x.device)
                first_reg_feature = 1 + self.meta.n_confidences
                for i, do_offset in enumerate(self.meta.vector_offsets):
                    if not do_offset:
                        continue
                    reg_x = x[:, :, first_reg_feature + i * 2:first_reg_feature + (i + 1) * 2]
                    reg_x.add_(index_field)

            # scale
            first_scale_feature = 1 + self.meta.n_confidences + self.meta.n_vectors * 2
            scales_x = x[:, :, first_scale_feature:first_scale_feature + self.meta.n_scales]
            scales_x[:] = torch.nn.functional.softplus(scales_x)
        elif not self.training and not self.inplace_ops:
            # TODO: CoreMLv4 does not like strided slices.
            # Strides are avoided when switching the first and second dim
            # temporarily.
            x = torch.transpose(x, 1, 2)

            # width
            width_x = x[:, 0:1]

            # classification
            classes_x = x[:, 1:1 + self.meta.n_confidences]
            classes_x = torch.sigmoid(classes_x)

            # regressions x
            first_reg_feature = 1 + self.meta.n_confidences
            regs_x = [
                x[:, first_reg_feature + i * 2:first_reg_feature + (i + 1) * 2]
                for i in range(self.meta.n_vectors)
            ]
            # regressions x: add index
            index_field = index_field_torch(
                (feature_height, feature_width), device=x.device, unsqueeze=(1, 0))
            # TODO: coreml export does not work with the index_field creation in the graph.
            index_field = torch.from_numpy(index_field.numpy())
            regs_x = [reg_x + index_field if do_offset else reg_x
                      for reg_x, do_offset in zip(regs_x, self.meta.vector_offsets)]

            # scale
            first_scale_feature = 1 + self.meta.n_confidences + self.meta.n_vectors * 2
            scales_x = x[:, first_scale_feature:first_scale_feature + self.meta.n_scales]
            scales_x = torch.nn.functional.softplus(scales_x)

            # concat
            x = torch.cat([width_x, classes_x, *regs_x, scales_x], dim=1)

            # TODO: CoreMLv4 problem (see above).
            x = torch.transpose(x, 1, 2)

        return x
