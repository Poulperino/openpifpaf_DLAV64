import numpy as np
from openpifpaf.visualizer import Base
try:
    import matplotlib
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
except ImportError:
    plt = None
    make_axes_locatable = None

from openpifpaf import show
from .. import headmeta


class Visualizer(Base):
    def __init__(self, meta: headmeta.Butterfly):
        super().__init__(meta.name)
        self.meta = meta
        # self.butterfly_full = False
        # self.fields_indices = self.process_indices(fields_indices)
        # if self.fields_indices and self.fields_indices[0][0] == -1:
        #     self.butterfly_full = True
        self.file_prefix = None
        # self.show_seed_confidence = show_seed_confidence
        # self.show = show
        # self.image = None
        # self.processed_image = None

    # @staticmethod
    # def process_indices(indices):
    #     return [[int(e) for e in i.split(',')] for i in indices]

    # def set_image(self, image, processed_image):
    #     if image is None and processed_image is not None:
    #         # image not given, so recover an image of the correct shape
    #         image = np.moveaxis(np.asarray(processed_image), 0, -1)
    #         image = np.clip((image + 2.0) / 4.0, 0.0, 1.0)
    #
    #     self.image = image
    #     self.processed_image = processed_image

    def resized_image(self, io_scale):
        resized_image = np.moveaxis(
            np.asarray(self.processed_image)[:, ::int(io_scale), ::int(io_scale)], 0, -1)
        return np.clip((resized_image + 2.0) / 4.0, 0.0, 1.0)

    def seeds_butterfly(self, seeds, io_scale):
        print('seeds')
        # if self.butterfly_full:
        #     field_indices = {f[1] for f in seeds}
        # field_indices = {f[0] for f in self.fields_indices}#{f[1] for f in seeds}
        field_indices = self.indices('confidence')
        if len(field_indices)==0:
            return
        if len(field_indices)==1 and field_indices[0] == -1:
            field_indices = np.arange(len(self.meta.categories))

        fig_file = self.file_prefix + '.butterfly.seeds.png' if self.file_prefix else None
        with self.image_canvas(self.processed_image(), margin=[0.0, 0.01, 0.05, 0.01]) as ax:
            cmap = matplotlib.cm.get_cmap('viridis_r')
            cnorm = matplotlib.colors.Normalize(vmin=0, vmax=1)
            show.white_screen(ax, alpha=0.5)
            for f in field_indices:
                x = [seed[2] for seed in seeds if seed[1] == f]
                y = [seed[3] for seed in seeds if seed[1] == f]
                w = [seed[4] for seed in seeds if seed[1] == f]
                h = [seed[5] for seed in seeds if seed[1] == f]
                c = [seed[0] for seed in seeds if seed[1] == f]
                ax.plot(x, y, 'o')
                for xx, yy, cc, ww, hh in zip(x, y, c, w, h):
                    color = cmap(cnorm(cc))
                    rectangle = matplotlib.patches.Rectangle(
                        (xx - ww/2, yy - hh/2), ww, hh,
                        color=color, zorder=1, alpha=0.5, linewidth=1, fill=True)

                    ax.add_artist(rectangle)
                    ax.text(xx, yy, '{:.2f}'.format(cc))

    #@staticmethod
    def occupied(self, occ):
        field_indices = self.indices('confidence')
        if len(field_indices)==0:
            return
        occ = occ.copy()
        occ[occ > 0] = 1.0
        with show.canvas() as ax:
            ax.imshow(occ)

    def butterfly_raw(self, butterfly, io_scale):
        print('raw butterfly')
        intensity_fields, reg_fields, reg_fields_b, width_fields, height_fields = butterfly  # pylint: disable=unused-variable
        # if self.butterfly_full:
        #     self.fields_indices = [[i] for i in np.arange(intensity_fields.shape[0])[np.max(intensity_fields, axis=(1,2))>0.1]]
        field_indices = self.indices('confidence')

        if len(self.indices('confidence'))==1 and self.indices('confidence')[0] == -1:
            field_indices = np.arange(intensity_fields.shape[0])[np.nanmax(intensity_fields, axis=(1,2))>0.2]

        for f in field_indices:
            print('butterfly confidence field - index', f)
            fig_file = self.file_prefix + '.butterfly{}.c.png'.format(f) if self.file_prefix else None
            with self.image_canvas(self.processed_image(), margin=[0.0, 0.01, 0.05, 0.01]) as ax:
                # ax.imshow(self.resized_image(io_scale))
                im = ax.imshow(self.scale_scalar(intensity_fields[f], self.meta.stride), alpha=0.9,
                               vmin=0.0, vmax=1.0, cmap='YlOrRd')

                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size='3%', pad=0.05)
                plt.colorbar(im, cax=cax)

                # ax.get_xaxis().set_visible(False)
                # ax.get_yaxis().set_visible(False)

        for f in field_indices:
            print('butterfly vector field - index', f)
            fig_file = self.file_prefix + '.butterfly{}.v.png'.format(f) if self.file_prefix else None
            with self.image_canvas(self.processed_image(), margin=[0.0, 0.01, 0.05, 0.01]) as ax:
                # ax.imshow(self.image)
                show.white_screen(ax, alpha=0.5)
                show.quiver(ax, reg_fields[f],
                            confidence_field=intensity_fields[f],
                            reg_uncertainty=reg_fields_b[f], uv_is_offset=True,
                            cmap='viridis_r', clim=(0.5, 1.0),
                            threshold=0.1, xy_scale=self.meta.stride)

                # ax.get_xaxis().set_visible(False)
                # ax.get_yaxis().set_visible(False)

        for f in field_indices:
            print('butterfly wh field - index', f)
            fig_file = self.file_prefix + '.butterfly{}.w.png'.format(f) if self.file_prefix else None
            # with show.canvas(fig_file, show=self.show,) as ax:
            with self.image_canvas(self.processed_image(), margin=[0.0, 0.01, 0.05, 0.01]) as ax:
                # ax.imshow(self.image)
                show.white_screen(ax, alpha=0.5)
                show.boxes_wh(ax, np.exp(width_fields[f]), np.exp(height_fields[f]), regression_field=reg_fields[f],
                            confidence_field=intensity_fields[f],
                            cmap='Greens', linewidth=2, regression_field_is_offset=True,
                            xy_scale=self.meta.stride, fill=False)

                # ax.get_xaxis().set_visible(False)
                # ax.get_yaxis().set_visible(False)

    def butterfly_hr(self, butterflyhr):
        print('butterflyhr')
        # if self.butterfly_full:
        #     self.fields_indices = [[i] for i in np.arange(butterflyhr.shape[0])[np.max(butterflyhr, axis=(1,2))>0.1]]
        field_indices = self.indices('confidence')

        if len(self.indices('confidence'))==1 and self.indices('confidence')[0] == -1:
            field_indices = np.arange(butterflyhr.shape[0])[np.nanmax(butterflyhr, axis=(1,2))>0.2]

        for f in field_indices:
            fig_file = (
                self.file_prefix + '.butterfly{}.hr.png'.format(f)
                if self.file_prefix else None
            )
            # with show.canvas(fig_file, figsize=(8, 5), show=self.show) as ax:
            with self.image_canvas(self.processed_image(), margin=[0.0, 0.01, 0.05, 0.01]) as ax:
                # ax.imshow(self.image)
                o = ax.imshow(butterflyhr[f], alpha=0.9, vmin=0.0, vmax=1.0, cmap='Oranges')

                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size='3%', pad=0.05)
                plt.colorbar(o, cax=cax)

                # ax.get_xaxis().set_visible(False)
                # ax.get_yaxis().set_visible(False)
                # ax.set_xlim(0, self.image.shape[1])
                # ax.set_ylim(self.image.shape[0], 0)
    def butterflyhr_wh(self, butterflyhr_width, butterflyhr_height, butterflyhr, threshold=0.2):
        print('butterflyhr wh')
        # if self.butterfly_full:
        #     self.fields_indices = [[i] for i in np.arange(butterflyhr.shape[0])[np.max(butterflyhr, axis=(1,2))>0.1]]
        field_indices = self.indices('confidence')

        if len(self.indices('confidence'))==1 and self.indices('confidence')[0] == -1:
            field_indices = np.arange(butterflyhr.shape[0])[np.nanmax(butterflyhr, axis=(1,2))>0.2]

        for f in field_indices:
            fig_file = (
                self.file_prefix + '.butterfly{}.hr_wh.png'.format(f)
                if self.file_prefix else None
            )
            # with show.canvas(fig_file, show=self.show) as ax:
            with self.image_canvas(self.processed_image(), margin=[0.0, 0.01, 0.05, 0.01]) as ax:
                # ax.imshow(self.image)
                show.white_screen(ax, alpha=0.5)
                show.boxes_wh(ax, butterflyhr_width[f], butterflyhr_height[f],
                            confidence_field=butterflyhr[f],
                            xy_scale=1, fill=False, threshold=threshold)

                # ax.get_xaxis().set_visible(False)
                # ax.get_yaxis().set_visible(False)
