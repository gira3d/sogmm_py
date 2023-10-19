import argparse
import numpy as np
from cprint import *
import time
import threading

from sogmm_py.utils import np_to_o3d_tensor
from isogmm_full_frame import ISOGMMFullFrame

import open3d as o3d
import open3d.core as o3c
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering


def set_enabled(widget, enable):
    widget.enabled = enable
    for child in widget.get_children():
        child.enabled = enable


class ISOGMMVisualizer:
    def __init__(self, isogmm):
        self.isogmm = isogmm

        self.window = gui.Application.instance.create_window(
            'GIRA3D - Reconstruction', 1280, 800)

        w = self.window
        em = w.theme.font_size
        spacing = int(np.round(0.25 * em))
        vspacing = int(np.round(0.5 * em))
        margins = gui.Margins(vspacing)

        # First panel - empty for now
        self.panel = gui.Vert(spacing, margins)

        # Application control
        b = gui.ToggleSwitch('Resume/Pause')
        b.set_on_clicked(self._on_switch)
        self.panel.add_child(b)

        # Scene widget
        self.widget3d = gui.SceneWidget()

        # Now add all the complex panels
        w.add_child(self.panel)
        w.add_child(self.widget3d)

        self.widget3d.scene = rendering.Open3DScene(self.window.renderer)
        self.widget3d.scene.set_background([1, 1, 1, 1])

        w.set_on_layout(self._on_layout)
        w.set_on_close(self._on_close)

        # helpful booleans
        self.is_started = False
        self.is_running = False
        self.is_complete = False

        threading.Thread(name='UpdateMain', target=self.update_main).start()

    # Toggle callback: application's main controller
    def _on_switch(self, is_on):
        if not self.is_started:
            gui.Application.instance.post_to_main_thread(
                self.window, self._on_start)
        self.is_running = not self.is_running

    def _on_start(self):
        self.is_started = True

    def _on_layout(self, ctx):
        em = ctx.theme.font_size

        panel_width = 20 * em
        rect = self.window.content_rect

        self.panel.frame = gui.Rect(rect.x, rect.y, panel_width, rect.height)

        x = self.panel.frame.get_right()
        self.widget3d.frame = gui.Rect(x, rect.y,
                                       rect.get_right() - x, rect.height)

    def _on_close(self):
        self.is_complete = True

        if self.is_started:
            # save stuff etc.
            cprint.ok('Finished.')

        return True

    def init_render(self):
        self.window.set_needs_layout()

        bbox = o3d.geometry.AxisAlignedBoundingBox([-5, -5, -5], [5, 5, 5])
        # self.widget3d.setup_camera(60, bbox, [0, 0, 0])
        self.widget3d.look_at([0, 0, 0], [0, 1, 3], [0, 0, -1])

    def update_main(self):
        gui.Application.instance.post_to_main_thread(
            self.window, lambda: self.init_render())

        isogmm.idx = 0
        while not self.is_complete:
            if not self.is_started or not self.is_running:
                time.sleep(0.05)
                continue

            self.model_updated = isogmm.process_frame(isogmm.idx)

            resampled = None
            if self.model_updated:
                frustum = o3d.geometry.LineSet.create_camera_visualization(
                    isogmm.W_d, isogmm.H_d, isogmm.K_d,
                    np.linalg.inv(isogmm.pose), 0.2)
                frustum.paint_uniform_color([0.961, 0.475, 0.000])

                resampled = np_to_o3d_tensor(
                    isogmm.sg.local_models[-1].sample(640 * 480, 2.2))
                mat = rendering.MaterialRecord()
                mat.shader = 'defaultUnlit'
                mat.sRGB_color = True
                self.widget3d.scene.scene.add_geometry('resampled_' + str(isogmm.idx),
                                                       resampled,
                                                       mat)
                mat = rendering.MaterialRecord()
                mat.shader = "unlitLine"
                mat.line_width = 5.0
                self.widget3d.scene.add_geometry(
                    "frustum_" + str(isogmm.idx), frustum, mat)

            isogmm.idx += 1
            if isogmm.idx >= isogmm.nframes:
                self.is_complete = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Incremental-SOGMM")
    parser.add_argument('--path_datasets', type=str)
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--nframes', type=int)
    parser.add_argument('--bandwidth', type=float)

    args = parser.parse_args()

    isogmm = ISOGMMFullFrame(args.path_datasets,
                             args.dataset_name,
                             nframes=args.nframes,
                             bandwidth=args.bandwidth)

    app = gui.Application.instance
    app.initialize()

    isogmm_vis = ISOGMMVisualizer(isogmm)

    app.run()
