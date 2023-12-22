# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


"""
The matplotlib plotter implementation for all the primitive tasks (in our case: lines and
dots)
"""
from typing import Any, Callable, Dict, List

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3

import numpy as np


class SimplePlotter():


    def __init__(self, skeleton_state) -> None:


        
        self._skeleton_state = skeleton_state
        self._lines, self._dots = self.get_lines_dots()

        self._fig = plt.figure()
        self._ax = self._fig.add_subplot(111, projection='3d')


        self._plot_lines()
        self._plot_dots()

        # Set equal aspect ratio for both axes
        # self._ax.set_aspect('equal')
        # Automatically adjust axis limits to fit the data
     
        self._set_aspect_equal_3d()
        plt.legend()
        plt.show()

    def get_lines_dots(self):
        """Get all the lines and dots needed to draw the skeleton state
        """
        assert (
            len(self._skeleton_state.tensor.shape) == 1
        ), "the state has to be zero dimensional"

        dots = self._skeleton_state.global_translation.numpy()
        skeleton_tree = self._skeleton_state.skeleton_tree
        parent_indices = skeleton_tree.parent_indices.numpy()
        lines = []
        labels = []
        for node_index in range(len(skeleton_tree)):
            parent_index = parent_indices[node_index]
            labels.append(skeleton_tree.node_names[node_index])
            if parent_index != -1:
                lines.append([dots[node_index], dots[parent_index]])
        lines = np.array(lines)
        self.labels = labels
        return lines, dots
    
    def _plot_lines(self):
        for i in range(len(self._lines)):
           
            self._ax.plot(
                *self._lines_extract_xyz_impl(i, self._lines),
                color='red',
                linewidth=2,
                alpha=1
            )
        return
    
    def _plot_dots(self):
        self._ax.plot(
            self._dots[:, 0],
            self._dots[:, 1],
            self._dots[:, 2],
            c='blue',
            linestyle="",
            marker=".",
            markersize=4,
            alpha=1,
        )
        # Label each dot with text
        for i, (xi, yi, zi) in enumerate(zip(self._dots[:, 0], self._dots[:, 1], self._dots[:, 2])):
            self._ax.text(xi, yi, zi, self.labels[i], fontsize=7, color='r')
        return
    

    def _lines_extract_xyz_impl(self, index, lines_task):
        return lines_task[index, :, 0], lines_task[index, :, 1], lines_task[index, :, 2]
    

    def _set_aspect_equal_3d(self):
        xlim = self._ax.get_xlim3d()
        ylim = self._ax.get_ylim3d()
        zlim = self._ax.get_zlim3d()

        xmean = np.mean(xlim)
        ymean = np.mean(ylim)
        zmean = np.mean(zlim)

        plot_radius = max(
            [
                abs(lim - mean_)
                for lims, mean_ in ((xlim, xmean), (ylim, ymean), (zlim, zmean))
                for lim in lims
            ]
        )

        self._ax.set_xlabel('X')
        self._ax.set_ylabel('Y')
        self._ax.set_zlabel('Z')

        self._ax.set_xlim3d([xmean - plot_radius, xmean + plot_radius])
        self._ax.set_ylim3d([ymean - plot_radius, ymean + plot_radius])
        self._ax.set_zlim3d([zmean - plot_radius, zmean + plot_radius])


