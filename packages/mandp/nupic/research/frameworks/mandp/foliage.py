# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2020, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

import collections
import math

import numpy as np
import torch

Rect = collections.namedtuple("Rect", ["top", "left", "height", "width"])


class RandomImage:
    """
    Coordinates use matrix indexing [row][column], so objects are placed by
    specifying their "top" and "left".
    """
    def __init__(self, viewport_height=64, viewport_width=64):
        self.viewport_height = viewport_height
        self.viewport_width = viewport_width

        self.rects = []
        # Place rectangles randomly within the middle of the image.

        max_height = viewport_height / 2
        max_width = viewport_width / 2
        min_top = viewport_height / 4
        min_left = viewport_width / 4

        for _ in range(3):
            height = torch.rand(1).item() * max_height
            width = torch.rand(1).item() * max_width
            top = min_top + torch.rand(1).item() * (max_height - height)
            left = min_left + torch.rand(1).item() * (max_width - width)
            self.rects.append(Rect(top, left, height, width))

    def get_viewport_image(self, camera_top, camera_left):
        """
        Get a list of objects as they'll appear at a given camera position.
        Truncate the parts that don't appear in the camera's viewport.
        """
        viewport_top = camera_top - self.viewport_height / 2
        viewport_left = camera_left - self.viewport_width / 2

        rects = []
        for rect in self.rects:
            # Translate
            top = rect.top - viewport_top
            left = rect.left - viewport_left
            height = rect.height
            width = rect.width

            # Detect subset of rect that is in the viewport
            if top < 0:
                if top + height < 0:
                    continue

                truncated = 0 - top
                top = 0
                height -= truncated

            if top + height > self.viewport_height:
                if top >= self.viewport_height:
                    continue

                truncated = top + height - self.viewport_height
                height -= truncated

            if left < 0:
                if left + width < 0:
                    continue

                truncated = 0 - left
                left = 0
                width -= truncated

            if left + width > self.viewport_width:
                if left >= self.viewport_width:
                    continue

                truncated = left + width - self.viewport_width
                width -= truncated

            rects.append(Rect(top=top, left=left, height=height, width=width))
        return rects

    def render(self, camera_top, camera_left):
        """
        Render the camera's viewport into a set of discrete pixels.
        """
        rects = self.get_viewport_image(camera_top, camera_left)

        # It's significantly faster to create in numpy than convert to torch at the end.
        img = np.zeros((self.viewport_height, self.viewport_width), dtype=np.float)

        for rect in rects:
            outer_left_edge = math.floor(rect.left)
            outer_top_edge = math.floor(rect.top)
            right_edge = rect.left + rect.width
            bottom_edge = rect.top + rect.height

            outer_right_edge = math.floor(rect.left + rect.width)
            outer_bottom_edge = math.floor(rect.top + rect.height)
            outer_height = outer_bottom_edge - outer_top_edge
            outer_width = outer_right_edge - outer_left_edge

            left_fraction = 1 - (rect.left - outer_left_edge)
            top_fraction = 1 - (rect.top - outer_top_edge)
            bottom_fraction = bottom_edge - outer_bottom_edge
            right_fraction = right_edge - outer_right_edge

            # Detect cases where an edge extends exactly to an integer boundary.
            # (Avoid drawing a zero edge, especially since it is likely 1 pixel
            # outside the viewport)
            if bottom_edge == outer_bottom_edge:
                outer_bottom_edge -= 1
                bottom_fraction = 1.0
            if right_edge == outer_right_edge:
                outer_right_edge -= 1
                right_fraction = 1.0

            # Color the top left pixel
            img[outer_top_edge, outer_left_edge] += left_fraction * top_fraction

            if outer_height > 0:
                # color the left side
                img[outer_top_edge + 1:outer_bottom_edge,
                    outer_left_edge] += left_fraction

                # color the bottom left pixel
                img[outer_bottom_edge,
                    outer_left_edge] += left_fraction * bottom_fraction

            if outer_width > 0:
                # color the top side
                img[outer_top_edge,
                    outer_left_edge + 1:outer_right_edge] += top_fraction

                # color the top right pixel
                img[outer_top_edge,
                    outer_right_edge] += top_fraction * right_fraction

            if outer_height > 0 and outer_width > 0:
                # color the right side
                img[outer_top_edge + 1:outer_bottom_edge,
                    outer_right_edge] += right_fraction

                # color the bottom side
                img[outer_bottom_edge,
                    outer_left_edge + 1:outer_right_edge] += bottom_fraction

                # color the bottom right pixel
                img[outer_bottom_edge,
                    outer_right_edge] += bottom_fraction * right_fraction

                if outer_height > 1 and outer_width > 1:
                    # Color the interior
                    img[outer_top_edge + 1:outer_bottom_edge,
                        outer_left_edge + 1:outer_right_edge] += 1.0

        return torch.Tensor(img)

    def render_random_straight_path(self, num_steps):
        # choose random starting point
        habitable_height = self.viewport_height / 2
        habitable_width = self.viewport_width / 2

        while True:
            start_habitable_top = torch.rand(1).item() * habitable_height
            start_habitable_left = torch.rand(1).item() * habitable_width

            # choose a random direction
            found_valid_direction = False
            retries_remaining = 100
            while True:
                direction = torch.rand(1).item() * 2 * np.pi

                # Make sure each step will always change pixels.
                if abs(math.cos(direction)) >= 0.5:
                    # Direction is more horizontal than vertical
                    # Step far enough to move 1 pixel up.
                    min_distance_per_step = 1 / abs(math.cos(direction))
                else:
                    # Direction is more vertical than horizontal.
                    # Step far enough to move 1 pixel sideways.
                    min_distance_per_step = 1 / abs(math.sin(direction))

                min_distance = num_steps * min_distance_per_step

                if direction == 0 or direction == math.pi:
                    vertical_intersection_distance = np.inf
                elif 0 < direction < math.pi:
                    # Check intersection point with top boundary
                    vertical_intersection_distance = (
                        start_habitable_top / abs(math.sin(direction))
                    )
                else:
                    # Check intersection point with bottom boundary
                    vertical_intersection_distance = (
                        (habitable_height - start_habitable_top)
                        / abs(math.sin(direction))
                    )

                if direction == math.pi / 2 or direction == 3 * math.pi / 2:
                    horizontal_intersection_distance = np.inf
                elif math.pi / 2 < direction < 3 * math.pi / 2:
                    horizontal_intersection_distance = (
                        start_habitable_left / abs(math.cos(direction))
                    )
                else:
                    horizontal_intersection_distance = (
                        (habitable_width - start_habitable_left)
                        / abs(math.cos(direction))
                    )

                max_distance = min(vertical_intersection_distance,
                                   horizontal_intersection_distance)
                if max_distance <= min_distance:
                    if retries_remaining > 0:
                        retries_remaining -= 1
                        continue
                    else:
                        break

                distance = (
                    min_distance
                    + torch.rand(1).item() * (max_distance - min_distance)
                )

                found_valid_direction = True
                break

            if found_valid_direction:
                break

        start_top = start_habitable_top + (habitable_height / 2)
        start_left = start_habitable_left + (habitable_width / 2)
        end_top = start_top + distance * -math.sin(direction)
        end_left = start_left + distance * math.cos(direction)

        imgs = torch.stack([
            self.render(camera_top, camera_left)
            for camera_top, camera_left
            in zip(np.linspace(start_top, end_top, num=num_steps,
                               endpoint=True),
                   np.linspace(start_left, end_left, num=num_steps,
                               endpoint=True))])

        return imgs.view(num_steps, -1)


class FoliageDataset(torch.utils.data.IterableDataset):
    def __iter__(self):
        while True:
            ri = RandomImage()
            yield ri.render_random_straight_path(10)
