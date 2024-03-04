"""Roborock map image parser."""

import logging

from PIL import Image
from PIL.Image import Image as ImageType
from PIL.Image import Resampling
from vacuum_map_parser_base.config.color import ColorsPalette, SupportedColor
from vacuum_map_parser_base.config.image_config import ImageConfig

_LOGGER = logging.getLogger(__name__)


class RoborockImageParser:
    """Roborock image parser."""

    MAP_OUTSIDE = 0x00
    MAP_WALL = 0x01
    MAP_INSIDE = 0xFF
    MAP_SCAN = 0x07

    def __init__(self, palette: ColorsPalette, image_config: ImageConfig):
        self._colors_palette = palette
        self._image_config = image_config

    def parse(
        self, raw_data: bytes, width: int, height: int, carpet_map: set[int] | None
    ) -> tuple[ImageType | None, dict[int, tuple[int, int, int, int]]]:
        rooms = {}
        scale = self._image_config.scale
        cached_colors = self._colors_palette.cached_colors
        cached_room_colors = self._colors_palette.cached_room_colors
        trim_left = int(self._image_config.trim.left * width / 100)
        trim_right = int(self._image_config.trim.right * width / 100)
        trim_top = int(self._image_config.trim.top * height / 100)
        trim_bottom = int(self._image_config.trim.bottom * height / 100)
        trimmed_height = height - trim_top - trim_bottom
        trimmed_width = width - trim_left - trim_right
        trimmed_left_width = trim_left + width
        image = Image.new("RGBA", (trimmed_width, trimmed_height))
        if width == 0 or height == 0:
            return None, {}
        pixels = image.load()
        for img_y in range(trimmed_height):
            img_x_offset = trimmed_left_width * (img_y + trim_bottom)
            for img_x in range(trimmed_width):
                idx = img_x + img_x_offset
                pixel_type = raw_data[idx]
                x = img_x
                y = trimmed_height - img_y - 1
                if carpet_map is not None and idx in carpet_map and (x + y) % 2:
                    pixels[x, y] = cached_colors[SupportedColor.CARPETS]
                elif pixel_type == RoborockImageParser.MAP_OUTSIDE:
                    pixels[x, y] = cached_colors[SupportedColor.MAP_OUTSIDE]
                elif pixel_type == RoborockImageParser.MAP_WALL:
                    pixels[x, y] = cached_colors[SupportedColor.MAP_WALL]
                elif pixel_type == RoborockImageParser.MAP_INSIDE:
                    pixels[x, y] = cached_colors[SupportedColor.MAP_INSIDE]
                elif pixel_type == RoborockImageParser.MAP_SCAN:
                    pixels[x, y] = cached_colors[SupportedColor.SCAN]
                else:
                    obstacle = pixel_type & 0x07
                    if obstacle == 0:
                        pixels[x, y] = cached_colors[SupportedColor.GREY_WALL]
                    elif obstacle == 1:
                        pixels[x, y] = cached_colors[SupportedColor.MAP_WALL_V2]
                    elif obstacle == 7:
                        room_number = RoborockImageParser._get_room_number(pixel_type)
                        room_x = img_x + trim_left
                        room_y = img_y + trim_bottom
                        if room_number not in rooms:
                            rooms[room_number] = (room_x, room_y, room_x, room_y)
                        else:
                            rooms[room_number] = (
                                min(rooms[room_number][0], room_x),
                                min(rooms[room_number][1], room_y),
                                max(rooms[room_number][2], room_x),
                                max(rooms[room_number][3], room_y),
                            )
                        try:
                            pixels[x, y] = cached_room_colors[room_number]
                        except KeyError:
                            # Since rooms can go above the 16 we preprocess, we handle the key error here and add it to
                            # our local version of the cache and the real cache.
                            cached_room_colors[room_number] = self._colors_palette.get_room_color(room_number)
                            pixels[x, y] = cached_room_colors[room_number]
                    else:
                        pixels[x, y] = cached_colors[SupportedColor.UNKNOWN]
        if scale != 1 and width != 0 and height != 0:
            image = image.resize((int(trimmed_width * scale), int(trimmed_height * scale)), resample=Resampling.NEAREST)
        return image, rooms

    @staticmethod
    def get_room_at_pixel(raw_data: bytes, width: int, x: int, y: int) -> int | None:
        room_number = None
        pixel_type = raw_data[x + width * y]
        if pixel_type not in [RoborockImageParser.MAP_INSIDE, RoborockImageParser.MAP_SCAN]:
            if pixel_type & 0x07 == 7:
                room_number = RoborockImageParser._get_room_number(pixel_type)
        return room_number

    @staticmethod
    def _get_room_number(pixel_type: int) -> int:
        return (pixel_type & 0xFF) >> 3
