"""Map parser for roborock.vacuum.e2.

The e2 does not use the map format of the other Roborock vacuums. Its map file
(all integers little endian):

======  ======  =================================================================
offset  length  content
======  ======  =================================================================
0x00    4       magic ``RRSA``
0x04    2       major version
0x06    2       CRC-16/IBM-3740 (CCITT-FALSE) of everything from offset 0x08
0x08    4       map index
0x0C    4       map sequence
0x10    2       length
0x12            blocks: type (2), size (2, includes these 4 bytes), payload
======  ======  =================================================================

Block types:

1. charger position: x (4), y (4); 0x7FFFFFFF if unknown
2. robot position: x (4), y (4)
3. map image, see below
4. robot angle: signed degrees (2)

The map image payload holds a chunk count at offset 2 (at most 50), the compressed
length of every chunk as one byte each from offset 3, and the chunk data from
offset 55. The chunks are raw LZ4 blocks that decompress back to back into one
buffer of 101 x 101 bytes. Each byte packs a square of 2 x 2 cells with 2 bits per
cell (bits 0-1 lower left, 2-3 upper left, 4-5 lower right, 6-7 upper right), so
the map is a grid of 202 x 202 cells, stored bottom row first. Cell values are
0 (outside), 2 (wall) and 1 or 3 (inside). One cell is 150 units of the
coordinates used for the charger and the robot.

The image shows the raw cells. The vendor app post-processes them in a way
that is not understood, so its map can look slightly different.
"""

import gzip
import logging
from enum import IntEnum
from typing import Any

from PIL import Image
from PIL.Image import Image as ImageType
from PIL.Image import Resampling
from vacuum_map_parser_base.config.color import Color, SupportedColor
from vacuum_map_parser_base.map_data import ImageData, MapData, Point
from vacuum_map_parser_base.map_data_parser import MapDataParser

from .lz4 import decompress_block

_LOGGER = logging.getLogger(__name__)


class RoborockE2BlockType(IntEnum):
    """roborock.vacuum.e2 map block type."""

    CHARGER = 1
    ROBOT_POSITION = 2
    IMAGE = 3
    ROBOT_ANGLE = 4


class RoborockE2MapDataParser(MapDataParser):
    """Map parser for roborock.vacuum.e2."""

    MAGIC = b"RRSA"
    HEADER_LENGTH = 0x12
    PACKED_SIZE = 101
    GRID_SIZE = 2 * PACKED_SIZE
    CELL_SIZE = 150
    MAX_CHUNKS = 50
    CHUNKS_OFFSET = 3 + MAX_CHUNKS + 2
    UNKNOWN_POSITION = 0x7FFFFFFF

    CELL_OUTSIDE = 0
    CELL_WALL = 2

    def unpack_map(self, raw_encoded: bytes, *args: Any, **kwargs: Any) -> bytes:
        if raw_encoded[:2] == b"\x1f\x8b":
            return gzip.decompress(raw_encoded)
        return raw_encoded

    def parse(self, raw: bytes, *args: Any, **kwargs: Any) -> MapData:
        if raw[:4] != self.MAGIC:
            raise ValueError("Not a roborock.vacuum.e2 map: RRSA header missing")

        map_data = MapData(self.GRID_SIZE * self.CELL_SIZE / 2, 1000)
        map_data.additional_parameters["major_version"] = self._get_int(raw, 0x04, 2)
        map_data.additional_parameters["map_index"] = self._get_int(raw, 0x08, 4)
        map_data.additional_parameters["map_sequence"] = self._get_int(raw, 0x0C, 4)
        map_data.additional_parameters["is_valid"] = self._get_int(raw, 0x06, 2) == self._crc16(raw[0x08:])

        robot: tuple[int, int] | None = None
        angle: int | None = None
        cells: bytearray | None = None
        for block_type, payload in self._iterate_blocks(raw):
            match block_type:
                case RoborockE2BlockType.CHARGER:
                    charger = self._parse_position(payload)
                    if charger is not None:
                        map_data.charger = Point(*charger)
                case RoborockE2BlockType.ROBOT_POSITION:
                    robot = self._parse_position(payload)
                case RoborockE2BlockType.IMAGE:
                    cells = self._parse_cells(payload)
                case RoborockE2BlockType.ROBOT_ANGLE:
                    angle = self._get_int(payload, 0, 2, signed=True)
                case _:
                    _LOGGER.debug("UNKNOWN BLOCK TYPE: %d, length %d", block_type, len(payload))

        if robot is not None:
            map_data.vacuum_position = Point(robot[0], robot[1], angle)

        map_data.image = self._create_image_data(cells)
        if not map_data.image.is_empty:
            self._image_generator.draw_map(map_data)
        return map_data

    def _iterate_blocks(self, raw: bytes) -> list[tuple[int, bytes]]:
        blocks = []
        pos = self.HEADER_LENGTH
        while pos + 4 <= len(raw):
            block_type = self._get_int(raw, pos, 2)
            block_size = self._get_int(raw, pos + 2, 2)
            if block_size < 4:
                _LOGGER.warning("Invalid block size %d at offset %d", block_size, pos)
                break
            blocks.append((block_type, raw[pos + 4 : pos + block_size]))
            pos += block_size
        return blocks

    def _parse_position(self, payload: bytes) -> tuple[int, int] | None:
        x = self._get_int(payload, 0, 4)
        y = self._get_int(payload, 4, 4)
        if self.UNKNOWN_POSITION in (x, y):
            return None
        return x, y

    def _parse_cells(self, payload: bytes) -> bytearray:
        """Decompress the map image and unpack it to one byte per cell, bottom row first."""
        chunk_count = payload[2]
        if chunk_count > self.MAX_CHUNKS:
            raise ValueError(f"Too many map chunks: {chunk_count}")
        packed = bytearray()
        pos = self.CHUNKS_OFFSET
        for chunk_length in payload[3 : 3 + chunk_count]:
            decompress_block(payload[pos : pos + chunk_length], packed)
            pos += chunk_length

        packed_length = self.PACKED_SIZE * self.PACKED_SIZE
        if len(packed) != packed_length:
            _LOGGER.warning("Unexpected map size %d, expected %d", len(packed), packed_length)
            packed = packed[:packed_length].ljust(packed_length, b"\x00")

        size = self.GRID_SIZE
        cells = bytearray(size * size)
        for index, value in enumerate(packed):
            if value == 0:
                continue
            row, col = divmod(index, self.PACKED_SIZE)
            cell = 2 * row * size + 2 * col
            cells[cell] = value & 0x3
            cells[cell + size] = (value >> 2) & 0x3
            cells[cell + 1] = (value >> 4) & 0x3
            cells[cell + size + 1] = (value >> 6) & 0x3
        return cells

    def _create_image_data(self, cells: bytearray | None) -> ImageData:
        bounds = self._get_bounds(cells) if cells is not None else None
        if cells is None or bounds is None:
            empty = self._image_generator.create_empty_map_image()
            return ImageData(0, 0, 0, 0, 0, self._image_config, empty, self._map_to_image)

        left, top, width, height = bounds
        image = self._create_image(cells, left, top, width, height)
        return ImageData(
            self.PACKED_SIZE * self.PACKED_SIZE,
            top,
            left,
            height,
            width,
            self._image_config,
            image,
            self._map_to_image,
        )

    def _get_bounds(self, cells: bytearray) -> tuple[int, int, int, int] | None:
        """Bounding box of the known cells as (left, top, width, height); top is the lowest row."""
        size = self.GRID_SIZE
        rows = [row for row in range(size) if any(cells[row * size : (row + 1) * size])]
        if not rows:
            return None
        cols = [col for col in range(size) if any(cells[row * size + col] for row in rows)]
        return cols[0], rows[0], cols[-1] - cols[0] + 1, rows[-1] - rows[0] + 1

    def _create_image(self, cells: bytearray, left: int, top: int, width: int, height: int) -> ImageType:
        trim = self._image_config.trim
        trim_left = int(trim.left * width / 100)
        trim_right = int(trim.right * width / 100)
        trim_top = int(trim.top * height / 100)
        trim_bottom = int(trim.bottom * height / 100)
        trimmed_width = width - trim_left - trim_right
        trimmed_height = height - trim_top - trim_bottom

        colors = self._palette.cached_colors
        outside = self._rgba(colors[SupportedColor.MAP_OUTSIDE])
        wall = self._rgba(colors[SupportedColor.MAP_WALL])
        inside = self._rgba(colors[SupportedColor.MAP_INSIDE])

        pixels = bytearray()
        # image rows run top-down, grid rows bottom-up
        for img_y in range(trimmed_height):
            row = top + trim_bottom + trimmed_height - 1 - img_y
            start = row * self.GRID_SIZE + left + trim_left
            for value in cells[start : start + trimmed_width]:
                if value == self.CELL_OUTSIDE:
                    pixels += outside
                elif value == self.CELL_WALL:
                    pixels += wall
                else:
                    pixels += inside

        image = Image.frombytes("RGBA", (trimmed_width, trimmed_height), bytes(pixels))
        scale = self._image_config.scale
        if scale != 1:
            image = image.resize((int(trimmed_width * scale), int(trimmed_height * scale)), resample=Resampling.NEAREST)
        return image

    @staticmethod
    def _rgba(color: Color) -> bytes:
        return bytes(color) if len(color) == 4 else bytes(color) + b"\xff"

    @staticmethod
    def _map_to_image(p: Point) -> Point:
        return Point(p.x / RoborockE2MapDataParser.CELL_SIZE, p.y / RoborockE2MapDataParser.CELL_SIZE)

    @staticmethod
    def _get_int(data: bytes, address: int, length: int, signed: bool = False) -> int:
        return int.from_bytes(data[address : address + length], "little", signed=signed)

    @staticmethod
    def _crc16(data: bytes) -> int:
        """CRC-16/IBM-3740 (CCITT-FALSE): polynomial 0x1021, initial value 0xFFFF."""
        crc = 0xFFFF
        for value in data:
            crc ^= value << 8
            for _ in range(8):
                crc = ((crc << 1) ^ 0x1021) if crc & 0x8000 else (crc << 1)
                crc &= 0xFFFF
        return crc
