"""Roborock map parser."""

import gzip
import logging
from enum import Enum
from typing import Any

from vacuum_map_parser_base.config.color import ColorsPalette
from vacuum_map_parser_base.config.drawable import Drawable
from vacuum_map_parser_base.config.image_config import ImageConfig
from vacuum_map_parser_base.config.size import Sizes
from vacuum_map_parser_base.config.text import Text
from vacuum_map_parser_base.map_data import (
    Area,
    ImageData,
    MapData,
    Obstacle,
    ObstacleDetails,
    Path,
    Point,
    Room,
    Wall,
    Zone,
)
from vacuum_map_parser_base.map_data_parser import MapDataParser

from .image_parser import RoborockImageParser

_LOGGER = logging.getLogger(__name__)


class RoborockBlockType(Enum):
    """Roborock map block type."""

    CHARGER = 1
    IMAGE = 2
    PATH = 3
    GOTO_PATH = 4
    GOTO_PREDICTED_PATH = 5
    CURRENTLY_CLEANED_ZONES = 6
    GOTO_TARGET = 7
    ROBOT_POSITION = 8
    NO_GO_AREAS = 9
    VIRTUAL_WALLS = 10
    BLOCKS = 11
    NO_MOPPING_AREAS = 12
    OBSTACLES = 13
    IGNORED_OBSTACLES = 14
    OBSTACLES_WITH_PHOTO = 15
    IGNORED_OBSTACLES_WITH_PHOTO = 16
    CARPET_MAP = 17
    MOP_PATH = 18
    NO_CARPET_AREAS = 19
    DIGEST = 1024


class RoborockMapDataParser(MapDataParser):
    """Roborock map parser."""

    KNOWN_OBSTACLE_TYPES = {
        0: "cable",
        1: "pet waste",
        2: "shoes",
        3: "poop",
        4: "pedestal",
        5: "extension cord",
        9: "weighting scale",
        10: "clothes",
        25: "dustpan",
        26: "furniture with a crossbar",
        27: "furniture with a crossbar",
        34: "clothes",
        48: "cable",
        49: "pet",
        50: "pet",
        51: "fabric/paper balls",
    }

    def __init__(
        self,
        palette: ColorsPalette,
        sizes: Sizes,
        drawables: list[Drawable],
        image_config: ImageConfig,
        texts: list[Text],
    ):
        super().__init__(palette, sizes, drawables, image_config, texts)
        self._image_parser = RoborockImageParser(palette, image_config)

    def unpack_map(self, raw_encoded: bytes, *args: Any, **kwargs: Any) -> bytes:
        return gzip.decompress(raw_encoded)

    def parse(self, raw: bytes, *args: Any, **kwargs: Any) -> MapData:
        map_data = MapData(25500, 1000)
        map_header_length = RoborockMapDataParser._get_int16(raw, 0x02)
        map_data.additional_parameters["major_version"] = RoborockMapDataParser._get_int16(raw, 0x08)
        map_data.additional_parameters["minor_version"] = RoborockMapDataParser._get_int16(raw, 0x0A)
        map_data.additional_parameters["map_index"] = RoborockMapDataParser._get_int32(raw, 0x0C)
        map_data.additional_parameters["map_sequence"] = RoborockMapDataParser._get_int32(raw, 0x10)
        block_start_position = map_header_length
        img_start: int | None = None
        img_data = None
        img_data_length = None
        img_header_length = None
        img_header = None

        while block_start_position < len(raw):
            block_header_length = RoborockMapDataParser._get_int16(raw, block_start_position + 0x02)
            header = RoborockMapDataParser._get_bytes(raw, block_start_position, block_header_length)
            block_type = RoborockMapDataParser._get_int16(header, 0x00)
            block_data_length = RoborockMapDataParser._get_int32(header, 0x04)
            block_data_start = block_start_position + block_header_length
            data = RoborockMapDataParser._get_bytes(raw, block_data_start, block_data_length)

            match block_type:
                case RoborockBlockType.CHARGER.value:
                    map_data.charger = RoborockMapDataParser._parse_object_position(block_data_length, data)
                case RoborockBlockType.IMAGE.value:
                    img_start = block_start_position
                    img_data_length = block_data_length
                    img_header_length = block_header_length
                    img_data = data
                    img_header = header
                case RoborockBlockType.ROBOT_POSITION.value:
                    map_data.vacuum_position = RoborockMapDataParser._parse_object_position(block_data_length, data)
                case RoborockBlockType.PATH.value:
                    map_data.path = RoborockMapDataParser._parse_path(block_start_position, header, raw)
                case RoborockBlockType.GOTO_PATH.value:
                    map_data.goto_path = RoborockMapDataParser._parse_path(block_start_position, header, raw)
                case RoborockBlockType.GOTO_PREDICTED_PATH.value:
                    map_data.predicted_path = RoborockMapDataParser._parse_path(block_start_position, header, raw)
                case RoborockBlockType.CURRENTLY_CLEANED_ZONES.value:
                    map_data.zones = RoborockMapDataParser._parse_zones(data, header)
                case RoborockBlockType.GOTO_TARGET.value:
                    map_data.goto = RoborockMapDataParser._parse_goto_target(data)
                case RoborockBlockType.DIGEST.value:
                    map_data.additional_parameters["is_valid"] = True
                case RoborockBlockType.VIRTUAL_WALLS.value:
                    map_data.walls = RoborockMapDataParser._parse_walls(data, header)
                case RoborockBlockType.NO_GO_AREAS.value:
                    map_data.no_go_areas = RoborockMapDataParser._parse_area(header, data)
                case RoborockBlockType.NO_MOPPING_AREAS.value:
                    map_data.no_mopping_areas = RoborockMapDataParser._parse_area(header, data)
                case RoborockBlockType.OBSTACLES.value:
                    map_data.obstacles = RoborockMapDataParser._parse_obstacles(data, header)
                case RoborockBlockType.IGNORED_OBSTACLES.value:
                    map_data.ignored_obstacles = RoborockMapDataParser._parse_obstacles(data, header)
                case RoborockBlockType.OBSTACLES_WITH_PHOTO.value:
                    map_data.obstacles_with_photo = RoborockMapDataParser._parse_obstacles(data, header)
                case RoborockBlockType.IGNORED_OBSTACLES_WITH_PHOTO.value:
                    map_data.ignored_obstacles_with_photo = RoborockMapDataParser._parse_obstacles(data, header)
                case RoborockBlockType.BLOCKS.value:
                    block_pairs = RoborockMapDataParser._get_int16(header, 0x08)
                    map_data.blocks = RoborockMapDataParser._get_bytes(data, 0, block_pairs)
                case RoborockBlockType.MOP_PATH.value:
                    points_mask = RoborockMapDataParser._get_bytes(raw, block_data_start, block_data_length)
                    # only the map_data.path points where points_mask == 1 are in mop_path
                    if map_data.path is not None:
                        map_data.mop_path = RoborockMapDataParser._parse_mop_path(map_data.path, points_mask)
                case RoborockBlockType.CARPET_MAP.value:
                    data = RoborockMapDataParser._get_bytes(raw, block_data_start, block_data_length)
                    # only the indexes where value == 1 are in carpet_map
                    map_data.carpet_map = RoborockMapDataParser._parse_carpet_map(data)
                case RoborockBlockType.NO_CARPET_AREAS.value:
                    map_data.no_carpet_areas = RoborockMapDataParser._parse_area(header, data)
                case _:
                    _LOGGER.debug(
                        "UNKNOWN BLOCK TYPE: %d, header length %d, data length %d",
                        block_type,
                        block_header_length,
                        block_data_length,
                    )

            block_start_position = block_start_position + block_data_length + RoborockMapDataParser._get_int8(header, 2)

        if (
            img_data is not None
            and img_data_length is not None
            and img_data_length is not None
            and img_header is not None
            and img_header_length is not None
        ):
            image, rooms = self._parse_image(
                img_data_length,
                img_header_length,
                img_data,
                img_header,
                map_data.carpet_map,
            )
            map_data.image = image
            map_data.rooms = rooms

        if map_data.image is not None and not map_data.image.is_empty:
            self._image_generator.draw_map(map_data)
            if (
                map_data.rooms is not None
                and len(map_data.rooms) > 0
                and map_data.vacuum_position is not None
                and img_start is not None
            ):
                map_data.vacuum_room = RoborockMapDataParser._get_current_vacuum_room(
                    img_start, raw, map_data.vacuum_position
                )
        return map_data

    @staticmethod
    def _map_to_image(p: Point) -> Point:
        return Point(p.x / 50, p.y / 50)

    @staticmethod
    def _image_to_map(x: float) -> float:
        return x * 50

    def _parse_image(
        self,
        block_data_length: int,
        block_header_length: int,
        data: bytes,
        header: bytes,
        carpet_map: set[int] | None,
    ) -> tuple[ImageData, dict[int, Room]]:
        image_size = block_data_length
        image_top = RoborockMapDataParser._get_int32(header, block_header_length - 16)
        image_left = RoborockMapDataParser._get_int32(header, block_header_length - 12)
        image_height = RoborockMapDataParser._get_int32(header, block_header_length - 8)
        image_width = RoborockMapDataParser._get_int32(header, block_header_length - 4)
        image, rooms_raw = self._image_parser.parse(data, image_width, image_height, carpet_map)
        if image is None:
            image = self._image_generator.create_empty_map_image()
        rooms = {}
        for number, room in rooms_raw.items():
            rooms[number] = Room(
                RoborockMapDataParser._image_to_map(room[0] + image_left),
                RoborockMapDataParser._image_to_map(room[1] + image_top),
                RoborockMapDataParser._image_to_map(room[2] + image_left),
                RoborockMapDataParser._image_to_map(room[3] + image_top),
                number,
            )
        return (
            ImageData(
                image_size,
                image_top,
                image_left,
                image_height,
                image_width,
                self._image_config,
                image,
                RoborockMapDataParser._map_to_image,
            ),
            rooms,
        )

    @staticmethod
    def _get_current_vacuum_room(block_start_position: int, raw: bytes, vacuum_position: Point) -> int | None:
        block_header_length = RoborockMapDataParser._get_int16(raw, block_start_position + 0x02)
        header = RoborockMapDataParser._get_bytes(raw, block_start_position, block_header_length)
        block_data_length = RoborockMapDataParser._get_int32(header, 0x04)
        block_data_start = block_start_position + block_header_length
        data = RoborockMapDataParser._get_bytes(raw, block_data_start, block_data_length)
        image_top = RoborockMapDataParser._get_int32(header, block_header_length - 16)
        image_left = RoborockMapDataParser._get_int32(header, block_header_length - 12)
        image_width = RoborockMapDataParser._get_int32(header, block_header_length - 4)
        p = RoborockMapDataParser._map_to_image(vacuum_position)
        room = RoborockImageParser.get_room_at_pixel(data, image_width, round(p.x - image_left), round(p.y - image_top))
        return room

    @staticmethod
    def _parse_carpet_map(data: bytes) -> set[int]:
        carpet_map = set()

        for i, v in enumerate(data):
            if v:
                carpet_map.add(i)
        return carpet_map

    @staticmethod
    def _parse_goto_target(data: bytes) -> Point:
        x = RoborockMapDataParser._get_int16(data, 0x00)
        y = RoborockMapDataParser._get_int16(data, 0x02)
        return Point(x, y)

    @staticmethod
    def _parse_object_position(block_data_length: int, data: bytes) -> Point:
        x = RoborockMapDataParser._get_int32(data, 0x00)
        y = RoborockMapDataParser._get_int32(data, 0x04)
        a = None
        if block_data_length > 8:
            a = RoborockMapDataParser._get_int32(data, 0x08)
            if a > 0xFF:
                a = (a & 0xFF) - 256
        return Point(x, y, a)

    @staticmethod
    def _parse_walls(data: bytes, header: bytes) -> list[Wall]:
        wall_pairs = RoborockMapDataParser._get_int16(header, 0x08)
        walls = []
        for wall_start in range(0, wall_pairs * 8, 8):
            x0 = RoborockMapDataParser._get_int16(data, wall_start + 0)
            y0 = RoborockMapDataParser._get_int16(data, wall_start + 2)
            x1 = RoborockMapDataParser._get_int16(data, wall_start + 4)
            y1 = RoborockMapDataParser._get_int16(data, wall_start + 6)
            walls.append(Wall(x0, y0, x1, y1))
        return walls

    @staticmethod
    def _parse_obstacles(data: bytes, header: bytes) -> list[Obstacle]:
        obstacle_pairs = RoborockMapDataParser._get_int16(header, 0x08)
        obstacles: list[Obstacle] = []
        if obstacle_pairs == 0:
            return obstacles
        obstacle_size = int(len(data) / obstacle_pairs)
        for obstacle_start in range(0, obstacle_pairs * obstacle_size, obstacle_size):
            x = RoborockMapDataParser._get_int16(data, obstacle_start + 0)
            y = RoborockMapDataParser._get_int16(data, obstacle_start + 2)
            details = ObstacleDetails()
            if obstacle_size >= 6:
                details.type = RoborockMapDataParser._get_int16(data, obstacle_start + 4)
                if details.type in RoborockMapDataParser.KNOWN_OBSTACLE_TYPES:
                    details.description = RoborockMapDataParser.KNOWN_OBSTACLE_TYPES[details.type]
                if obstacle_size >= 10:
                    u1 = RoborockMapDataParser._get_int16(data, obstacle_start + 6)
                    u2 = RoborockMapDataParser._get_int16(data, obstacle_start + 8)
                    details.confidence_level = 0 if u2 == 0 else u1 * 10.0 / u2
                    if obstacle_size == 28 and (data[obstacle_start + 12] & 0xFF) > 0:
                        txt = RoborockMapDataParser._get_bytes(data, obstacle_start + 12, 16)
                        details.photo_name = txt.decode("ascii")
            obstacles.append(Obstacle(x, y, details))
        return obstacles

    @staticmethod
    def _parse_zones(data: bytes, header: bytes) -> list[Zone]:
        zone_pairs = RoborockMapDataParser._get_int16(header, 0x08)
        zones = []
        for zone_start in range(0, zone_pairs * 8, 8):
            x0 = RoborockMapDataParser._get_int16(data, zone_start + 0)
            y0 = RoborockMapDataParser._get_int16(data, zone_start + 2)
            x1 = RoborockMapDataParser._get_int16(data, zone_start + 4)
            y1 = RoborockMapDataParser._get_int16(data, zone_start + 6)
            zones.append(Zone(x0, y0, x1, y1))
        return zones

    @staticmethod
    def _parse_path(block_start_position: int, header: bytes, raw: bytes) -> Path:
        path_points = []
        end_pos = RoborockMapDataParser._get_int32(header, 0x04)
        point_length = RoborockMapDataParser._get_int32(header, 0x08)
        point_size = RoborockMapDataParser._get_int32(header, 0x0C)
        angle = RoborockMapDataParser._get_int32(header, 0x10)
        start_pos = block_start_position + 0x14
        for pos in range(start_pos, start_pos + end_pos, 4):
            x = RoborockMapDataParser._get_int16(raw, pos)
            y = RoborockMapDataParser._get_int16(raw, pos + 2)
            path_points.append(Point(x, y))
        return Path(point_length, point_size, angle, [path_points])

    @staticmethod
    def _parse_mop_path(path: Path, mask: bytes) -> Path:
        mop_paths = []
        points_num = 0
        for each_path in path.path:
            mop_path_points = []
            for i, point in enumerate(each_path):
                if mask[i]:
                    mop_path_points.append(point)
                    if (i + 1) < len(mask) and not mask[i + 1]:
                        points_num += len(mop_path_points)
                        mop_paths.append(mop_path_points)
                        mop_path_points = []

            points_num += len(mop_path_points)
            mop_paths.append(mop_path_points)
        return Path(points_num, path.point_size, path.angle, mop_paths)

    @staticmethod
    def _parse_area(header: bytes, data: bytes) -> list[Area]:
        area_pairs = RoborockMapDataParser._get_int16(header, 0x08)
        areas = []
        for area_start in range(0, area_pairs * 16, 16):
            x0 = RoborockMapDataParser._get_int16(data, area_start + 0)
            y0 = RoborockMapDataParser._get_int16(data, area_start + 2)
            x1 = RoborockMapDataParser._get_int16(data, area_start + 4)
            y1 = RoborockMapDataParser._get_int16(data, area_start + 6)
            x2 = RoborockMapDataParser._get_int16(data, area_start + 8)
            y2 = RoborockMapDataParser._get_int16(data, area_start + 10)
            x3 = RoborockMapDataParser._get_int16(data, area_start + 12)
            y3 = RoborockMapDataParser._get_int16(data, area_start + 14)
            areas.append(Area(x0, y0, x1, y1, x2, y2, x3, y3))
        return areas

    @staticmethod
    def _get_bytes(data: bytes, start_index: int, size: int) -> bytes:
        return data[start_index : start_index + size]

    @staticmethod
    def _get_int8(data: bytes, address: int) -> int:
        return data[address] & 0xFF

    @staticmethod
    def _get_int16(data: bytes, address: int) -> int:
        return ((data[address + 0] << 0) & 0xFF) | ((data[address + 1] << 8) & 0xFFFF)

    @staticmethod
    def _get_int32(data: bytes, address: int) -> int:
        return (
            ((data[address + 0] << 0) & 0xFF)
            | ((data[address + 1] << 8) & 0xFFFF)
            | ((data[address + 2] << 16) & 0xFFFFFF)
            | ((data[address + 3] << 24) & 0xFFFFFFFF)
        )
