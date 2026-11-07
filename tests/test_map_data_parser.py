"""Tests for RoborockMapDataParser._parse_obstacles photo_name extraction.

Synthetic payloads only: no real map data and no real photo IDs are used.
"""

import struct
import unittest

from vacuum_map_parser_roborock.map_data_parser import RoborockMapDataParser

FAKE_PHOTO_ID = "0123456789abcdef"


def _header(obstacle_count: int) -> bytes:
    header = bytearray(10)
    struct.pack_into("<H", header, 8, obstacle_count)
    return bytes(header)


def _entry(entry_size: int, photo_id: str | None) -> bytes:
    entry = bytearray(entry_size)
    struct.pack_into("<h", entry, 0, 25548)  # x
    struct.pack_into("<h", entry, 2, 23428)  # y
    struct.pack_into("<h", entry, 4, 48)  # type: cable
    struct.pack_into("<h", entry, 6, 9)  # confidence numerator
    struct.pack_into("<h", entry, 8, 10)  # confidence denominator
    if photo_id is not None:
        entry[12:28] = photo_id.encode("ascii")
    return bytes(entry)


class TestParseObstaclesPhotoName(unittest.TestCase):
    def test_36_byte_entry_photo_name_is_parsed(self) -> None:
        """Current firmware writes 36-byte entries; photo ID is at offset 12."""
        data = _entry(36, FAKE_PHOTO_ID)
        obstacles = RoborockMapDataParser._parse_obstacles(data, _header(1))
        self.assertEqual(len(obstacles), 1)
        self.assertEqual(obstacles[0].details.photo_name, FAKE_PHOTO_ID)
        self.assertEqual(obstacles[0].details.type, 48)

    def test_28_byte_entry_photo_name_is_parsed(self) -> None:
        """Legacy 28-byte entries keep working exactly as before."""
        data = _entry(28, FAKE_PHOTO_ID)
        obstacles = RoborockMapDataParser._parse_obstacles(data, _header(1))
        self.assertEqual(len(obstacles), 1)
        self.assertEqual(obstacles[0].details.photo_name, FAKE_PHOTO_ID)

    def test_36_byte_entry_without_photo_has_no_photo_name(self) -> None:
        data = _entry(36, None)
        obstacles = RoborockMapDataParser._parse_obstacles(data, _header(1))
        self.assertEqual(len(obstacles), 1)
        self.assertIsNone(obstacles[0].details.photo_name)


if __name__ == "__main__":
    unittest.main()
