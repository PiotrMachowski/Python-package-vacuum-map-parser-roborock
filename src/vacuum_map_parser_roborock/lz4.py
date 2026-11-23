"""Decoder for raw LZ4 blocks (the block format, without the LZ4 frame)."""

MIN_MATCH = 4


def _read_length(data: bytes, pos: int, length: int) -> tuple[int, int]:
    """Extend a 4-bit length with the following bytes if it is 15 (every 255 byte adds and continues)."""
    if length != 0xF:
        return length, pos
    while True:
        value = data[pos]
        pos += 1
        length += value
        if value != 0xFF:
            return length, pos


def decompress_block(src: bytes, out: bytearray) -> None:
    """Decompress one LZ4 block and append the result to ``out``.

    Matches may refer back into data that is already in ``out``, so several blocks
    that share their history can be decompressed one after another into the same buffer.
    """
    pos = 0
    end = len(src)
    while pos < end:
        token = src[pos]
        pos += 1

        literal_length, pos = _read_length(src, pos, token >> 4)
        out += src[pos : pos + literal_length]
        pos += literal_length
        if pos >= end:
            # the last sequence of a block has literals only
            break

        offset = src[pos] | (src[pos + 1] << 8)
        pos += 2
        if offset == 0 or offset > len(out):
            raise ValueError(f"Invalid LZ4 match offset {offset} at output position {len(out)}")
        match_length, pos = _read_length(src, pos, token & 0xF)
        match_length += MIN_MATCH

        start = len(out) - offset
        if offset >= match_length:
            out += out[start : start + match_length]
        else:
            # overlapping match: repeats the last ``offset`` bytes
            for i in range(match_length):
                out.append(out[start + i])
