class Filter:
    """
    A simple object for preserving the position and bit pattern of a Square,
    Circular, or Triangular watermark.
    """
    def __init__(self, pos: tuple[int, int], bit: int):
        self._pos = pos
        self._bit = bit

    def get_pos(self) -> tuple[int, int]:
        return self._pos

    def get_bit(self) -> int:
        return self._bit