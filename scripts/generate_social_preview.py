"""Generate the pixel-art sharing card with Python's standard library."""

from pathlib import Path
import struct
import zlib


WIDTH, HEIGHT = 1200, 630
PIXELS = bytearray(WIDTH * HEIGHT * 3)
FONT = {
    'A': '01110 10001 10001 11111 10001 10001 10001',
    'B': '11110 10001 10001 11110 10001 10001 11110',
    'C': '01111 10000 10000 10000 10000 10000 01111',
    'D': '11110 10001 10001 10001 10001 10001 11110',
    'E': '11111 10000 10000 11110 10000 10000 11111',
    'F': '11111 10000 10000 11110 10000 10000 10000',
    'G': '01111 10000 10000 10111 10001 10001 01111',
    'H': '10001 10001 10001 11111 10001 10001 10001',
    'I': '11111 00100 00100 00100 00100 00100 11111',
    'J': '00111 00010 00010 00010 10010 10010 01100',
    'K': '10001 10010 10100 11000 10100 10010 10001',
    'L': '10000 10000 10000 10000 10000 10000 11111',
    'M': '10001 11011 10101 10101 10001 10001 10001',
    'N': '10001 11001 10101 10011 10001 10001 10001',
    'O': '01110 10001 10001 10001 10001 10001 01110',
    'P': '11110 10001 10001 11110 10000 10000 10000',
    'Q': '01110 10001 10001 10001 10101 10010 01101',
    'R': '11110 10001 10001 11110 10100 10010 10001',
    'S': '01111 10000 10000 01110 00001 00001 11110',
    'T': '11111 00100 00100 00100 00100 00100 00100',
    'U': '10001 10001 10001 10001 10001 10001 01110',
    'V': '10001 10001 10001 10001 10001 01010 00100',
    'W': '10001 10001 10001 10101 10101 10101 01010',
    'X': '10001 10001 01010 00100 01010 10001 10001',
    'Y': '10001 10001 01010 00100 00100 00100 00100',
    'Z': '11111 00001 00010 00100 01000 10000 11111',
    '.': '00000 00000 00000 00000 00000 00110 00110',
    ' ': '00000 00000 00000 00000 00000 00000 00000',
}
CREAM = (234, 243, 216)
GREEN = (200, 230, 160)
MUTED = (143, 174, 142)


def rect(x, y, width, height, color):
    for row in range(max(0, y), min(HEIGHT, y + height)):
        left, right = max(0, x), min(WIDTH, x + width)
        if right > left:
            offset = (row * WIDTH + left) * 3
            PIXELS[offset:offset + (right - left) * 3] = bytes(color) * (right - left)


def text(value, x, y, scale, color):
    for char in value:
        for row, bits in enumerate(FONT[char].split()):
            for col, bit in enumerate(bits):
                if bit == '1':
                    rect(x + col * scale, y + row * scale, scale, scale, color)
        x += scale * 6


def generate():
    rect(0, 0, WIDTH, HEIGHT, (16, 37, 27))
    for x in range(0, WIDTH, 40):
        rect(x, 0, 1, HEIGHT, (22, 46, 33))
    for y in range(0, HEIGHT, 40):
        rect(0, y, WIDTH, 1, (22, 46, 33))
    rect(56, 58, 8, 26, GREEN)
    text('THE AI ARCADE', 80, 64, 3, GREEN)
    text('ATARI', 56, 140, 12, CREAM)
    text('PETTING', 56, 244, 12, CREAM)
    text('ZOO', 56, 348, 12, GREEN)
    text('CLASSIC GAMES. LEARNING AI.', 58, 462, 3, MUTED)
    text('ATARIPETTINGZOO.COM', 58, 552, 3, GREEN)

    # A CRT cabinet with a simple paddle game, rather than copyrighted artwork.
    rect(640, 113, 492, 382, (8, 21, 16))
    rect(628, 101, 492, 382, (66, 91, 58))
    rect(646, 119, 456, 316, (9, 25, 18))
    rect(660, 133, 428, 288, (24, 54, 36))
    for y in range(140, 420, 22):
        rect(871, y, 5, 12, (79, 110, 69))
    rect(690, 221, 12, 74, GREEN)
    rect(1046, 292, 12, 74, GREEN)
    rect(934, 251, 16, 16, CREAM)
    rect(916, 256, 9, 7, (79, 110, 69))
    for y in range(136, 421, 4):
        rect(660, y, 428, 1, (19, 46, 30))
    text('HUMAN VS AI', 666, 453, 3, GREEN)
    rect(1080, 452, 12, 12, GREEN)
    rect(797, 483, 152, 24, (66, 91, 58))
    rect(748, 507, 250, 12, (66, 91, 58))

    def chunk(kind, data):
        return (struct.pack('!I', len(data)) + kind + data
                + struct.pack('!I', zlib.crc32(kind + data) & 0xffffffff))

    rows = b''.join(b'\0' + PIXELS[y * WIDTH * 3:(y + 1) * WIDTH * 3]
                    for y in range(HEIGHT))
    png = (b'\x89PNG\r\n\x1a\n'
           + chunk(b'IHDR', struct.pack('!2I5B', WIDTH, HEIGHT, 8, 2, 0, 0, 0))
           + chunk(b'IDAT', zlib.compress(rows, 9)) + chunk(b'IEND', b''))
    destination = Path(__file__).resolve().parents[1] / 'app/static/images/social-preview.png'
    destination.write_bytes(png)
    print(f'Generated {destination.name}: {WIDTH} x {HEIGHT}, {len(png)} bytes')


if __name__ == '__main__':
    generate()
