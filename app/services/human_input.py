"""Resolve held directions and fire to one supported Atari action."""

# ALE action numbers are stable; PettingZoo exposes per-game indices into them.
# Button bits: up=1, right=2, down=4, left=8, fire=16.
_ACTION_BUTTONS = (0, 16, 1, 2, 8, 4, 3, 9, 6, 12,
                   17, 18, 24, 20, 19, 25, 22, 28)
_DIRECTIONS = {
    '8': (0, -1), '9': (1, -1), '6': (1, 0), '3': (1, 1),
    '2': (0, 1), '1': (-1, 1), '4': (-1, 0), '7': (-1, -1),
}
_ALIASES = {
    'q': '7', 'w': '8', 'e': '9', 'a': '4', 'd': '6',
    'z': '1', 'x': '2', 'c': '3', ' ': 's',
    'arrowup': '8', 'arrowright': '6', 'arrowdown': '2', 'arrowleft': '4',
}


def resolve_human_action(keys, action_mapping):
    """Combine keys independently of press order, without enlarging action spaces.

    Opposing directions cancel. If a game lacks the exact combination, retain
    as many requested buttons as possible, preferring movement on ties.
    """
    if not isinstance(keys, (list, tuple)):
        keys = ()
    held = {_ALIASES.get(key.lower(), key.lower())
            for key in keys if isinstance(key, str)}
    directions = [_DIRECTIONS[key] for key in held if key in _DIRECTIONS]
    left = any(x < 0 for x, _ in directions)
    right = any(x > 0 for x, _ in directions)
    up = any(y < 0 for _, y in directions)
    down = any(y > 0 for _, y in directions)
    buttons = ((1 if up and not down else 0)
               | (2 if right and not left else 0)
               | (4 if down and not up else 0)
               | (8 if left and not right else 0)
               | (16 if 's' in held else 0))
    candidates = []
    for index, action in enumerate(action_mapping):
        mask = _ACTION_BUTTONS[int(action)]
        if mask == buttons:
            return index
        if mask & buttons == mask:
            candidates.append((bin(mask).count('1'), bin(mask & 15).count('1'), -index))
    # Some games (Wizard of Wor) have no idle action: retain index zero.
    return -max(candidates)[2] if candidates else 0
