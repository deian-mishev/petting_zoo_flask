from itertools import permutations
from pathlib import Path
import runpy
import unittest


resolve = runpy.run_path(str(Path(__file__).resolve().parents[1] /
                            'app/services/human_input.py'))['resolve_human_action']
FULL = tuple(range(18))
PADDLE = (0, 1, 3, 4, 11, 12)  # Pong and Space Invaders minimal action set.
WIZARD = (1, 2, 3, 4, 5, 10, 11, 12, 13)


class HumanInputTests(unittest.TestCase):
    def test_all_directions_with_fire_use_combined_actions(self):
        for key, expected in [('8', 10), ('6', 11), ('4', 12), ('2', 13),
                              ('9', 14), ('7', 15), ('3', 16), ('1', 17)]:
            for keys in permutations([key, 's']):
                with self.subTest(keys=keys):
                    self.assertEqual(resolve(keys, FULL), expected)

    def test_two_direction_keys_and_fire_work_in_any_order(self):
        for keys in permutations(['w', 'd', ' ']):
            self.assertEqual(resolve(keys, FULL), 14)
        self.assertEqual(resolve(['arrowup', 'arrowleft'], FULL), 7)

    def test_releasing_fire_or_movement_keeps_other_held_button(self):
        self.assertEqual(resolve(['6', 's'], FULL), 11)
        self.assertEqual(resolve(['6'], FULL), 3)
        self.assertEqual(resolve(['s'], FULL), 1)
        self.assertEqual(resolve([], FULL), 0)

    def test_opposing_directions_cancel_without_cancelling_fire(self):
        self.assertEqual(resolve(['4', '6'], FULL), 0)
        self.assertEqual(resolve(['8', '2', 's'], FULL), 1)
        self.assertEqual(resolve(['9', '4'], FULL), 2)

    def test_aliases_do_not_duplicate_direction_and_unknown_keys_are_ignored(self):
        self.assertEqual(resolve(['W', '8', 'd', ' ', 's'], FULL), 14)
        self.assertEqual(resolve(['Shift', 'Control', 2, None], FULL), 0)
        for malformed in [None, {}, 's']:
            self.assertEqual(resolve(malformed, FULL), 0)

    def test_reduced_action_sets_use_their_own_indices(self):
        for mapping in [PADDLE, WIZARD]:
            for keys in permutations(['6', 's']):
                self.assertEqual(mapping[resolve(keys, mapping)], 11)
        self.assertEqual(resolve(['8', 's'], WIZARD), 5)
        self.assertEqual(resolve(['4', 's'], PADDLE), 5)

    def test_unsupported_buttons_fall_back_to_supported_requested_buttons(self):
        self.assertEqual(PADDLE[resolve(['8', 's'], PADDLE)], 1)
        self.assertIn(WIZARD[resolve(['7', 's'], WIZARD)], (10, 12))
        self.assertEqual(resolve([], WIZARD), 0)
        # Without combined actions, retain movement instead of freezing to fire.
        self.assertEqual(resolve(['6', 's'], (0, 1, 3)), 2)


if __name__ == '__main__':
    unittest.main()
