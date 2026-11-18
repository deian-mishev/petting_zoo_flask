"""Measure Atari capture/JPEG work without loading, training, or saving models.

Run: python scripts/benchmark_rendering.py --game boxing_v2 --frames 300
"""

import argparse
import importlib
import random
import statistics
import time

import cv2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--game', default='boxing_v2', choices=(
        'boxing_v2', 'tennis_v3', 'wizard_of_wor_v3', 'mario_bros_v3',
        'ice_hockey_v2', 'double_dunk_v3', 'pong_v3', 'space_invaders_v2'))
    parser.add_argument('--frames', type=int, default=300)
    args = parser.parse_args()
    if args.frames < 1:
        parser.error('--frames must be positive')
    factory = importlib.import_module(f'pettingzoo.atari.{args.game}')
    env = factory.env(render_mode='rgb_array')
    rng = random.Random(0)
    capture_times = []
    conversion_times = []
    results = {quality: {'times': [], 'sizes': []} for quality in (60, 75, 90)}
    try:
        env.reset(seed=0)
        agents = iter(env.agent_iter())
        for _ in range(args.frames):
            # Two player turns advance one emulator frame. Keep simulation out
            # of the capture/encoding timing and use identical frames per quality.
            for _ in range(2):
                agent = next(agents)
                _, _, terminated, truncated, _ = env.last()
                if terminated or truncated:
                    env.reset(seed=0)
                    agents = iter(env.agent_iter())
                    agent = next(agents)
                env.step(rng.randrange(env.action_space(agent).n))
            start = time.perf_counter()
            rgb = env.render()
            captured = time.perf_counter()
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            converted = time.perf_counter()
            capture_times.append((captured - start) * 1000)
            conversion_times.append((converted - captured) * 1000)
            for quality, values in results.items():
                start = time.perf_counter()
                success, buffer = cv2.imencode('.jpg', bgr, (cv2.IMWRITE_JPEG_QUALITY, quality))
                if not success:
                    raise RuntimeError('JPEG encoding failed')
                payload = buffer.tobytes()
                values['times'].append((time.perf_counter() - start) * 1000)
                values['sizes'].append(len(payload))
        print(f'{args.game}: {args.frames} samples, {rgb.shape[1]} x {rgb.shape[0]} pixels')
        print(f'Capture median: {statistics.median(capture_times):.3f} ms')
        print(f'RGB conversion median: {statistics.median(conversion_times):.3f} ms')
        for quality, values in results.items():
            mean_size = statistics.mean(values['sizes'])
            print(f'JPEG {quality}: median {statistics.median(values["times"]):.3f} ms, '
                  f'mean {mean_size:.0f} bytes, '
                  f'{mean_size * 30 / 1024:.1f} KiB/s payload at 30 FPS')
        print('Excludes inference, transport overhead, browser decoding, and network latency.')
    finally:
        env.close()


if __name__ == '__main__':
    main()
