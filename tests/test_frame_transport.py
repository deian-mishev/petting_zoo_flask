"""Exercise binary frame credit and controls over real loopback transports."""

import os
import json
from pathlib import Path
import runpy
import threading
import unittest
from unittest.mock import patch

from flask import Flask, request
from flask_socketio import SocketIO
import socketio as socketio_client
from simple_websocket import Client as WebSocketClient
from werkzeug.serving import make_server, WSGIRequestHandler

FrameDelivery = runpy.run_path(str(Path(__file__).resolve().parents[1] /
                                  "app/services/frame_delivery.py"))["FrameDelivery"]


class QuietHandler(WSGIRequestHandler):
    def log(self, *args, **kwargs):
        pass


class FrameTransportTests(unittest.TestCase):
    def check_transport(self, transport):
        app = Flask(__name__)
        app.secret_key = "stream-test"
        server_socket = SocketIO(app, async_mode="threading", ping_interval=.5, ping_timeout=5)
        stop = threading.Event()
        allow_decode = threading.Event()
        window_full = threading.Event()
        resumed = threading.Event()
        worker_done = threading.Event()
        received = []
        inputs = []
        failures = []

        @server_socket.on('input')
        def on_input(keys):
            inputs.append(keys)
            return keys

        @server_socket.on('begin')
        def begin():
            sid = request.sid

            def produce():
                try:
                    delivery = FrameDelivery(lambda frame, **kw: server_socket.emit(
                        'frame', frame, to=sid, **kw))
                    sequence = 0
                    while not stop.wait(.005):
                        sequence += 1
                        delivery.send_if_due(lambda: str(sequence).encode())
                except Exception as error:
                    failures.append(error)
                finally:
                    worker_done.set()

            server_socket.start_background_task(produce)

        http = make_server('127.0.0.1', 0, app, threaded=True, request_handler=QuietHandler)
        thread = threading.Thread(target=http.serve_forever, daemon=True)
        thread.start()
        client = socketio_client.Client(reconnection=False)

        @client.on('frame')
        def on_frame(data):
            received.append(data)
            if len(received) == 3:
                window_full.set()
            if len(received) > 3:
                resumed.set()
            # Return sends the real Socket.IO acknowledgement only after decoding.
            allow_decode.wait(5)

        try:
            with patch.dict(os.environ, {"STREAM_FPS": "30", "STREAM_MAX_IN_FLIGHT": "3"}):
                client.connect(f'http://127.0.0.1:{http.server_port}', transports=[transport])
                client.emit('begin')
                self.assertTrue(window_full.wait(3), f'{transport}: no initial frames')
                self.assertFalse(resumed.wait(.15), 'Frames exceeded the unacknowledged window')
                self.assertEqual(client.call('input', ['7'], timeout=2), ['7'])
                self.assertEqual(client.call('input', [], timeout=2), [])
                self.assertEqual(inputs, [['7'], []])
                allow_decode.set()
                self.assertTrue(resumed.wait(3), 'Acknowledgements did not resume streaming')
                self.assertTrue(all(isinstance(frame, bytes) for frame in received))
                self.assertEqual(failures, [])
        finally:
            stop.set()
            allow_decode.set()
            worker_done.wait(1)
            if client.connected:
                client.disconnect()
            http.shutdown()
            http.server_close()
            thread.join(timeout=2)

    def test_websocket_frames_and_input(self):
        # Exercise Socket.IO's binary event/ack wire format using the WebSocket
        # library already required by the server; no optional client dependency.
        app = Flask(__name__)
        server = SocketIO(app, async_mode='threading')
        now = [0.0]
        deliveries = {}
        inputs = []

        @server.on('begin')
        def begin():
            sid = request.sid
            delivery = FrameDelivery(lambda frame, **kw: server.emit('frame', frame, to=sid, **kw),
                                     clock=lambda: now[0])
            deliveries[sid] = delivery
            for index in range(4):
                now[0] += .04
                delivery.send_if_due(lambda: b'jpeg')

        @server.on('input')
        def on_input(keys):
            inputs.append(keys)
            return keys

        @server.on('next')
        def next_frame():
            now[0] += .04
            deliveries[request.sid].send_if_due(lambda: b'newest')

        http = make_server('127.0.0.1', 0, app, threaded=True, request_handler=QuietHandler)
        thread = threading.Thread(target=http.serve_forever, daemon=True)
        thread.start()
        client = None
        try:
            with patch.dict(os.environ, {"STREAM_FPS": "30", "STREAM_MAX_IN_FLIGHT": "3"}):
                client = WebSocketClient.connect(
                    f'ws://127.0.0.1:{http.server_port}/socket.io/?EIO=4&transport=websocket')
                self.assertTrue(client.receive(timeout=3).startswith('0'))  # Engine.IO handshake
                client.send('40')  # Socket.IO namespace connect
                self.assertTrue(client.receive(timeout=3).startswith('40'))
                client.send('42["begin"]')
                ack_ids = []
                for _ in range(3):
                    header = client.receive(timeout=3)
                    self.assertTrue(header.startswith('451-'))  # Binary event, one attachment
                    ack_ids.append(header[4:header.index('[')])
                    self.assertEqual(client.receive(timeout=3), b'jpeg')
                self.assertIsNone(client.receive(timeout=.15))
                client.send('421["input",["7"]]')
                self.assertEqual(client.receive(timeout=3), '431' + json.dumps([['7']], separators=(',', ':')))
                client.send('422["input",[]]')
                self.assertEqual(client.receive(timeout=3), '432[[]]')
                self.assertEqual(inputs, [['7'], []])
                client.send('43' + ack_ids[0] + '[]')
                client.send('42["next"]')
                self.assertTrue(client.receive(timeout=3).startswith('451-'))
                self.assertEqual(client.receive(timeout=3), b'newest')
        finally:
            if client is not None:
                client.close()
            http.shutdown()
            http.server_close()
            thread.join(timeout=2)

    def test_polling_frames_and_input(self):
        self.check_transport('polling')


if __name__ == '__main__':
    unittest.main()
