"""Optional TensorBoard summaries for one autonomous training run."""

from collections import defaultdict
from datetime import datetime, timezone
import logging
import os
from pathlib import Path
from uuid import uuid4

import tensorflow as tf


_LOGGER = logging.getLogger(__name__)


class TrainingMetrics:
    def __init__(self, environment):
        self.writer = None
        self.log_dir = None
        self.updates = defaultdict(int)
        if os.getenv("TENSORBOARD_ENABLED", "false").lower() not in {"true", "1", "yes"}:
            return
        run = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex
        self.log_dir = Path(os.getenv("TENSORBOARD_LOG_DIR", "logs/tensorboard")) / environment / run
        try:
            self.writer = tf.summary.create_file_writer(str(self.log_dir))
            _LOGGER.info("TensorBoard training run: %s", self.log_dir)
        except Exception:
            _LOGGER.warning("Could not create TensorBoard writer; training will continue", exc_info=True)

    def scalar(self, name, value, step):
        if self.writer is None:
            return
        try:
            with self.writer.as_default():
                tf.summary.scalar(name, value, step=step)
        except Exception:
            _LOGGER.warning("TensorBoard logging failed; disabling this run's writer", exc_info=True)
            self.close()

    def update(self, agent_name, loss):
        # A replay buffer below the minibatch threshold does not perform an update.
        if loss is None:
            return
        self.updates[agent_name] += 1
        self.scalar(f"train/{agent_name}/loss", loss, self.updates[agent_name])

    def flush(self):
        if self.writer is not None:
            try:
                self.writer.flush()
            except Exception:
                _LOGGER.warning("Could not flush TensorBoard summaries", exc_info=True)
                self.close()

    def close(self):
        writer, self.writer = self.writer, None
        if writer is not None:
            try:
                writer.close()
            except Exception:
                _LOGGER.warning("Could not close TensorBoard writer", exc_info=True)
