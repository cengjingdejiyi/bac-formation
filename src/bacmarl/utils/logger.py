"""日志：metrics.csv 为主，TensorBoard 可选（负责人 E）。"""
from __future__ import annotations

import csv
from pathlib import Path


class RunLogger:
    def __init__(self, run_dir: str | Path, use_tb: bool = True):
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.dir / 'metrics.csv'
        self._keys: list[str] | None = None
        self._writer = None
        self._fh = None
        self.tb = None
        if use_tb:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.tb = SummaryWriter(log_dir=str(self.dir / 'tb'))
            except Exception:
                self.tb = None

    def log(self, row: dict, step: int) -> None:
        row = dict(row)
        row['env_steps'] = int(step)
        if self._writer is None:
            self._keys = ['env_steps'] + sorted(k for k in row if k != 'env_steps')
            self._fh = open(self.csv_path, 'w', newline='', encoding='utf-8')
            self._writer = csv.DictWriter(self._fh, fieldnames=self._keys)
            self._writer.writeheader()
        full = {k: row.get(k, '') for k in self._keys}
        self._writer.writerow(full)
        self._fh.flush()
        if self.tb is not None:
            for k, v in row.items():
                if isinstance(v, (int, float)):
                    self.tb.add_scalar(k, v, step)

    def close(self) -> None:
        if self._fh:
            self._fh.close()
        if self.tb is not None:
            self.tb.close()
