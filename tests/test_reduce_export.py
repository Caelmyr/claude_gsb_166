"""Regression tests for reduce result completeness and CSV export."""

import collections
import csv
import http.server
import io
import json
import os
import shutil
import sys
import tempfile
import threading
import types
import unittest

from backend.common.hashing import partition_for
from backend.worker.executor import _run_map, _run_reduce
from backend.worker.shuffle_store import ShuffleStore


class _ShuffleHandler(http.server.BaseHTTPRequestHandler):
    """Serves a map task's shuffle partition files exactly like a worker."""

    root = ""

    def do_GET(self):
        # /shuffle/{job}/{task}/part-XXXX.jsonl
        rel = self.path.split("/", 2)[2]
        path = os.path.join(self.root, rel)
        pairs = []
        try:
            with open(path, encoding="utf-8") as f:
                pairs = [json.loads(line) for line in f if line.strip()]
        except OSError:
            self.send_response(404)
            self.end_headers()
            return
        body = json.dumps(pairs).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class TestReduceFlushesLastKey(unittest.TestCase):
    """The sorted-stream group loop must flush its final group.

    A previous ``len(results) < 0`` guard made the trailing flush a no-op, so
    the last key (in sort order) of every reduce partition vanished.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.job_id = "job-rt"
        self.num_partitions = 3
        # Deliberately include words that sort to the end of the alphabet so a
        # dropped trailing group is observable.
        self.lines = [
            "alpha bravo zulu alpha",
            "charlie delta yankee bravo",
            "echo xray zulu foxtrot",
            "alpha golf hotel echo",
        ]
        self.reference = collections.Counter()
        for line in self.lines:
            self.reference.update(line.split())

        self.map_ids = ["m-0000", "m-0001"]
        for idx, mid in enumerate(self.map_ids):
            _run_map({
                "task_id": mid, "job_id": self.job_id, "kind": "map",
                "mapper": "wordcount_mapper", "params": {},
                "partition_count": self.num_partitions,
                "records": self.lines[idx::2],
                "spill_records": 10,
            }, self.tmp, lambda p, a, b: None)

        handler = type("H", (_ShuffleHandler,), {"root": os.path.join(self.tmp, "shuffle")})
        self.server = http.server.HTTPServer(("127.0.0.1", 0), handler)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def tearDown(self):
        self.server.shutdown()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_every_key_reduced_including_partition_tails(self):
        reduced: dict[str, int] = {}
        for p in range(self.num_partitions):
            out = _run_reduce({
                "task_id": f"r-{p:04d}", "job_id": self.job_id, "kind": "reduce",
                "reducer": "count_reducer", "params": {}, "partition": p,
                "fetch_plan": [
                    {"worker_url": f"http://127.0.0.1:{self.port}", "map_task_id": mid}
                    for mid in self.map_ids
                ],
                "spill_records": 5, "tmp_dir": self.tmp,
            }, lambda prog, a, b: None)
            for rec in out["results"]:
                reduced[rec["key"]] = rec["count"]

        # Every distinct key must survive, with the correct summed count.
        self.assertEqual(set(reduced), set(self.reference))
        self.assertEqual(reduced, dict(self.reference))

        # Explicitly pin the previously-dropped trailing key of each partition.
        tails: dict[int, list[str]] = collections.defaultdict(list)
        for word in self.reference:
            tails[partition_for(word, self.num_partitions)].append(word)
        for words in tails.values():
            self.assertIn(max(words), reduced)


class TestCsvExport(unittest.TestCase):
    """CSV export must lead with ``key`` and include columns from all rows."""

    @staticmethod
    def _as_csv(records):
        # backend.master.server imports Flask; stub it out so this stays a
        # stdlib-only test (only the CSV builder is exercised here).
        if "flask" not in sys.modules:
            flask = types.ModuleType("flask")

            class _Response:
                def __init__(self, body, mimetype=None, headers=None):
                    self.body = body

            flask.Response = _Response
            flask.Flask = lambda *a, **k: types.SimpleNamespace(
                add_url_rule=lambda *a, **k: None)
            flask.jsonify = lambda *a, **k: None
            flask.request = types.SimpleNamespace(args={})
            flask.send_from_directory = lambda *a, **k: None
            sys.modules["flask"] = flask

        from backend.master.server import Master
        master = Master.__new__(Master)
        job = types.SimpleNamespace(job_id="jobx")
        return master._as_csv(job, records).body

    def test_key_column_and_late_statistics_columns_exported(self):
        body = self._as_csv([
            {"key": "alpha", "count": 3},
            {"key": "bravo", "count": 5, "sum": 20, "avg": 4.0},
        ])
        rows = list(csv.reader(io.StringIO(body)))
        self.assertEqual(rows[0][0], "key")
        self.assertEqual(rows[0], ["key", "count", "sum", "avg"])
        # Columns first seen on a later row must still carry their data.
        self.assertEqual(rows[2], ["bravo", "5", "20", "4.0"])
        self.assertEqual(rows[1], ["alpha", "3", "", ""])

    def test_empty_records_have_header_at_least(self):
        body = self._as_csv([])
        rows = list(csv.reader(io.StringIO(body)))
        self.assertEqual(rows, [["key"]])


if __name__ == "__main__":
    unittest.main()
