"""Regression tests for result completeness and CSV export.

Covers two reported bugs:

* the reducer dropped the final key group of every partition, so a WordCount
  job came back with fewer keys than expected (the last keys vanished);
* the CSV export omitted the ``key`` identifier column and any statistics
  columns that only appeared beyond the first 50 records.
"""

import collections
import csv
import io
import shutil
import tempfile
import unittest
from unittest import mock

from backend.common.http_client import HttpClient
from backend.tasks.samples import generate_input_records
from backend.worker.executor import _run_map, _run_reduce
from backend.worker.shuffle_store import ShuffleStore, parse_partition_index

try:
    import flask  # noqa: F401
    _HAS_FLASK = True
except ImportError:
    _HAS_FLASK = False


class TestReduceCompleteness(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _wordcount(self, records, partitions):
        """Run a real map + reduce over local shuffle files; return {key: count}."""
        map_spec = {
            "task_id": "m-0000", "job_id": "job", "kind": "map",
            "mapper": "wordcount_mapper", "reducer": "count_reducer",
            "params": {}, "partition_count": partitions, "records": records,
            "spill_records": 100, "tmp_dir": self.tmp,
        }
        _run_map(map_spec, self.tmp, lambda p, a, b: None)

        store = ShuffleStore(self.tmp)

        def fake_get_json(self_client, url, default=None):
            part_file = url.rsplit("/", 1)[-1]
            return store.read_partition("job", "m-0000", parse_partition_index(part_file))

        counts = {}
        with mock.patch.object(HttpClient, "get_json", fake_get_json):
            for p in range(partitions):
                spec = {
                    "task_id": f"r-{p:04d}", "job_id": "job", "kind": "reduce",
                    "mapper": "wordcount_mapper", "reducer": "count_reducer",
                    "params": {}, "partition": p,
                    "fetch_plan": [{"worker_url": "http://unused", "map_task_id": "m-0000"}],
                    "spill_records": 100, "tmp_dir": self.tmp,
                }
                out = _run_reduce(spec, lambda pr, a, b: None)
                for rec in out["results"]:
                    counts[rec["key"]] = rec["count"]
        return counts

    def test_reduce_emits_every_key(self):
        records = generate_input_records("wordcount", 400, seed=3)
        reference = collections.Counter()
        for line in records:
            for w in line.lower().split():
                reference[w] += 1
        self.assertEqual(self._wordcount(records, partitions=3), dict(reference))

    def test_single_key_partition_is_not_dropped(self):
        # A partition holding exactly one key group still emits it (the old
        # flush condition swallowed the final group unconditionally).
        self.assertEqual(self._wordcount(["apple"], partitions=1), {"apple": 1})


@unittest.skipUnless(_HAS_FLASK, "flask is not installed")
class TestCsvExport(unittest.TestCase):
    def _csv_rows(self, records):
        from backend.common.models import Job
        from backend.master.server import Master

        job = Job(job_id="j1", name="wc", mapper="wordcount_mapper",
                  reducer="count_reducer", num_map_tasks=1, num_reduce_tasks=1)
        resp = Master._as_csv(Master.__new__(Master), job, records)
        return list(csv.reader(io.StringIO(resp.get_data(as_text=True))))

    def test_header_starts_with_key(self):
        rows = self._csv_rows([{"key": "a", "count": 1}])
        self.assertEqual(rows[0][0], "key")
        self.assertEqual(rows[1], ["a", "1"])

    def test_columns_discovered_beyond_first_50_records(self):
        records = [{"key": f"w{i}", "count": i} for i in range(60)]
        records.append({"key": "late", "count": 1, "sum": 5, "avg": 2.5})
        rows = self._csv_rows(records)
        header = rows[0]
        self.assertEqual(header, ["key", "count", "sum", "avg"])
        late = rows[-1]
        self.assertEqual(late[header.index("sum")], "5")
        self.assertEqual(late[header.index("avg")], "2.5")
        self.assertEqual(len(rows) - 1, len(records))


if __name__ == "__main__":
    unittest.main()
