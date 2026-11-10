"""Convert per-job ValuePLCLog CSVs into per-device sensor and job Parquets.

The CSV clock is interpreted as Asia/Shanghai local time. InfluxDB job records
are used only for task names and authoritative job start/end timestamps. Every
CSV observation is retained, including duplicate timestamps; no resampling,
imputation, or train/validation/test selection is performed here.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
FILE_PATTERN = re.compile(
    r"^ValuePLCLog_(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})_Machine(0[1-4])\.csv$"
)
BEIJING = "Asia/Shanghai"
DATE_FIELDS = {"负压测试日期", "正压测试日期"}


def collect_csv_files(input_dir: Path) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(input_dir.glob("*.csv")):
        match = FILE_PATTERN.fullmatch(path.name)
        if match is None:
            raise ValueError(f"Unexpected CSV filename: {path.name}")
        device_number = int(match.group(3))
        host = f"{device_number:02d}-X450-{device_number}"
        grouped[host].append(path)
    if not grouped:
        raise FileNotFoundError(f"No ValuePLCLog CSV files in {input_dir}")
    return dict(sorted(grouped.items()))


def read_headers(files: dict[str, list[Path]]) -> list[str]:
    base = None
    include_platform = False
    for paths in files.values():
        for path in paths:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                header = next(csv.reader(handle))
            if not header or header[0] != "记录时间" or len(header) != len(set(header)):
                raise ValueError(f"Invalid or duplicate CSV header: {path}")
            if "平台到位" in header:
                if header[-1] != "平台到位":
                    raise ValueError(f"Optional platform field must be last: {path}")
                include_platform = True
                header = header[:-1]
            if base is None:
                base = header
            elif header != base:
                raise ValueError(f"Unexpected header variation: {path}")
    if base is None:
        raise ValueError("No CSV headers")
    return base + (["平台到位"] if include_platform else [])


def load_jobs(path: Path, host: str) -> pd.DataFrame:
    raw = pd.read_parquet(path)
    required = {"time", "host", "field", "value_float", "value_string"}
    if not required.issubset(raw.columns):
        raise ValueError(f"Jobs Parquet missing columns {required - set(raw.columns)}: {path}")
    raw = raw.loc[raw["host"].eq(host)]
    records: list[dict] = []
    for start, group in raw.groupby("time", sort=True):
        fields = {}
        for row in group.itertuples(index=False):
            fields[row.field] = (
                row.value_string if pd.notna(row.value_string) else row.value_float
            )
        end_ms = fields.get("EndTime")
        if pd.isna(end_ms):
            continue
        start_utc = pd.Timestamp(start).tz_convert("UTC")
        end_utc = pd.to_datetime(int(end_ms), unit="ms", utc=True)
        if end_utc < start_utc:
            raise ValueError(f"Job ends before it starts in {path}: {start_utc}")
        records.append(
            {
                "device_id": host,
                "job_id": f"{host}_{start_utc.strftime('%Y%m%dT%H%M%S%fZ')}",
                "job_name": str(fields.get("JobName") or ""),
                "batch_no": str(fields.get("BatchNo") or ""),
                "material": str(fields.get("Material") or ""),
                "job_start_utc": start_utc,
                "job_end_utc": end_utc,
            }
        )
    if not records:
        raise ValueError(f"No matching jobs for {host}: {path}")
    return pd.DataFrame.from_records(records)


def match_job(path: Path, host: str, jobs: pd.DataFrame, used_ids: set[str]) -> dict:
    match = FILE_PATTERN.fullmatch(path.name)
    assert match is not None
    filename_time = pd.Timestamp(
        match.group(1) + " " + match.group(2).replace("-", ":"), tz=BEIJING
    ).tz_convert("UTC")
    distances = (jobs["job_start_utc"] - filename_time).abs().dt.total_seconds()
    index = distances.idxmin()
    delta = float(distances.loc[index])
    if delta > 3:
        raise ValueError(f"No Jobs record within 3 seconds of {path.name}; nearest={delta:.3f}s")
    job = jobs.loc[index].to_dict()
    if job["job_id"] in used_ids:
        raise ValueError(f"Multiple CSVs matched the same job: {job['job_id']}")
    used_ids.add(job["job_id"])
    job["job_match_delta_seconds"] = delta
    return job


def sensor_schema(header: list[str]) -> pa.Schema:
    fields = [
        pa.field("time_utc", pa.timestamp("ns", tz="UTC")),
        pa.field("time_beijing", pa.timestamp("ns", tz=BEIJING)),
        pa.field("device_id", pa.string()),
        pa.field("job_id", pa.string()),
        pa.field("source_file", pa.string()),
        pa.field("source_row", pa.int64()),
    ]
    fields.extend(
        pa.field(name, pa.string() if name in DATE_FIELDS else pa.float64())
        for name in header
        if name != "记录时间"
    )
    return pa.schema(fields)


def convert_file(
    path: Path,
    host: str,
    job: dict,
    header_union: list[str],
    schema: pa.Schema,
    writer: pq.ParquetWriter,
    chunk_rows: int,
) -> dict:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle))
    width = len(header)
    numeric_positions = {
        index: "float64"
        for index, name in enumerate(header)
        if index > 0 and name not in DATE_FIELDS
    }
    stats = {
        "device_id": host,
        "job_id": job["job_id"],
        "job_name": job["job_name"],
        "batch_no": job["batch_no"],
        "material": job["material"],
        "job_start_utc": job["job_start_utc"],
        "job_end_utc": job["job_end_utc"],
        "job_start_beijing": job["job_start_utc"].tz_convert(BEIJING),
        "job_end_beijing": job["job_end_utc"].tz_convert(BEIJING),
        "source_file": path.name,
        "csv_rows": 0,
        "csv_header_columns": width,
        "duplicate_timestamps": 0,
        "time_gaps_over_2s": 0,
        "time_gaps_over_10s": 0,
        "max_time_gap_seconds": 0.0,
        "job_match_delta_seconds": job["job_match_delta_seconds"],
    }
    previous_time = None
    reader = pd.read_csv(
        path,
        encoding="utf-8-sig",
        header=None,
        skiprows=1,
        names=range(width),
        index_col=False,
        usecols=range(width),
        dtype=numeric_positions,
        chunksize=chunk_rows,
        low_memory=False,
        on_bad_lines="error",
    )
    for chunk in reader:
        chunk.columns = header
        times = pd.to_datetime(chunk.pop("记录时间"), errors="raise").dt.tz_localize(
            BEIJING
        )
        if times.isna().any():
            raise ValueError(f"Missing timestamps in {path.name}")
        if previous_time is not None:
            first_delta = (times.iloc[0] - previous_time).total_seconds()
            if first_delta < 0:
                raise ValueError(f"Out-of-order timestamps in {path.name}")
            stats["duplicate_timestamps"] += int(first_delta == 0)
            stats["time_gaps_over_2s"] += int(first_delta > 2)
            stats["time_gaps_over_10s"] += int(first_delta > 10)
            stats["max_time_gap_seconds"] = max(stats["max_time_gap_seconds"], first_delta)
        deltas = times.diff().dt.total_seconds().iloc[1:]
        if deltas.lt(0).any():
            raise ValueError(f"Out-of-order timestamps in {path.name}")
        stats["duplicate_timestamps"] += int(deltas.eq(0).sum())
        stats["time_gaps_over_2s"] += int(deltas.gt(2).sum())
        stats["time_gaps_over_10s"] += int(deltas.gt(10).sum())
        if len(deltas):
            stats["max_time_gap_seconds"] = max(
                stats["max_time_gap_seconds"], float(deltas.max())
            )
        previous_time = times.iloc[-1]
        if stats["csv_rows"] == 0:
            stats["csv_start_beijing"] = times.iloc[0]
            stats["csv_start_utc"] = times.iloc[0].tz_convert("UTC")
        stats["csv_end_beijing"] = times.iloc[-1]
        stats["csv_end_utc"] = times.iloc[-1].tz_convert("UTC")

        row_start = stats["csv_rows"] + 1
        stats["csv_rows"] += len(chunk)
        data = {
            "time_utc": times.dt.tz_convert("UTC").to_numpy(),
            "time_beijing": times.to_numpy(),
            "device_id": host,
            "job_id": job["job_id"],
            "source_file": path.name,
            "source_row": range(row_start, row_start + len(chunk)),
        }
        # Assign Series by position, not by the original CSV row index.
        for name in header_union[1:]:
            data[name] = chunk[name].to_numpy() if name in chunk else None
        output = pd.DataFrame(data)
        table = pa.Table.from_pandas(output, schema=schema, preserve_index=False)
        writer.write_table(table)

    if stats["csv_rows"] == 0:
        raise ValueError(f"Empty CSV: {path.name}")
    tolerance = pd.Timedelta(seconds=3)
    if (
        stats["csv_start_utc"] < job["job_start_utc"] - tolerance
        or stats["csv_end_utc"] > job["job_end_utc"] + tolerance
    ):
        raise ValueError(f"CSV time range extends outside matched job: {path.name}")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "more_features_added")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--chunk-rows", type=int, default=50_000)
    args = parser.parse_args()
    if args.chunk_rows < 1:
        parser.error("--chunk-rows must be positive")

    files = collect_csv_files(args.input_dir)
    header_union = read_headers(files)
    schema = sensor_schema(header_union)
    targets = []
    for host in files:
        targets.extend(
            [
                args.data_dir / f"influx_sensors_added_{host}_2026-07.parquet",
                args.data_dir / f"influx_jobs_added_{host}_2026-07.parquet",
            ]
        )
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite existing outputs: " + ", ".join(existing))

    for host, paths in files.items():
        old_jobs_path = args.data_dir / f"influx_jobs_{host}_2026-07.parquet"
        jobs = load_jobs(old_jobs_path, host)
        sensor_path = args.data_dir / f"influx_sensors_added_{host}_2026-07.parquet"
        jobs_path = args.data_dir / f"influx_jobs_added_{host}_2026-07.parquet"
        sensor_temp = sensor_path.with_suffix(".parquet.incomplete")
        jobs_temp = jobs_path.with_suffix(".parquet.incomplete")
        if sensor_temp.exists() or jobs_temp.exists():
            raise FileExistsError(f"Incomplete output already exists for {host}; inspect it first")
        used_ids: set[str] = set()
        job_rows = []
        try:
            with pq.ParquetWriter(sensor_temp, schema=schema, compression="zstd") as writer:
                for path in paths:
                    job = match_job(path, host, jobs, used_ids)
                    stats = convert_file(
                        path, host, job, header_union, schema, writer, args.chunk_rows
                    )
                    job_rows.append(stats)
                    print(f"{host}: {path.name}: {stats['csv_rows']:,} rows", flush=True)
            job_frame = pd.DataFrame.from_records(job_rows).sort_values("job_start_utc")
            pq.write_table(
                pa.Table.from_pandas(job_frame, preserve_index=False),
                jobs_temp,
                compression="zstd",
            )
            sensor_temp.replace(sensor_path)
            jobs_temp.replace(jobs_path)
        except Exception:
            # Only remove temporary files created by this invocation; originals stay intact.
            sensor_temp.unlink(missing_ok=True)
            jobs_temp.unlink(missing_ok=True)
            raise
        print(
            f"DONE {host}: {len(job_rows)} jobs, "
            f"{sum(row['csv_rows'] for row in job_rows):,} sensor rows",
            flush=True,
        )


if __name__ == "__main__":
    main()
