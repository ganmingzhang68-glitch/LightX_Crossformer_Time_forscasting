from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


CADENCE_SECONDS = 2
LONG_GAP_SECONDS = 10


def sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def sql_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def build_wide(
    source: Path,
    destination: Path,
    features: list[str],
    boolean_features: list[str],
) -> None:
    expressions = []
    for feature in features:
        expressions.append(
            "CAST(max(CASE WHEN field = "
            + sql_string(feature)
            + " THEN CASE WHEN value_type = 'bool' "
            + "THEN CAST(value_bool AS DOUBLE) ELSE value_float END END) AS FLOAT) AS "
            + sql_identifier(feature)
        )
    fields = ", ".join(sql_string(feature) for feature in features)
    query = f"""
        COPY (
            SELECT time, {', '.join(expressions)}
            FROM read_parquet({sql_string(str(source))})
            WHERE measurement = 'PLC-Sensors' AND field IN ({fields})
            GROUP BY time ORDER BY time
        ) TO {sql_string(str(destination))} (FORMAT PARQUET, COMPRESSION ZSTD)
    """
    connection = duckdb.connect()
    try:
        columns = {
            row[0] for row in connection.execute(
                f"DESCRIBE SELECT * FROM read_parquet({sql_string(str(source))})"
            ).fetchall()
        }
        required = {"time", "measurement", "field", "value_type", "value_bool", "value_float"}
        missing_columns = sorted(required - columns)
        if missing_columns:
            raise ValueError(f"Influx long-format Parquet is missing columns: {missing_columns}")
        available = {
            str(row[0]) for row in connection.execute(
                f"SELECT DISTINCT field FROM read_parquet({sql_string(str(source))}) "
                "WHERE measurement = 'PLC-Sensors'"
            ).fetchall()
        }
        missing_features = [feature for feature in features if feature not in available]
        if missing_features:
            raise ValueError(
                f"External device is missing {len(missing_features)} required features: "
                f"{missing_features}"
            )
        type_rows = connection.execute(
            f"SELECT DISTINCT field, value_type "
            f"FROM read_parquet({sql_string(str(source))}) "
            f"WHERE measurement = 'PLC-Sensors' AND field IN ({fields})"
        ).fetchall()
        types_by_feature: dict[str, set[str]] = {feature: set() for feature in features}
        for field, value_type in type_rows:
            types_by_feature[str(field)].add(str(value_type))
        mixed_types = {
            feature: sorted(types)
            for feature, types in types_by_feature.items()
            if len(types) != 1
        }
        boolean_set = set(boolean_features)
        wrong_boolean = {
            feature: sorted(types_by_feature[feature])
            for feature in boolean_features
            if types_by_feature[feature] != {"bool"}
        }
        wrong_continuous = {
            feature: sorted(types_by_feature[feature])
            for feature in features
            if feature not in boolean_set and "bool" in types_by_feature[feature]
        }
        if mixed_types or wrong_boolean or wrong_continuous:
            raise ValueError(
                "External feature types do not match the training contract; "
                f"mixed={mixed_types}, boolean={wrong_boolean}, "
                f"continuous={wrong_continuous}"
            )
        connection.execute("SET threads TO 4")
        connection.execute(query)
    finally:
        connection.close()


def apply_quality_rules(frame: pd.DataFrame, rules: dict, features: list[str]) -> int:
    unknown = sorted(set(rules) - set(features))
    if unknown:
        raise ValueError(f"Quality rules reference unknown features: {unknown}")
    invalidated = 0
    for feature, rule in rules.items():
        invalid = pd.Series(False, index=frame.index)
        for value in rule.get("invalid_values", []):
            invalid |= frame[feature].eq(float(value))
        if rule.get("valid_min") is not None:
            invalid |= frame[feature].lt(float(rule["valid_min"]))
        if rule.get("valid_max") is not None:
            invalid |= frame[feature].gt(float(rule["valid_max"]))
        invalidated += int(invalid.sum())
        frame.loc[invalid, feature] = np.nan
    return invalidated


def regularize(
    wide: pd.DataFrame,
    features: list[str],
    boolean_features: list[str],
) -> tuple[pd.DataFrame, list[dict]]:
    wide = wide.sort_values("time", kind="stable").drop_duplicates("time", keep="last")
    wide["time"] = pd.to_datetime(wide["time"], utc=True)
    segment_labels = wide["time"].diff().dt.total_seconds().gt(LONG_GAP_SECONDS).cumsum()
    continuous = [feature for feature in features if feature not in set(boolean_features)]
    outputs: list[pd.DataFrame] = []
    summaries: list[dict] = []
    segment_id = 0
    for _, group in wide.groupby(segment_labels, sort=True):
        group = group.set_index("time").sort_index()
        grid = pd.date_range(group.index[0], group.index[-1], freq="2s", tz="UTC")
        regular = group.reindex(grid)
        observed = regular[features].notna().all(axis=1)
        regular[continuous] = regular[continuous].interpolate(
            method="linear", limit=1, limit_area="inside"
        )
        regular[boolean_features] = regular[boolean_features].ffill(limit=1)
        valid = ~regular[features].isna().any(axis=1)
        split_labels = (~valid | ~valid.shift(fill_value=False)).cumsum()
        for _, valid_group in regular.loc[valid].groupby(split_labels.loc[valid]):
            if valid_group.empty:
                continue
            frame = valid_group.reset_index(names="time_utc")
            frame.insert(1, "time_beijing", frame["time_utc"].dt.tz_convert("Asia/Shanghai"))
            frame.insert(2, "segment_id", np.int32(segment_id))
            frame.insert(
                3,
                "observed_mask",
                observed.reindex(valid_group.index).fillna(False).to_numpy(dtype=bool),
            )
            for feature in continuous:
                frame[feature] = frame[feature].astype("float32")
            for feature in boolean_features:
                frame[feature] = frame[feature].round().clip(0, 1).astype("float32")
            outputs.append(frame)
            summaries.append({
                "segment_id": segment_id,
                "start_time_utc": frame["time_utc"].iloc[0].isoformat(),
                "end_time_utc": frame["time_utc"].iloc[-1].isoformat(),
                "rows": len(frame),
                "inserted_rows": int((~frame["observed_mask"]).sum()),
            })
            segment_id += 1
    if not outputs:
        raise RuntimeError("No complete 2-second segments remain after preprocessing")
    return pd.concat(outputs, ignore_index=True), summaries


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Convert one external device's Influx long-format Parquet to the model contract."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=root / "artifacts/feature_manifest.json")
    parser.add_argument("--quality-rules", type=Path, default=root / "config/plc_feature_quality_rules.json")
    args = parser.parse_args()
    source = args.input.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    features = list(manifest["feature_order"])
    boolean_features = list(manifest["boolean_features"])
    quality_rules = json.loads(args.quality_rules.read_text(encoding="utf-8")).get("features", {})
    with tempfile.TemporaryDirectory(prefix="lightx_crossformer_") as temporary:
        wide_path = Path(temporary) / "wide.parquet"
        build_wide(source, wide_path, features, boolean_features)
        wide = pd.read_parquet(wide_path)
    invalidated = apply_quality_rules(wide, quality_rules, features)
    prepared, segments = regularize(wide, features, boolean_features)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.lower() == ".csv":
        prepared.to_csv(args.output, index=False)
    else:
        prepared.to_parquet(args.output, index=False, compression="zstd")
    print(json.dumps({
        "output": str(args.output.resolve()),
        "feature_count": len(features),
        "rows": len(prepared),
        "segments": segments,
        "quality_rule_invalidated_points": invalidated,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
