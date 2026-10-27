# -*- coding: utf-8 -*-
"""
Porto 数据抽样预处理 —— TrajRL 复现最小验证流水线

输入: 原始 Porto train.csv (Kaggle / UCI)
      列: TRIP_ID, CALL_TYPE, ORIGIN_CALL, ORIGIN_STAND, TAXI_ID,
          TIMESTAMP, DAY_TYPE, MISSING_DATA, POLYLINE
流程: 流式读取全量 -> 过滤(空POLYLINE / 点数<2) -> 水塘抽样 k 条 ->
      解析 POLYLINE 为点序列, 计算几何/时间特征 ->
      输出 sample.jsonl + summary.json (校验依据)

口径 (对齐 docs/paper_extracted_facts.md §9 porto.json):
  * seq_len=128 是论文轨迹长度上限: 超过 128 的轨迹保留全部点并标记
    over_seq_len=True, 截断/丢弃由训练侧决定, 预处理不做静默截断。
  * delta_t: 原始 CSV 只有起点 TIMESTAMP, 无逐点时间戳, 因此按 Porto
    平均 15 秒采样间隔均匀推断, 输出列名为 delta_t_inferred,
    明确不是论文口径的时间间隔 (论文用 map-matching 路段通过时间,
    需 libcity 路网匹配后获得)。一切以此推断值做的实验都是近似验证。
  * 过滤规则: POLYLINE 为空、或解析后点数 < 2 的行丢弃。
  * MISSING_DATA=True 的行保留但标记, 作为下游可选筛选依据。

用法:
  python porto_preprocess.py \
      --input "C:/Users/PRLOUIS/Desktop/AuthorKit27/TrajRL_reproduction/train/train(1).csv" \
      --outdir "C:/Users/PRLOUIS/Desktop/trajrl_paper_repro_workdir/outputs/porto_sample1w" \
      --sample 10000 --seed 42
"""

import argparse
import csv
import json
import math
import os
import random
import sys
import time


EARTH_R = 6371000.0  # meters


def haversine_m(lon1, lat1, lon2, lat2):
    """Haversine distance in meters between two lon/lat points."""
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    return 2.0 * EARTH_R * math.asin(math.sqrt(a))


def parse_polyline(raw):
    """Parse POLYLINE string into list of (lon, lat) points; None on failure."""
    if not raw or raw == "[]":
        return None
    try:
        pts = json.loads(raw)
    except (ValueError, TypeError):
        return None
    if not isinstance(pts, list) or len(pts) == 0:
        return None
    out = []
    for p in pts:
        if not isinstance(p, (list, tuple)) or len(p) < 2:
            return None
        lon, lat = float(p[0]), float(p[1])
        if math.isnan(lon) or math.isnan(lat):
            return None
        out.append((lon, lat))
    return out


def build_record(row, pts, seq_cap, interval_s):
    """Build the JSON-serialisable record for one sampled trajectory."""
    n = len(pts)
    delta_s = []
    for i in range(1, n):
        d = haversine_m(pts[i - 1][0], pts[i - 1][1], pts[i][0], pts[i][1])
        delta_s.append(round(d, 2))
    cum = float(sum(delta_s))
    delta_t = [interval_s] * (n - 1)
    return {
        "trip_id": row["TRIP_ID"],
        "taxi_id": row["TAXI_ID"],
        "call_type": row["CALL_TYPE"],
        "day_type": row["DAY_TYPE"],
        "timestamp": int(row["TIMESTAMP"]),
        "missing_data": row["MISSING_DATA"].strip().lower() == "true",
        "n_points": n,
        "over_seq_len": n > seq_cap,
        "points": [[round(p[0], 6), round(p[1], 6)] for p in pts],
        "delta_s": delta_s,                    # meters per segment
        "cum_dist": round(cum, 2),             # total meters
        "delta_t_inferred": delta_t,           # seconds, UNIFORM INFERENCE
        "elapsed_inferred": interval_s * (n - 1),  # seconds, INFERENCE ONLY
    }


def main():
    ap = argparse.ArgumentParser(description="Porto sampling preprocess (minimal pipeline)")
    ap.add_argument("--input", required=True, help="path to Porto train(1).csv")
    ap.add_argument("--outdir", required=True, help="output directory (created if needed)")
    ap.add_argument("--sample", type=int, default=10000, help="reservoir sample size")
    ap.add_argument("--seed", type=int, default=42, help="random seed")
    ap.add_argument("--seq-cap", type=int, default=128, help="paper seq_len cap (porto.json)")
    ap.add_argument("--interval-s", type=float, default=15.0, help="assumed uniform sampling interval (sec)")
    args = ap.parse_args()

    t0 = time.time()
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    rng = random.Random(args.seed)
    reservoir = []  # list of (row, pts) kept with probability args.sample/kept_count
    stats = {
        "total_rows_read": 0,
        "dropped_empty_polyline": 0,
        "dropped_lt2_points": 0,
        "dropped_unparseable": 0,
        "kept_valid": 0,
        "missing_data_true_total": 0,
    }

    with open(args.input, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stats["total_rows_read"] += 1
            pts = parse_polyline(row.get("POLYLINE", ""))
            if pts is None:
                stats["dropped_empty_polyline"] += 1
                continue
            if len(pts) < 2:
                stats["dropped_lt2_points"] += 1
                continue
            stats["kept_valid"] += 1
            if row.get("MISSING_DATA", "").strip().lower() == "true":
                stats["missing_data_true_total"] += 1
            # reservoir sampling over valid trajectories
            if len(reservoir) < args.sample:
                reservoir.append((row, pts))
            else:
                j = rng.randint(0, stats["kept_valid"] - 1)
                if j < args.sample:
                    reservoir[j] = (row, pts)

    # dump sample
    sample_path = os.path.join(outdir, "sample.jsonl")
    records = []
    over_count = 0
    missing_in_sample = 0
    npts = []
    cum_dists = []
    with open(sample_path, "w", encoding="utf-8") as out:
        for row, pts in reservoir:
            rec = build_record(row, pts, args.seq_cap, args.interval_s)
            records.append(rec)
            if rec["over_seq_len"]:
                over_count += 1
            if rec["missing_data"]:
                missing_in_sample += 1
            npts.append(rec["n_points"])
            cum_dists.append(rec["cum_dist"])
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    npts_sorted = sorted(npts)
    cum_sorted = sorted(cum_dists)

    def pct(xs, q):
        if not xs:
            return 0.0
        idx = min(len(xs) - 1, int(round(len(xs) * q)))
        return xs[idx]

    summary = {
        "input": args.input,
        "outdir": outdir,
        "seq_cap": args.seq_cap,
        "interval_s": args.interval_s,
        "seed": args.seed,
        "sample_size": len(records),
        "stats": stats,
        "sample_stats": {
            "over_seq_len_count": over_count,
            "missing_data_true_in_sample": missing_in_sample,
            "n_points": {
                "min": npts_sorted[0] if npts_sorted else None,
                "p50": pct(npts_sorted, 0.5),
                "p90": pct(npts_sorted, 0.9),
                "max": npts_sorted[-1] if npts_sorted else None,
                "mean": round(sum(npts) / len(npts), 2) if npts else None,
            },
            "cum_dist_m": {
                "min": cum_sorted[0] if cum_sorted else None,
                "p50": pct(cum_sorted, 0.5),
                "p90": pct(cum_sorted, 0.9),
                "max": cum_sorted[-1] if cum_sorted else None,
                "mean": round(sum(cum_dists) / len(cum_dists), 2) if cum_dists else None,
            },
            "delta_t_inferred": "uniform " + str(args.interval_s) + "s (INFERENCE, not paper ground truth)",
        },
        "elapsed_s": round(time.time() - t0, 1),
    }
    summary_path = os.path.join(outdir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as out:
        json.dump(summary, out, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("sample file:", sample_path)
    print("summary file:", summary_path)


if __name__ == "__main__":
    sys.exit(main())