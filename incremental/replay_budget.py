"""
Replay theo TUNG LOP (--replay_class_ratio), cung dinh nghia voi cac bai khac (vd AFSIC-IOV:
current_m = int(so_mau_cua_lop * memory_ratio)). Moi edge = 1 client; moi lop tai edge do giu
max(1, int(n_lop_tai_edge * ratio)) mau, KHONG phai 1% cua toan bo du lieu cho moi edge.

Hai ham o day phuc vu RESUME tu checkpoint co bo nho theo luat cu (gan 100% du lieu lop cu):
  edge_class_counts : dem so mau tung lop o tung edge (doc nhan cac shard cua task da xong), co cache JSON
  trim_memories     : cat bo nho da khoi phuc xuong ratio, lay NGAU NHIEN co seed (khong phai herding)
"""
import json
import os
import numpy as np
import torch


def edge_class_counts(data_path, edge_client_map, task_classes_seq, tasks, load_fn,
                      cache_path=None, tag=""):
    """Tra ve {edge_id: {class: n}} cho cac lop thuoc `tasks` (danh sach task_id da xong)."""
    key = {"tag": tag, "tasks": list(tasks)}
    if cache_path and os.path.exists(cache_path):
        try:
            blob = json.load(open(cache_path))
            if blob.get("key") == key:
                return {int(e): {int(c): int(n) for c, n in d.items()} for e, d in blob["counts"].items()}
        except Exception:
            pass
    counts = {int(e): {} for e in edge_client_map}
    for t in tasks:
        classes = set(int(c) for c in task_classes_seq[t])
        for e, cids in edge_client_map.items():
            for cid in cids:
                X, y = load_fn(data_path, t, cid)
                if y is None:
                    continue
                yy = y if torch.is_tensor(y) else torch.as_tensor(y)
                for c in classes:
                    n = int((yy == c).sum().item())
                    if n:
                        counts[int(e)][c] = counts[int(e)].get(c, 0) + n
                del X, y
        print(f"[REPLAY-CLASS] Da dem so mau theo lop cho task {t}", flush=True)
    if cache_path:
        try:
            json.dump({"key": key, "counts": {str(e): {str(c): n for c, n in d.items()} for e, d in counts.items()}},
                      open(cache_path, "w"))
        except Exception:
            pass
    return counts


def trim_memories(edge_servers, counts, ratio, logger=None):
    """Cat bo nho moi lop xuong max(1, int(n_lop * ratio)); giu nguyen neu da nho hon."""
    before = after = 0
    for edge in edge_servers:
        em = edge.exemplar_manager
        for i, lab in enumerate(em.exemplar_labels):
            arr = em.exemplar_set[i]
            before += len(arr)
            n_cls = counts.get(edge.edge_id, {}).get(int(lab), len(arr))
            target = max(1, int(n_cls * ratio))
            if len(arr) > target:
                rng = np.random.default_rng(1000 * (edge.edge_id + 1) + int(lab))
                idx = np.sort(rng.choice(len(arr), size=target, replace=False))
                em.exemplar_set[i] = np.ascontiguousarray(arr[idx])
            after += len(em.exemplar_set[i])
    msg = (f"[REPLAY-CLASS] Cat bo nho da khoi phuc theo ratio={ratio}: "
           f"{before:,} -> {after:,} mau tren {len(edge_servers)} edge")
    (logger.info(msg) if logger else print(msg, flush=True))
    return before, after
