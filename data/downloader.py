"""
Tự động tải dataset NF-ToN-IoT-v2 và NF-UQ-NIDS-v2 nếu chưa có.

Nguồn:
- nf_ton_iot : HuggingFace (Nora9029/NF-ToN-IoT-v2  hoặc  Hmehdi515/NF-ToN-IoT-v2.csv)
- nf_uq_nids : HuggingFace (mginotx/NF-UQ-NIDS-v2)  /  UQ eSpace (bắt buộc đăng nhập)
"""
import os
import sys
import urllib.request
import urllib.error

HF_BASE = 'https://huggingface.co/datasets'

# === Cấu hình các nguồn tải ===
DATASET_CONFIG = {
    'nf_ton_iot': {
        'output_filename': 'NF-ToN-IoT-v2.csv',
        'description': 'NF-ToN-IoT-v2 (10 classes, ~5 GB CSV)',
        'sources': [
            # Nguồn 1: Nora9029 - train + test split
            {
                'type': 'hf_multi',
                'repo': 'Nora9029/NF-ToN-IoT-v2',
                'files': ['NF-ToN-IoT-v2-train.csv', 'NF-ToN-IoT-v2-test.csv']
            },
            # Nguồn 2: Hmehdi515 - train + val + test split
            {
                'type': 'hf_multi',
                'repo': 'Hmehdi515/NF-ToN-IoT-v2.csv',
                'files': ['train.csv', 'val.csv', 'test.csv']
            },
        ]
    },
    'nf_uq_nids': {
        'output_filename': 'NF-UQ-NIDS-v2.csv',
        'description': 'NF-UQ-NIDS-v2 (21 classes, ~13 GB CSV)',
        'sources': [
            # Nguồn 1: UQ eSpace direct (file lớn, có thể cần đăng nhập)
            {
                'type': 'direct',
                'url': (
                    'https://espace.library.uq.edu.au/data/UQ:0d13491/'
                    'NF-UQ-NIDS-v2.csv?dsi_version=c3c84e81697bdabb09d80abba5e6df6a'
                )
            },
        ]
    }
}


def _download_single_file(url, dest_path, label=''):
    """Tải 1 file từ URL về dest_path. Trả về True nếu thành công."""
    headers = {'User-Agent': 'Mozilla/5.0 (compatible; HFIN-NIDS/1.0)'}
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=60) as resp, \
             open(dest_path, 'wb') as out:
            total = int(resp.headers.get('Content-Length', 0))
            downloaded = 0
            block = 65536
            while True:
                chunk = resp.read(block)
                if not chunk:
                    break
                out.write(chunk)
                downloaded += len(chunk)
                mb = downloaded / 1_048_576
                if total:
                    pct = downloaded * 100 // total
                    tmb = total / 1_048_576
                    sys.stdout.write(f'\r  {label}: {mb:.1f}/{tmb:.1f} MB ({pct}%)  ')
                else:
                    sys.stdout.write(f'\r  {label}: {mb:.1f} MB  ')
                sys.stdout.flush()
        print()
        return True
    except Exception as e:
        print(f'\n  [✗] {label}: {e}')
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False


def _download_hf_multi(repo, files, dest_dir, output_path):
    """Tải nhiều file từ HuggingFace rồi ghép lại."""
    import pandas as pd

    tmp_parts = []
    for fname in files:
        url = f'{HF_BASE}/{repo}/resolve/main/{fname}'
        tmp = os.path.join(dest_dir, f'_tmp_{fname}')
        print(f'  → {fname}')
        if not _download_single_file(url, tmp, label=fname):
            # Xóa các phần đã tải
            for p in tmp_parts:
                if os.path.exists(p):
                    os.remove(p)
            return False
        tmp_parts.append(tmp)

    # Ghép file
    try:
        print(f'\n  Đang ghép {len(tmp_parts)} file...')
        dfs = []
        for p in tmp_parts:
            print(f'    Đọc {os.path.basename(p)}...')
            dfs.append(pd.read_csv(p))
        merged = pd.concat(dfs, ignore_index=True)
        merged.to_csv(output_path, index=False)
        mb = os.path.getsize(output_path) / 1_048_576
        print(f'  [✓] Đã ghép → {output_path} ({mb:.1f} MB)')
        return True
    except Exception as e:
        print(f'  [✗] Lỗi khi ghép: {e}')
        return False
    finally:
        for p in tmp_parts:
            if os.path.exists(p):
                os.remove(p)


def download_dataset(dataset_name, dest_dir='./data/raw/', force=False):
    """
    Tải dataset nếu chưa có.

    Args:
        dataset_name : 'nf_ton_iot' | 'nf_uq_nids'
        dest_dir     : Thư mục lưu file
        force        : Tải lại dù đã có

    Returns:
        str | None: Đường dẫn file CSV hoặc None nếu thất bại
    """
    if dataset_name not in DATASET_CONFIG:
        raise ValueError(f'Dataset không hợp lệ: "{dataset_name}". '
                         f'Chọn: {list(DATASET_CONFIG.keys())}')

    os.makedirs(dest_dir, exist_ok=True)
    cfg = DATASET_CONFIG[dataset_name]
    out_path = os.path.join(dest_dir, cfg['output_filename'])

    if os.path.exists(out_path) and not force:
        mb = os.path.getsize(out_path) / 1_048_576
        print(f'[INFO] Dataset đã tồn tại: {out_path} ({mb:.1f} MB)')
        return out_path

    print(f'\n[DOWNLOAD] {cfg["description"]}')
    print(f'  Đích: {out_path}\n')

    for src in cfg['sources']:
        stype = src['type']

        if stype == 'hf_multi':
            print(f'  Nguồn: HuggingFace → {src["repo"]}')
            try:
                success = _download_hf_multi(
                    src['repo'], src['files'], dest_dir, out_path
                )
            except ImportError:
                print('  [!] Cần pandas: pip install pandas')
                success = False
            if success:
                return out_path

        elif stype == 'direct':
            print(f'  Nguồn: Direct URL')
            tmp = out_path + '.tmp'
            if _download_single_file(src['url'], tmp, label=cfg['output_filename']):
                os.rename(tmp, out_path)
                mb = os.path.getsize(out_path) / 1_048_576
                print(f'  [✓] Đã lưu: {out_path} ({mb:.1f} MB)')
                return out_path
            # Nếu thất bại vì cần đăng nhập (403)
            print('  [!] Có thể cần đăng nhập UQ eSpace. Xem hướng dẫn bên dưới.')

    # --- Không tải được tự động ---
    print(f'\n[✗] Không thể tải tự động dataset "{dataset_name}".')
    if dataset_name == 'nf_uq_nids':
        print('\n  NF-UQ-NIDS-v2 cần tải thủ công từ UQ eSpace:')
        print('  1. Truy cập: https://espace.library.uq.edu.au/view/UQ:0d13491')
        print('  2. Đăng ký/đăng nhập tài khoản UQ')
        print(f'  3. Tải file NF-UQ-NIDS-v2.csv và đặt vào: {dest_dir}')
    else:
        print(f'\n  Tải thủ công từ: https://huggingface.co/datasets/Nora9029/NF-ToN-IoT-v2')
        print(f'  Đặt file vào: {dest_dir}')
    return None


def ensure_dataset(dataset_name, data_path='./data/raw/'):
    """
    Đảm bảo dataset có sẵn. Tự động tải nếu thiếu.
    Trả về đường dẫn đến file cụ thể.
    """
    if dataset_name not in DATASET_CONFIG:
        raise ValueError(f"Unknown dataset: {dataset_name}")
        
    cfg = DATASET_CONFIG[dataset_name]
    target_file = cfg['output_filename']
    
    # Nếu data_path là file cụ thể
    if os.path.isfile(data_path):
        return data_path

    # Nếu data_path là thư mục, kiểm tra file mục tiêu trong đó
    if os.path.isdir(data_path):
        specific_path = os.path.join(data_path, target_file)
        if os.path.exists(specific_path):
            return specific_path
            
        # Kiểm tra nếu có file parquet tương ứng (nếu người dùng tự convert)
        parquet_path = specific_path.replace('.csv', '.parquet')
        if os.path.exists(parquet_path):
            return parquet_path

    print(f'[INFO] Không tìm thấy dataset "{dataset_name}" tại: {data_path}')
    dest = download_dataset(dataset_name, dest_dir=data_path)
    if dest and os.path.exists(dest):
        return dest
        
    raise FileNotFoundError(
        f'\n[✗] Không tìm thấy và không thể tải dataset "{dataset_name}".\n'
        f'    Vui lòng tải thủ công và đặt vào: {data_path}'
    )


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='HFIN Dataset Downloader')
    p.add_argument('--dataset', type=str, default='nf_ton_iot',
                   choices=list(DATASET_CONFIG.keys()))
    p.add_argument('--dest', type=str, default='./data/raw/')
    p.add_argument('--force', action='store_true')
    args = p.parse_args()

    path = download_dataset(args.dataset, args.dest, force=args.force)
    sys.exit(0 if path else 1)
