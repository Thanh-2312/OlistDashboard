"""Ghép _build/nb/N*.py (định dạng '# %%') thành notebooks/*.ipynb rồi chạy để lưu sẵn kết quả.

Dùng: python _build/make_notebooks.py N1 N2      (không truyền gì = chạy cả N1..N7; N6 cần N5, N7 cần N6)
"""
import re
import sys
import time
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent.parent          # BTL_Cuoi_Ki
SRC = ROOT / '_build' / 'nb'
OUT = ROOT / 'notebooks'
NAMES = {'N1': 'N1_Khao_sat_he_thong_du_lieu', 'N2': 'N2_Kham_pha_du_lieu',
         'N3': 'N3_ETL_SQLite_DW', 'N4': 'N4_Chu_de_mo_rong', 'N5': 'N5_Lakehouse',
         'N6': 'N6_Lakehouse_ban_cau_truc_khong_cau_truc', 'N7': 'N7_AI_khai_thac_du_lieu'}


def build(key):
    src = (SRC / f'{key}.py').read_text(encoding='utf-8')
    nb = nbf.v4.new_notebook()
    for chunk in re.split(r'(?m)^# %%', src)[1:]:
        head, _, body = chunk.partition('\n')
        body = body.strip('\n')
        if '[markdown]' in head:
            md = '\n'.join(l[2:] if l.startswith('# ') else l.lstrip('#') for l in body.split('\n'))
            nb.cells.append(nbf.v4.new_markdown_cell(md))
        else:
            nb.cells.append(nbf.v4.new_code_cell(body))
    nb.metadata['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
    nb.metadata['language_info'] = {'name': 'python'}
    t0 = time.time()
    NotebookClient(nb, timeout=3600, kernel_name='python3', resources={'metadata': {'path': str(OUT)}}).execute()
    nbf.write(nb, OUT / f'{NAMES[key]}.ipynb')
    print(f'{key}: {len(nb.cells)} ô, chạy {time.time() - t0:.0f}s -> {NAMES[key]}.ipynb', flush=True)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    for k in (sys.argv[1:] or list(NAMES)):
        build(k)
