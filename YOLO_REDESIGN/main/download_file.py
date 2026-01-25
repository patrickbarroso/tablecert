#!/usr/bin/env python3
"""
download_tar.py
Baixa um arquivo .tar (ou .tar.gz) de uma URL e salva em um diretório especificado.

Uso:
    python download_tar.py <URL> <diretorio_destino>
Exemplo:
    python download_tar.py https://example.com/data.tar.gz /home/usuario/downloads
"""

import os
import sys
import requests
from pathlib import Path
import tarfile

def download_tar(url: str, output_dir: str):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = url.split("/")[-1] or "arquivo.tar"
    filepath = output_dir / filename

    print(f"📥 Baixando: {url}")
    print(f"📂 Salvando em: {filepath}")

    # stream=True baixa em blocos (evita carregar tudo na memória)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get("content-length", 0))
        chunk_size = 8192
        downloaded = 0

        with open(filepath, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    done = int(50 * downloaded / total_size) if total_size else 0
                    sys.stdout.write(f"\rProgresso: [{'=' * done}{' ' * (50-done)}] {downloaded/1024/1024:.2f} MB")
                    sys.stdout.flush()
    print("\n✅ Download concluído!")

    # Verifica se é um tar e extrai
    if tarfile.is_tarfile(filepath):
        print("📦 Extraindo conteúdo...")
        with tarfile.open(filepath, "r:*") as tar:
            tar.extractall(path=output_dir)
        print("✅ Extração concluída!")
    else:
        print("⚠️ O arquivo baixado não parece ser um .tar válido.")

    return filepath


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Uso: python download_tar.py <URL> <diretorio_destino>")
        sys.exit(1)

    url = sys.argv[1]
    output_dir = sys.argv[2]
    download_tar(url, output_dir)

