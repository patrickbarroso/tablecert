import os
import math

# Diretórios a validar
dirs = [
    "/ROOT/DATASET/PUBTABLES_YOLO_LIGHT_300K/train/labels",
    "/ROOT/DATASET/PUBTABLES_YOLO_LIGHT_300K/val/labels"
]

output_report = "labels_report.txt"
bad_files_output = "bad_labels.txt"
bad_lines_output = "bad_lines.txt"

def is_bad_value(v):
    """Verifica se valor é NaN, INF ou fora do intervalo permitido."""
    if math.isnan(v) or math.isinf(v):
        return True
    return False

def validar_linha(line):
    """
    Valida uma linha YOLO: class x y w h
    Retorna None se OK, ou string com erro.
    """
    original = line.strip()

    if not original:
        return "Linha vazia"

    parts = original.split()
    if len(parts) != 5:
        return f"Formato inválido (esperado 5 valores, recebeu {len(parts)})"

    try:
        cls = float(parts[0])
        x, y, w, h = map(float, parts[1:])
    except:
        return "Não numérico"

    # Verificar valores inválidos (NaN, inf)
    for v in [cls, x, y, w, h]:
        if is_bad_value(v):
            return "Valor NaN ou INF"

    # Classe negativa
    if cls < 0:
        return "Classe negativa"

    # Coordenadas fora de [0,1]
    if not (0 <= x <= 1):
        return f"x fora do intervalo: {x}"
    if not (0 <= y <= 1):
        return f"y fora do intervalo: {y}"
    if not (0 <= w <= 1):
        return f"w fora do intervalo: {w}"
    if not (0 <= h <= 1):
        return f"h fora do intervalo: {h}"

    # Box inválido
    if w <= 0 or h <= 0:
        return f"Box inválido (w/h <= 0): w={w}, h={h}"

    return None  # Está ok

def validar_diretorio(dir_path):
    """
    Valida todos os arquivos .txt dentro do diretório.
    Retorna (lista_de_erros_por_arquivo, total_ok, total_bad)
    """
    erros = {}
    total_ok = 0
    total_bad = 0

    for fname in os.listdir(dir_path):
        if not fname.endswith(".txt"):
            continue

        full_path = os.path.join(dir_path, fname)
        with open(full_path, "r") as f:
            lines = f.readlines()

        if len(lines) == 0:
            erros[fname] = ["Arquivo sem labels"]
            total_bad += 1
            continue

        file_errors = []
        for line in lines:
            erro = validar_linha(line)
            if erro:
                file_errors.append(f"{erro}  -->  '{line.strip()}'")

        if file_errors:
            erros[fname] = file_errors
            total_bad += 1
        else:
            total_ok += 1

    return erros, total_ok, total_bad


# ===========================
# PROCESSAR TODOS OS DIRETÓRIOS
# ===========================

report = []
bad_files = []
bad_lines = []

for d in dirs:
    report.append(f"\n\n============================================")
    report.append(f"VALIDANDO DIRETÓRIO: {d}")
    report.append(f"============================================")

    erros, ok, bad = validar_diretorio(d)

    report.append(f"Arquivos OK    : {ok}")
    report.append(f"Arquivos com erro: {bad}")

    if erros:
        report.append("\n--- DETALHES DOS ARQUIVOS COM ERROS ---")
        for fname, lst in erros.items():
            full_path = os.path.join(d, fname)
            bad_files.append(full_path)

            report.append(f"\nArquivo: {full_path}")
            for e in lst:
                report.append("   " + e)
                bad_lines.append(f"{full_path}: {e}")

# ===========================
# SALVAR RELATÓRIOS
# ===========================

with open(output_report, "w") as f:
    f.write("\n".join(report))

with open(bad_files_output, "w") as f:
    f.write("\n".join(bad_files))

with open(bad_lines_output, "w") as f:
    f.write("\n".join(bad_lines))

print("\n=======================================================")
print("VALIDAÇÃO CONCLUÍDA!")
print("=======================================================\n")
print(f"Relatório completo salvo em: {output_report}")
print(f"Lista de arquivos inválidos: {bad_files_output}")
print(f"Lista detalhada de linhas ruins: {bad_lines_output}\n")
print("Pronto para análise!")
