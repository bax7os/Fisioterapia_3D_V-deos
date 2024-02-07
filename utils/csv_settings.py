import pandas as pd

def create_csv(filename = 'output'):

    # Cria um DataFrame com as colunas "Frame" e "Angle"
    csv_file = pd.DataFrame({"Frame": [], "Angle": []})

    # Escreve o cabeçalho do DataFrame no arquivo CSV
    csv_file.to_csv(f"{filename}.csv", index=False)

    return csv_file


def write_csv_ang(csv_file, frame_count, angle, filename='output'):
    """Escreve o frame e o ângulo no DataFrame."""

    # Adiciona uma nova linha ao DataFrame
    csv_file.loc[len(csv_file)] = [frame_count, angle]
    csv_file['Frame'].astype(int)

    # Atualiza o DataFrame no arquivo CSV
    csv_file.to_csv(f"{filename}.csv", index=False)