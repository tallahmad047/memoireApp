import pandas as pd

# Fonction qui prend un DataFrame en paramètre
def ajouter_donnees_au_csv(data, csv_path):
    # Charger le fichier CSV existant dans un DataFrame
    df = pd.read_csv(csv_path)

    # Créer une nouvelle ligne avec les données que vous avez
    new_row = pd.DataFrame([data], columns=df.columns)

    # Ajouter la nouvelle ligne au DataFrame existant
    df = df.append(new_row, ignore_index=True)

    # Sauvegarder le DataFrame mis à jour dans le même fichier CSV
    df.to_csv(csv_path, index=False)
    return df


import pandas as pd

def calculate_bmi(df):
    # Calculer l'indice de masse corporelle (BMI)
    df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)

    # Obtenir les valeurs minimales et maximales de BMI
    bmi_min = int(df['bmi'].min())
    bmi_max = int(df['bmi'].max())

    # Discrétiser le BMI en 6 catégories
    df['bmi'] = pd.cut(df['bmi'], bins=6, labels=range(6), right=True, include_lowest=True)

    # Obtenir la distribution des catégories de BMI
    bmi= df["bmi"].value_counts(normalize=True)

    return bmi_min, bmi_max, bmi


