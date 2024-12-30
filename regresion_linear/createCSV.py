import pandas as pd

# Manually cleaning the extracted text into structured data
data = {
    "Reduccion de solidos": [3, 7, 11, 15, 18, 27, 29, 30, 30, 31, 31, 32, 33, 33, 34, 36, 36, 36, 37, 38, 39, 39, 39, 40, 41, 42, 42, 43, 44, 45, 46, 47, 50],
    "Reduccion demanda de oxigeno": [5, 11, 21, 16, 16, 28, 27, 25, 35, 30, 40, 32, 34, 32, 34, 37, 38, 34, 36, 38, 37, 36, 45, 39, 41, 40, 44, 37, 44, 46, 46, 49, 51],
}

# Create DataFrame
df = pd.DataFrame(data)

# Save as CSV
csv_path = "./reduccion_datos.csv"
df.to_csv(csv_path, index=False)

csv_path
