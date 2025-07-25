import pandas as pd
import statsmodels.api as sm
import tkinter as tk
from tkinter import ttk, scrolledtext

# ----------- Cargar y preparar los datos -----------

ruta_archivo = r"C:\Users\jasmi\OneDrive\Documents\Base de datos Autos para modelo de regresion lineal multiple.xlsx"
df = pd.read_excel(ruta_archivo)

# Limpiar el Dealer Cost
df["Dealer Cost"] = (
    df["Dealer Cost"]
    .astype(str)
    .str.replace("$", "", regex=False)
    .str.replace(",", "", regex=False)
    .str.strip()
    .astype(float)
)

# Modelo de regresión
X = df[["Dealer Cost", "Engine Size", "Cilinders", "Precio gas/galon"]]
Y = df["Costo por milla"]
X = sm.add_constant(X)
modelo = sm.OLS(Y, X).fit()

# Predecir y ordenar
df["Costo por milla estimado"] = modelo.predict(X)
df_ordenado = df.sort_values(by="Costo por milla estimado").reset_index(drop=True)

# Construir fórmula
formula = f"Costo por milla = {modelo.params['const']:.5f}"
for var in ["Dealer Cost", "Engine Size", "Cilinders", "Precio gas/galon"]:
    formula += f" + ({modelo.params[var]:.5f} * {var})"

# ----------- Interfaz Gráfica (tkinter) -----------

root = tk.Tk()
root.title("Eficiencia económica de autos")
root.geometry("800x600")

# Título
titulo = tk.Label(root, text="Autos ordenados por eficiencia económica estimada", font=("Arial", 14, "bold"))
titulo.pack(pady=10)

# Tabla
cols = ["Brand", "Model", "Costo por milla estimado"]
tree = ttk.Treeview(root, columns=cols, show="headings")

for col in cols:
    tree.heading(col, text=col)
    tree.column(col, width=200)

for _, row in df_ordenado.iterrows():
    tree.insert("", "end", values=(row["Brand"], row["Model"], f"{row['Costo por milla estimado']:.5f}"))

tree.pack(expand=True, fill="both", padx=10, pady=10)

# Fórmula
label_formula = tk.Label(root, text="Modelo de regresión lineal múltiple:", font=("Arial", 12, "bold"))
label_formula.pack(pady=(10, 0))

txt_formula = scrolledtext.ScrolledText(root, height=3, wrap=tk.WORD, font=("Courier", 10))
txt_formula.insert(tk.END, formula)
txt_formula.configure(state='disabled')
txt_formula.pack(fill="x", padx=10, pady=5)

# Ejecutar la app
root.mainloop()
