import json
import pandas as pd
import unicodedata

# Carregar os dados JSON
data = pd.read_json("./data/padaria_trab.json")

# Função para normalizar a acentuação e extrair a primeira palavra
def process_product(product):
    # Normalizar para corrigir acentuação
    product = unicodedata.normalize('NFKD', product).encode('ascii', 'ignore').decode('utf-8')
    # Extrair a primeira palavra
    #first_word = product.split()[0]
    word = product
    #if first_word == 'Doce':
    #    return 'Doce'
    return word

# Iterar sobre as linhas do DataFrame e processar os produtos
for index, row in data.iterrows():
    produtos = row["produtos"]
    produtos_processados = [process_product(produto) for produto in produtos]
    data.at[index, "produtos"] = produtos_processados

# Resultado final
print(json.dumps(data.to_dict(orient='records'), indent=4, ensure_ascii=False))
