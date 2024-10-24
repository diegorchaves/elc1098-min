import numpy as np 
import pandas as pd 
import json
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder 

json_data = pd.read_json("./padaria-ajeitada.json")
json_str = json_data.to_json(orient='records')
# print(json_data)

# Carregar os dados do JSON
data = json.loads(json_str)

# Extrair apenas a lista de produtos
transactions = [item['produtos'] for item in data]

# print(transactions)

# Utilizando TransactionEncoder para transformar os dados
encoder = TransactionEncoder()
encoded_array = encoder.fit(transactions).transform(transactions)

# print(encoded_array)

# Criar um DataFrame
df = pd.DataFrame(encoded_array, columns=encoder.columns_)
# print(df)

# Exemplo de uso do algoritmo Apriori
frequent_itemsets = apriori(df, min_support=0.1, use_colnames=True)
print(frequent_itemsets)

# Gerar regras de associação
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.3)

# print(rules)

# Filtrar regras 1 para 1
rules_1_to_1 = rules[(rules['antecedents'].apply(lambda x: len(x) == 1)) & 
                     (rules['consequents'].apply(lambda x: len(x) == 1))]

print(rules_1_to_1)

# Filtrar regras que têm 'doce' como consequente
rules_with_sweet = rules[rules['consequents'].apply(lambda x: 'Doce' in x)]

print(rules_with_sweet)