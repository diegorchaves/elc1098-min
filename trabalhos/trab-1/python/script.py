import pandas as pd 
import json
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder 

json_data = pd.read_json("./data/padaria-marcas.json")
json_str = json_data.to_json(orient='records')
print("JSON data:")
print(json_data)

# Carregar os dados do JSON
data = json.loads(json_str)

# Extrair apenas a lista de produtos
transactions = [item['produtos'] for item in data]
print("Transactions:")
print(transactions)

# Utilizando TransactionEncoder para transformar os dados
encoder = TransactionEncoder()
encoded_array = encoder.fit(transactions).transform(transactions)

print("Encoded array:")
print(encoded_array)

# Criar um DataFrame
df = pd.DataFrame(encoded_array, columns=encoder.columns_)
print("Data frame:")
print(df)

# Uso do algoritmo Apriori
frequent_itemsets = apriori(df, min_support=0.05, use_colnames=True)
print("Frequent itemsets:")
print(frequent_itemsets)

# Gerar regras de associação
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.3)

print("Regras gerais:")
print(rules.sort_values(by='confidence', ascending=False))

# Filtrar regras 1 para 1
rules_1_to_1 = rules[(rules['antecedents'].apply(lambda x: len(x) == 1)) & 
                     (rules['consequents'].apply(lambda x: len(x) == 1))]

print("Regras 1 para 1:")
print(rules_1_to_1.sort_values(by='confidence', ascending=False))

# Filtrar regras que têm 'doce' como consequente
#rules_with_sweet = rules[rules['consequents'].apply(lambda x: 'Doce' in x)]

#print("Regras com doce:")
#print(rules_with_sweet)