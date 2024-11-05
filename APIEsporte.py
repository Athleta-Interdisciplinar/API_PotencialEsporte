from flask import Flask, request, jsonify
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Carregando o pipeline que já inclui o MultiLabelBinarizer
with open('Modelos serializados\\pipeline_knn.pkl', 'rb') as file:
    pipeline = pickle.load(file)

@app.route('/analise', methods=['POST'])
def analisar():
    dados = request.json.get('dados')
    dados_df = pd.DataFrame(dados)
    
    # Detecta automaticamente as colunas categóricas
    colunas_categoricas = dados_df.select_dtypes(include=['object']).columns
    
    # Codifica cada coluna categórica
    for col in colunas_categoricas:
        le = LabelEncoder()
        dados_df[col] = le.fit_transform(dados_df[col])

    # Faz a previsão usando o pipeline carregado
    resultado = pipeline.predict(dados_df)
    return jsonify({'resultado': resultado.tolist()})

if __name__ == '__main__':
    app.run(port=5000, debug=True)