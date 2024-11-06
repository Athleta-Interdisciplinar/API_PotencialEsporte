from flask import Flask, request, jsonify
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

with open('./pipeline_knn.pkl', 'rb') as file:
    pipeline = pickle.load(file)

@app.route('/analise', methods=['POST'])
def analisar():
    dados = request.json.get('dados')
    if dados is None:
        return jsonify({'erro': 'Dados não fornecidos'}), 400
        
    dados_df = pd.DataFrame(dados)

    colunas_categoricas = dados_df.select_dtypes(include=['object']).columns

    for col in colunas_categoricas:
        le = LabelEncoder()
        dados_df[col] = le.fit_transform(dados_df[col])

    resultado = pipeline.predict(dados_df)
    return jsonify({'resultado': resultado.tolist()})

if __name__ == '__main__':
   app.run(host='0.0.0.0', port=5000, debug=True)
