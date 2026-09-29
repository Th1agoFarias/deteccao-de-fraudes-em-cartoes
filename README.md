# Credit Card Fraud Detection

Projeto de análise exploratória e classificação de transações com cartão de crédito para detectar fraudes. O fluxo completo, as interpretações e as métricas estão nos notebooks de EDA e modelagem.

## Estrutura

```text
credit_fraud/
├── data/
│   ├── raw/
│   │   ├── .gitkeep
│   │   └── creditcard.csv       
│   └── .gitignore
├── notebooks/
│   ├── classification_credit_EDA.ipynb
│   └── modeling_credit_fraude.ipynb
├── src/
│   ├── __init__.py
│   ├── eda_utils.py
│   └── modeling.py
├── .gitignore
├── README.md
└── requirements.txt
```

### Código reutilizável

- `src/eda_utils.py`: estatísticas descritivas, gráficos por classe, correlação e resumo de outliers.
- `src/modeling.py`: avaliação de modelos, validação cruzada, avaliação de thresholds e gráficos de desempenho.

## Dados

Coloque o arquivo CSV na raiz do projeto, no caminho:

`data/raw/creditcard.csv`

Os notebooks carregam os dados com `df = pd.read_csv("data/raw/creditcard.csv")`. Inicie o Jupyter na raiz do projeto para que esse caminho relativo e os imports de `src` sejam resolvidos corretamente.

Neste projeto, o dataset é usado na EDA para explorar o desbalanceamento e as diferenças entre transações normais e fraudulentas; depois, serve de base para treinar e avaliar modelos de detecção de fraude.

O Credit Card Fraud Detection reúne 284.807 transações ocorridas em dois dias, das quais 492 são fraudes (cerca de 0,172%). Foi publicado no Kaggle pelo Machine Learning Group da ULB, a partir de dados coletados e analisados em colaboração com a Worldline e a ULB. Fonte: [Credit Card Fraud Detection (Kaggle)](https://www.kaggle.com/mlg-ulb/creditcardfraud/data).

O CSV não é incluído no repositório. O `.gitignore` exclui o dataset e ambientes/cache locais; o arquivo `data/raw/.gitkeep` preserva a pasta vazia.

## Ambiente e instalação

Os notebooks foram preparados para Python 3.11. Crie e ative um ambiente virtual e instale as dependências:

```bash
python -m venv .venv
```

No Windows, ative o ambiente com:

```powershell
.venv\Scripts\Activate.ps1
```

Instale os pacotes:

```bash
python -m pip install -r requirements.txt
```

## Execução

A partir da raiz do projeto, inicie o Jupyter:

```bash
jupyter lab
```

Execute os notebooks nesta ordem:

1. `notebooks/classification_credit_EDA.ipynb`
2. `notebooks/modeling_credit_fraude.ipynb`

A seção final do notebook de modelagem consolida os resultados. Para reproduzir os cálculos e gráficos, use **Restart Kernel / Run All**.

## EDA

O notebook de EDA inspeciona dimensões, tipos, valores ausentes e duplicatas. Depois cria divisões estratificadas de treino, validação e holdout antes da análise orientada ao target; essa análise é feita no treino para preservar validação e holdout.

A exploração cobre o desbalanceamento de `Class`, as variáveis `Amount` e `Time`, os componentes `V1` a `V28`, correlação e outliers. Correlação é tratada como evidência exploratória, não como causalidade ou importância preditiva. Outliers são identificados, mas não removidos automaticamente, pois podem representar comportamento fraudulento.

## Modelagem e validação

O notebook compara Regressão Logística, Random Forest e XGBoost. Usa validação cruzada estratificada em cinco folds e prioriza **Average Precision (AP)**, mais informativa que acurácia isolada para este target muito desbalanceado.

O SMOTE é aplicado dentro do pipeline, apenas aos dados de treino de cada fold. Como o SMOTE usa distâncias entre vizinhos, `StandardScaler` é aplicado antes dele. O threshold é escolhido na validação; o holdout permanece separado até a avaliação final.

## Resultados

O dataset contém 284.807 transações e 31 colunas: 284.315 transações normais e 492 fraudes (0,173%). A divisão estratificada usada no notebook tem 182.276 observações no treino, 45.569 na validação e 56.962 no holdout.

Na comparação inicial da validação, XGBoost obteve AP de 0,826643. Na validação cruzada, Random Forest teve AP médio de 0,839120 ± 0,040610 sem SMOTE e 0,850057 ± 0,029655 com SMOTE (proporção 0,1). O pipeline Random Forest + StandardScaler + SMOTE foi usado na avaliação final.

O threshold selecionado na validação foi 0,6. No holdout, o resultado foi:

| Métrica | Resultado |
|---|---:|
| Accuracy | 0,999561 |
| Precision | 0,910112 |
| Recall | 0,826531 |
| F1 | 0,866310 |
| Average Precision | 0,878417 |
| ROC-AUC | 0,964211 |

Com esse threshold, o modelo detectou 81 das 98 fraudes e gerou 8 falsos positivos. A matriz de confusão está organizada com linhas como classe real e colunas como classe prevista:

| Classe real | Prevista normal | Prevista fraude |
|---|---:|---:|
| Normal | 56.856 | 8 |
| Fraude | 17 | 81 |

Como fraudes representam somente 0,173% das transações, a acurácia deve ser interpretada junto com AP, precisão e recall. O threshold operacional também deve considerar os custos de falsos positivos e falsos negativos.

## Dependências

As versões compatíveis dos pacotes estão registradas em `requirements.txt`, incluindo pandas, scikit-learn, imbalanced-learn, XGBoost, matplotlib, seaborn e JupyterLab.

## Próximos passos

- Integrar um LLM para analisar as transações sinalizadas pelo modelo e apoiar a verificação de possíveis fraudes.
- Desenvolver uma aplicação Streamlit para consultar previsões e visualizar os resultados do modelo de forma interativa.
