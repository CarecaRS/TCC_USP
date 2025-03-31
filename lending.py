# A fazer ainda: verificar datas, pegar taxas de juros básicas dos EUA nessas épocas
#
# URLs de referência:
# https://www.kaggle.com/code/faressayah/lending-club-loan-defaulters-prediction
# https://www.kaggle.com/code/pavlofesenko/minimizing-risks-for-loan-investments
# https://www.kaggle.com/code/pileatedperch/predicting-charge-off-from-initial-listing-data
#
# Banco de dados:
# https://www.kaggle.com/datasets/wordsforthewise/lending-club/data
# Refere-se aos empréstimos cedidos pelo Lending Club (EUA), de 2007 até 2018Q4.
#
# Informações sobre as features:
# https://www.kaggle.com/datasets/imsparsh/lending-club-loan-dataset-2007-2011
# 
# Informações econômicas:
# Como taxa de juros de referência será utilizada a Effective Federal Funds Rate:
# https://www.newyorkfed.org/markets/reference-rates/additional-information-about-reference-rates#effr_obfr_calculation_methodology
# https://www.newyorkfed.org/markets/reference-rates/effr
# 
# Download do CSV: https://fred.stlouisfed.org/series/EFFR
#
# https://ycharts.com/indicators/effective_federal_funds_rate#:~:text=Basic%20Info-,Effective%20Federal%20Funds%20Rate%20is%20at%204.33%25%2C%20compared%20to%204.33,long%20term%20average%20of%204.61%25.  (sim, acaba com um ponto mesmo)
# The Effective Federal Funds Rate is the rate set by the FOMC (Federal Open Market Committee)
# for banks to borrow funds from each other. The Federal Funds Rate is extremely important because
# it can act as the benchmark to set other rates. Historically, the Federal Funds Rate reached as
# high as 22.36% in 1981 during the recession. Additionally, after the financial crisis in 2008-2009,
# the Federal Funds rate nearly reached zero when quantitative easing was put into effect.
#
# FR 2040: https://www.federalreserve.gov/apps/reportingforms/Report/Index/FR_2420
#
# https://fred.stlouisfed.org/series/DFEDTARU  # limite superior
# https://fred.stlouisfed.org/series/DFEDTARL  # limite inferior
# https://fred.stlouisfed.org/series/DFEDTAR  # target, tem a informação da mudança em 2008
#
# Para efeitos de raciocínio, a Effective Federal Funds Rate representa para os Estados Unidos o que a
# taxa DI (depósito interbancário) representa para o Brasil. Ambas servem como proxy para a taxa de
# juros básica de suas respectivas nações/economias. Diferente do Brasil, os EUA não possuem uma
# taxa-alvo (como é comumente conhecida a nossa Selic), eles trabalham com piso e teto de referência
# desde 2008. Por questões de reproducibilidade, opta-se então por trabalhar com a EFFR tendo em vista
# essa taxa ser divulgada continuamente desde 1954 até hoje.
#
# Explica o que é taxa básica de juros, explica o que é o FED.
#
# Em relação aos lags da taxa de juros: eu não consegui encontrar histórico de projeções
# da taxa básica de juros anterior a 2020, então tentar usar a taxa real futura mais um
# ruído aleatório de até 0.5pp
#
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_rows', 250)
pd.set_option('display.max_columns', None)
np.set_printoptions(suppress=True, precision=6)
%autoindent OFF 

#from datetime import datetime
#from sklearn.metrics import r2_score, log_loss
#from sklearn.experimental import enable_iterative_imputer
#from sklearn.impute import IterativeImputer
#from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import OneHotEncoder
#from sklearn import metrics
#import xgboost as xgb
#from catboost import CatBoostClassifier
#from lightgbm import LGBMClassifier


############
# FUNÇÕES
#=========================


# Função para manipulação das datas do arquivo com as taxas de juros do FED
def separa_datas(df, col):
    # Manipulação das informações
    dias = []  # cria lista vazia para ser preenchida com a informação dos dias
    mes_ano = []  # cria lista vazia para ser preenchida com mês/ano
    #
    for i in range(0, df.shape[0]):  # preenche as listas
        dias.append(df.loc[i, col].strftime('%d'))
        mes_ano.append(df.loc[i, col].strftime('%b-%Y'))
    #
    novo_df = {'dia':dias, 'mes_ano':mes_ano}  # cria um dict com as informações
    #
    return pd.DataFrame(novo_df)  # retorna um novo dataframe


############
# DATA WRANGLING
#=========================

# Tratamento do arquivo com as taxas de juros
effr = pd.read_csv('./data/EFFR.csv')
effr['observation_date'] = pd.to_datetime(effr['observation_date'])  # transforma a informação de cronologia de object para datas
#
mask_pos = effr['observation_date'] >= '2020-01-01'  # seleciona as datas pós ano de 2019
mask_pre = effr['observation_date'] <= '2006-12-31'  # seleciona as dadas pré ano de 2017
dropar = np.array(effr[mask_pos | mask_pre].index)  # filtra os índices referente a essas datas em um array
effr = effr.drop(dropar).dropna().reset_index(drop=True)  # dropa as datas selecionadas e elimina NaNs (dias como Natal e Ano-Novo, por exemplo), refazendo o índice
#
datas = separa_datas(effr, 'observation_date')
taxas = (effr['EFFR']/100).round(8)  # transforma as taxas de percentual para decimal, isolando em uma serie. Utilizada função round(8) pois alguns elementos estavam sendo calculados com um '1' na décima casa decimal
effr = pd.concat([datas, taxas], axis=1)
#
# O registro da concessão dos empréstimos contém informação apenas de mês/ano do contrato, não da data em que foi feito. Para utilização
# de taxa de juros vigente bem como taxa futura esperada, será utilizada a média das taxas reais praticadas em cada respectivo mês.
effr = effr.groupby(['mes_ano'])['EFFR'].mean().reset_index()  # agrega as informações por mês, informando a média
effr['mes_ano'] = pd.to_datetime(effr['mes_ano'], format='%b-%Y')  # manipulação de object para data
effr = effr.sort_values(by='mes_ano').reset_index(drop=True)  # ordena em função da data
effr['mes_ano'] = separa_datas(effr, 'mes_ano')['mes_ano']  # faz o ajuste de mês e ano novamente






effr.shift(6)

expec_6m

expec_12m





# Leitura do arquivo com as informações dos contratos
emprestimos = pd.read_csv('./data/loans.csv', low_memory=False) # hiperparâmetro 'low_memory' aqui apenas para não retornar aviso no console

# Verificação das dimensões do dataset
print(f'Número de observações (linhas): {emprestimos.shape[0]}')  # 2260701
print(f'Número de features (colunas): {emprestimos.shape[1]}')  # 151

# A feature 'loan_status' contem as informações necessárias:
# - Fully paid: empréstimos quitados
# - Current: empréstimos ativos
# - Charged off: empréstimos vencidos já reclassificados como prejuízo
# - Late (31-120 days): empréstimos em atraso
# - In Grace Period: empréstimos atrasados mas que ainda podem ser pagos sem ônus, neste estudo considera-se este enquadramento
#                    como período entre 1-15 dias de atraso, dadas as demais características do banco de dados
# - Late (16-30 days): empréstimos em atraso
# - Does not meet the credit policy (ambos): descartado, pois não atendem a política de crédito
# - Default: empréstimos atrasados a mais de 120 dias mas ainda não enquadrados em prejuízo
emprestimos['loan_status'].value_counts()

# Esse estudo se propõe a analisar o risco de atraso de uma operação no tocante à questão de prejuízo à instituição financeira.
# Para tanto, serão considerados os empréstimos cujos loan_status estejam enquadrados em default ou pior, pois assume-se que
# os contratos que possuem atraso de até 120 dias possuam uma chance razoável de recebimento por parte da instituição, bem como
# na literatura em inglês o estudado é o risco efetivo de default, não o risco de atraso.


############
# EXPLORATORY DATA ANALYSIS (EDA)
#=========================

# Verificando as features com maiores % de valores NaN:
# 38 features com NaNs acima de 90% do total de observações;
# 44 acima de 50% das observações;
# 46 acima de 40% das observações;
# 58 acima de 37% das observações.
len(emprestimos.isnull().sum()[emprestimos.isnull().sum()/emprestimos.shape[0] > 0.37])

# Dada a alteração na quantidade de variáveis entre 37 e 40% de NaNs, opta-se por eliminar o menor número possível
# entre essas duas, logo, as features com quantidade de NaNs equivalente a 40% ou mais das observações serão eliminadas
feature_drop = emprestimos.isnull().sum()[emprestimos.isnull().sum()/emprestimos.shape[0] > 0.4].index.values
#emprestimos = emprestimos.drop(feature_drop, axis=1)

# Contudo, algumas NaN podem ser preenchidas por alguma estrutura lógica. Assim, cada uma das features
# constantes em 'feature_drop' serão analisadas individualmente

# member_id
# desc
mths_since_last_delinq
mths_since_last_record
next_pymnt_d
mths_since_last_major_derog
annual_inc_joint
dti_joint
verification_status_joint
mths_since_rcnt_il
il_util
mths_since_recent_bc_dlq
mths_since_recent_revolv_delinq
revol_bal_joint
sec_app_fico_range_low
sec_app_fico_range_high
sec_app_earliest_cr_line
sec_app_revol_util
sec_app_open_act_il
sec_app_num_rev_accts
sec_app_chargeoff_within_12_mths
sec_app_collections_12_mths_ex_med
sec_app_mths_since_last_major_derog
hardship_type
hardship_reason
hardship_status
deferral_term
hardship_amount
hardship_start_date
hardship_end_date
payment_plan_start_date
hardship_length_
hardship_dpd
hardship_loan_status
orig_projected_additional_accrued_interest
hardship_payoff_balance_amount
hardship_last_payment_ammount
debt_settlement_flag_date
settlement_status
settlement_date
settlement_amount
settlement_percentage
settlement_term




emprestimos[emprestimos['desc'].isnull() == False]['desc']  # são anotações diversas no contrato. NaNs substituídos por 'none'

feature_drop









# Verificação das dimensões do dataset limpo
print(f'Número de observações (linhas): {emprestimos.shape[0]}')  # 2260701
print(f'Número de features (colunas): {emprestimos.shape[1]}')  # 105


float_cols = emprestimos.dtypes[emprestimos.dtypes == 'float64'].index.values  # 81
obj_cols = emprestimos.dtypes[emprestimos.dtypes == 'object'].index.values  # 24

emprestimos[obj_cols]

emprestimos[float_cols]


for col in emprestimos[feature_drop].columns:
    print(f'\nFeature {col}:')
    print(emprestimos[col].value_counts())

emprestimos[emprestimos['desc'].isnull() == False]['desc']

emprestimos[feature_drop].value_counts()


