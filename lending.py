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
# DATA WRANGLING: TAXAS DE JUROS
#=========================

# Tratamento do arquivo com as taxas de juros
effr = pd.read_csv('./data/EFFR.csv')
effr['observation_date'] = pd.to_datetime(effr['observation_date'])  # transforma a informação de cronologia de object para datas
#
mask_pos = effr['observation_date'] >= '2020-01-01'  # seleciona as datas pós ano de 2019
mask_pre = effr['observation_date'] <= '2005-12-31'  # seleciona as dadas pré ano de 2006
dropar = np.array(effr[mask_pos | mask_pre].index)  # filtra os índices referente a essas datas em um array
effr = effr.drop(dropar).dropna().reset_index(drop=True)  # dropa as datas selecionadas e elimina NaNs (dias como Natal e Ano-Novo, por exemplo), refazendo o índice
#
datas = separa_datas(effr, 'observation_date')
taxas = effr['EFFR']  # isola as taxas de juros
effr = pd.concat([datas, taxas], axis=1)
#
# O registro da concessão dos empréstimos contém informação apenas de mês/ano do contrato, não da data em que foi feito. Para utilização
# de taxa de juros vigente bem como taxa futura esperada, será utilizada a média das taxas reais praticadas em cada respectivo mês.
effr = effr.groupby(['mes_ano'])['EFFR'].mean().reset_index()  # agrega as informações por mês, informando a média
effr['mes_ano'] = pd.to_datetime(effr['mes_ano'], format='%b-%Y')  # manipulação de object para data
effr = effr.sort_values(by='mes_ano').reset_index(drop=True)  # ordena em função da data
effr['mes_ano'] = separa_datas(effr, 'mes_ano')['mes_ano']  # faz o ajuste de mês e ano novamente
#
# O ruído é baseado nos últimos 6 períodos para a expectativa futura de 6 meses e
# baseado nos últimos 12 períodos para a expectativa futura de 12 meses
expec_6m = effr['EFFR'].rolling(7).mean().dropna().reset_index(drop=True)
expec_12m = effr['EFFR'].rolling(13).mean().dropna().reset_index(drop=True)
#
# Criação das expectativas futuras das taxas de juros com adição de ruído.
desvio = 0.5  # meio ponto percentual a.a.
#
ruido_6m = []
for i in range(0,len(expec_6m)):
    temp = np.random.normal(expec_6m[i], desvio)
    ruido_6m.append(temp)
expec_6m = abs(pd.Series(ruido_6m, name='expec6m'))  # utiliza valores em módulo, não existem juros negativos na prática
#
ruido_12m = []
for i in range(0,len(expec_12m)):
    temp = np.random.normal(expec_12m[i], desvio)
    ruido_12m.append(temp)
expec_12m = abs(pd.Series(ruido_12m, name='expec12m'))
#
# Une as expectativas de taxas em um único dataframe
expectativas = pd.concat([expec_6m, expec_12m], axis=1)
#
# E une o dataframe original
effr = pd.concat([effr, expectativas], axis=1)
#
# Com as manipulações já feitas, seleciona-se apenas o período compreendido pelo
# banco de dados contendo as informações dos contratos (2007 a 2018)
effr = effr.drop(range(0,12)).dropna().reset_index(drop=True)


############
# DATA WRANGLING: INFORMAÇÕES DOS CONTRATOS
#=========================




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


