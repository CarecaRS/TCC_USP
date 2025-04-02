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


# Função para buscar a explicação de cada feature no dicionário
def significado(x: object):
    dic = pd.read_csv('./data/dictionary.csv', index_col=False)
    print(f'Feature: {x.upper()}')
    print(f'Explicação: {dic[dic['Feature'] == x]['Descrição'].values}')


# Função para a análise individual de cada feature -- depende de 'feature_drop'
def analise(x: int):
    significado(feature_drop[x])
    print(f'Quantidade total de NaNs: {emprestimos[feature_drop[x]].isnull().sum()}')
    print(f'Proporção desses NaNs: {round((emprestimos[feature_drop[x]].isnull().sum()/len(emprestimos))*100, 2)}%')



############
# DATA WRANGLING: TAXAS DE JUROS
#=========================

# Tratamento do arquivo com as taxas de juros
effr = pd.read_csv('./data/EFFR.csv')
effr['observation_date'] = pd.to_datetime(effr['observation_date'])  # transforma a informação de cronologia de object para datas
#
mask_pos = effr['observation_date'] >= '2021-04-01'  # seleciona as datas pós 2020Q1
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
expec_6m = effr['EFFR'].rolling(7).mean().dropna().reset_index(drop=True)  # desloca as observações 6 períodos à frente
expec_12m = effr['EFFR'].rolling(13).mean().dropna().reset_index(drop=True)  # desloca 12 períodos
#
# Criação das expectativas futuras das taxas de juros com adição de ruído.
desvio = 0.5  # meio ponto percentual a.a.
#
ruido_6m = []
for i in range(0,len(expec_6m)):
    temp = np.random.normal(expec_6m[i], desvio)  # gera um ruído aleatório ao redor da taxa real utilizando o desvio estabelecido
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
emprestimos = pd.read_csv('./data/loans2020.csv', low_memory=False) # hiperparâmetro 'low_memory' aqui apenas para não retornar aviso no console


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
# 29 features com NaNs acima de 90% do total de observações;
# 35 acima de 50% das observações;
# 35 acima de 40% das observações;
# 37 acima de 30% das observações;
# 49 acima de 29% das observações.
len(emprestimos.isnull().sum()[emprestimos.isnull().sum()/emprestimos.shape[0] > 0.3])

# Dada a alteração na quantidade de variáveis entre 37 e 40% de NaNs, opta-se por eliminar o menor número possível
# entre essas duas, logo, as features com quantidade de NaNs equivalente a 40% ou mais das observações serão eliminadas
feature_drop = emprestimos.isnull().sum()[emprestimos.isnull().sum()/emprestimos.shape[0] > 0.3].index.values

# Verificação do que se tratam essas features com altos índices de NaNs:
for col in feature_drop:
    print(significado(col))

# Análise individual de cada uma dessas features
for i in range(0, len(feature_drop)):
    print(f'Feature número: {i+1}/{len(feature_drop)}')
    analise(i)
    print('\n')

# Feature mths_since_last_delinq: meses desde a última inadimplência. Se nunca foi inadimplente, esse campo
# certamente será NaN. Imputado 999. De repente incluir uma nova feature tipo 'já inadimpliu?', e esses NaNs
# colocar tudo como 'não' lá.
emprestimos[feature_drop[0]] = emprestimos[feature_drop[0]].fillna(999)

# Feature mths_since_last_record: semelhante à acima
emprestimos[feature_drop[1]] = emprestimos[feature_drop[1]].fillna(999)

# Feature next_pymnt_d: pode dropar
emprestimos[emprestimos[feature_drop[2]].isnull() == False][[feature_drop[2], 'issue_d', 'loan_status']]

# Feature mths_since_last_major_derog: semelhante à primeira analisada
emprestimos[feature_drop[3]] = emprestimos[feature_drop[3]].fillna(999)

# Feature annual_inc_joint: deixa igual à renda individual será?
# Feature dti_joint mesmo raciocínio, deixa igual à dti
# Feature verification_status_joint pode fazer uma categoria nova como 'None' ou 'No Applicable'
# mths_since_rcnt_il pega a data do último contrato, subtrai da data atual
# il_util pega o saldo total e pega o limite de crédito, divide um pelo outro
# mths_since_recent_bc_dlq similar à primeira feature, de repente tem que pensar algo diferente nessas coisas
# mths_since_recent_revol_delinq mesmo raciocínio
# revol_bal_joint aqui tem que ver o que se esses NaNs são de contas individuais. Se sim, puxa o valor da conta individual. Bom criar uma coluna categórica de 'conjunto ou não'
# sec_app_XYZ (11 features) se não tem segundo proponente, aqui vai ser NaN sempre
# hardship_type pode incluir 'None' aqui, por exemplo. De repente nova feature hardship sim/não
# hardship_reason aqui tem que olhar de perto, faz um binning e OHE
# harsthip_status quase idêntico ao type. Se não tem hardship, não tem acordo.
# deferral_term mesma coisa, esse deve ser o prazo do acordo de renegociação
# hardship_amount se não tem acordo, aqui vai zer NaN mesmo. Preenche com zeros.
# hardship_start_date se não tem hardship, não tem início.
# hardship_end_date mesma coisa
# payment_plan_start_date data devida do primeiro pagamento
# hardship_length prazo do acordo
# hardship_dpd fez contrato de renegociação e atrasou por X dias
# hardship_loan_status esse aqui deve determinar todos os outros hardships, dar uma olhada com carinho.
# orig_projected_additional_accrued_interest se não tem renegociação, esse campo é zerado. Preenche com zero.
# hardship_payoff_balance_amount saldo inicial do acordo hardship. se não tem acordo, aqui vai ser NaN mesmo (ou zero)
# hardship_last_payment_amount se não fez acordo, isso não tem nem como existir.








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


