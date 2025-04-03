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
        mes_ano.append(df.loc[i, col].strftime('%b-%Y').lower())
    #
    novo_df = {'dia':dias, 'mes_ano':mes_ano}  # cria um dict com as informações
    #
    return pd.DataFrame(novo_df)  # retorna um novo dataframe


# Função para buscar a explicação de cada feature no dicionário e analisar NaNs
def analise(lista, dataframe: pd.DataFrame):
    """
    :param lista: lista de features a se verificar
    :dtype lista: list
    :param dataframe: o banco de dados que contem todas as features que serão verificadas
    :type dataframe: DataFrame
    """
    def significado(x: object):
        dic = pd.read_csv('./data/dictionary.csv', index_col=False)
        print(f'Feature: {str(x).upper()}')
        print(f'Explicação: {dic[dic['Feature'] == str(x)]['Descrição'].values}')
    if len(lista) > 1:
        for i in range(0, len(lista)):
            print('\n')
            print(f'Feature #{i+1}/{len(lista)}')
            significado(lista[i])
            print(f'Quantidade total de NaNs: {dataframe[lista[i]].isnull().sum()}')
            print(f'Proporção desses NaNs: {round((dataframe[lista[i]].isnull().sum()/len(dataframe))*100, 2)}%')
    elif len(lista) == 1:
        print('\n')
        significado(lista[0])
        print(f'Quantidade total de NaNs: {dataframe[lista[0]].isnull().sum()}')
        print(f'Proporção desses NaNs: {round((dataframe[lista[0]].isnull().sum()/len(dataframe))*100, 2)}%')



############
# DATA WRANGLING: TAXAS DE JUROS
#=========================

# Tratamento do arquivo com as taxas de juros
effr = pd.read_csv('./data/EFFR.csv')
effr['observation_date'] = pd.to_datetime(effr['observation_date'])  # transforma a informação de cronologia de object para datas
#!
mask_pos = effr['observation_date'] >= '2021-04-01'  # seleciona as datas pós 2020Q1
mask_pre = effr['observation_date'] <= '2005-12-31'  # seleciona as dadas pré ano de 2006
dropar = np.array(effr[mask_pos | mask_pre].index)  # filtra os índices referente a essas datas em um array
effr = effr.drop(dropar).dropna().reset_index(drop=True)  # dropa as datas selecionadas e elimina NaNs (dias como Natal e Ano-Novo, por exemplo), refazendo o índice
#!
datas = separa_datas(effr, 'observation_date')
taxas = effr['EFFR']  # isola as taxas de juros
effr = pd.concat([datas, taxas], axis=1)
#!
# O registro da concessão dos empréstimos contém informação apenas de mês/ano do contrato, não da data em que foi feito. Para utilização
# de taxa de juros vigente bem como taxa futura esperada, será utilizada a média das taxas reais praticadas em cada respectivo mês.
effr = effr.groupby(['mes_ano'])['EFFR'].mean().reset_index()  # agrega as informações por mês, informando a média
effr['mes_ano'] = pd.to_datetime(effr['mes_ano'], format='%b-%Y')  # manipulação de object para data
effr = effr.sort_values(by='mes_ano').reset_index(drop=True)  # ordena em função da data
effr['mes_ano'] = separa_datas(effr, 'mes_ano')['mes_ano']  # faz o ajuste de mês e ano novamente
#!
# O ruído é baseado nos últimos 6 períodos para a expectativa futura de 6 meses e
# baseado nos últimos 12 períodos para a expectativa futura de 12 meses
expec_6m = effr['EFFR'].rolling(7).mean().dropna().reset_index(drop=True)  # desloca as observações 6 períodos à frente
expec_12m = effr['EFFR'].rolling(13).mean().dropna().reset_index(drop=True)  # desloca 12 períodos
#!
# Criação das expectativas futuras das taxas de juros com adição de ruído.
desvio = 0.5  # meio ponto percentual a.a.
#!
ruido_6m = []
for i in range(0,len(expec_6m)):
    temp = np.random.normal(expec_6m[i], desvio)  # gera um ruído aleatório ao redor da taxa real utilizando o desvio estabelecido
    ruido_6m.append(temp)
expec_6m = abs(pd.Series(ruido_6m, name='expec6m'))  # utiliza valores em módulo, não existem juros negativos na prática
#!
ruido_12m = []
for i in range(0,len(expec_12m)):
    temp = np.random.normal(expec_12m[i], desvio)
    ruido_12m.append(temp)
expec_12m = abs(pd.Series(ruido_12m, name='expec12m'))
#!
# Une as expectativas de taxas em um único dataframe
expectativas = pd.concat([expec_6m, expec_12m], axis=1)
#!
# E une o dataframe original
effr = pd.concat([effr, expectativas], axis=1)
#!
# Com as manipulações já feitas, seleciona-se apenas o período compreendido pelo
# banco de dados contendo as informações dos contratos (2007 a 2018)
effr = effr.drop(range(0,12)).dropna().reset_index(drop=True)


############
# DATA WRANGLING: INFORMAÇÕES DOS CONTRATOS
#=========================

# Leitura do arquivo com as informações dos contratos
emprestimos = pd.read_csv('./data/loans2020.csv', low_memory=False) # hiperparâmetro 'low_memory' aqui apenas para não retornar aviso no console
#!
# Ajuste de duas features enquadradas como 'object' em função de notação percentual
emprestimos['revol_util'] = emprestimos['revol_util'].str.replace('%', '').astype('float64')
emprestimos['int_rate'] = emprestimos['int_rate'].str.replace('%', '').astype('float64')
#!
# Algumas observações em revol_util não estão corretas (cálculo revol_bal/total_rev_hi_lim). Empiricamente
# a ocorrência de registros diferentes é de menos de 3% do total, então opta-se por recalcular toda a feature
# revol_util
utilizado = emprestimos['revol_bal']
total = emprestimos['total_rev_hi_lim']
revol_util = utilizado/total
emprestimos['revol_util'] = revol_util

# A feature 'loan_status' contem as informações necessárias:
# - Fully paid: empréstimos quitados
# - Current: empréstimos ativos
# - Charged off: empréstimos vencidos já reclassificados como prejuízo
# - Late (31-120 days): empréstimos em atraso
# - In Grace Period: empréstimos atrasados mas que ainda podem ser pagos sem ônus, neste estudo considera-se este enquadramento
#                    como período entre 1-15 dias de atraso, dadas as demais características do banco de dados
# - Late (16-30 days): empréstimos em atraso
# - Does not meet the credit policy (ambos): descartados? não atendem a política de crédito - vai saber
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
# Para verificação:
# len(emprestimos.isnull().sum()[emprestimos.isnull().sum()/emprestimos.shape[0] > 0.3])

# Dada a alteração na quantidade de variáveis entre 37 e 40% de NaNs, opta-se por eliminar o menor número possível
# entre essas duas, logo, as features com quantidade de NaNs equivalente a 40% ou mais das observações serão eliminadas
feature_drop = emprestimos.isnull().sum()[emprestimos.isnull().sum()/emprestimos.shape[0] > 0.3].index.values

# Análise individual de cada uma dessas features
analise(feature_drop, emprestimos)

# Seleção das features de renegociação (15 variáveis no total)
hard_cols = []
hard_cols.append(feature_drop[34])  # essas três features não possuem 'hardship' no nome, então são
hard_cols.append(feature_drop[30])  # inseridas manualmente
hard_cols.append(feature_drop[26])
for hardship in emprestimos.columns.to_list():
    if "hardship" in hardship:
        hard_cols.append(hardship)
#!
#!
# Seleção das features de aplicação conjunta (15 variáveis)
sep_ap = []
for other_applicant in emprestimos.columns.to_list():
    if "sec_app_" in other_applicant:
        sep_ap.append(other_applicant)
    if "joint" in other_applicant:
        sep_ap.append(other_applicant)


# Segregando features de aplicação conjunta e de renegociação
joint_ap = emprestimos[sep_ap].copy()
emprestimos.drop(sep_ap, axis=1, inplace=True)
#!
hardships = emprestimos[hard_cols].copy()
emprestimos.drop(hard_cols, axis=1, inplace=True)

# Salva as informações de forma individual em Parquet (mais rápido, menos espaço)
#hardships.to_parquet('data/hardships.parquet')
#joint_ap.to_parquet('data/joint_ap.parquet')

############
# DATA WRANGLING: TRATAMENTO DOS CONTRATOS COM RENEGOCIAÇÃO
#=========================
hardships = pd.read_parquet('data/hardships.parquet')
hard_cols = hardships.columns.to_list()

# Relembrando cada uma das features
analise(hard_cols, hardships)

# Parte-se do princípio que quem não tenha status definido de hardship
# é em função de não ter realizado acordo algum
mask = hardships['hardship_loan_status'].isnull() == False
#!
# Criando feature nova identificando quais contratos tem renegociação
hardships['fez_hardship'] = 0
hardships.loc[mask, 'fez_hardship'] = 1
#!
# Verificação de NaNs dentro destes que fizeram hardship
hardships.loc[mask].isnull().sum()[hardships.loc[mask].isnull().sum() > 0]
#!
# hardship_flag preenchendo com a moda ('N')
#hardships.loc[mask][hardships.loc[mask, 'hardship_flag'].isnull()].T
hardships.loc[mask, 'hardship_flag'] = hardships.loc[mask, 'hardship_flag'].fillna('N')
#!
# hardship_status preenchendo com 'ACTIVE'
#hardships.loc[mask][hardships.loc[mask, 'hardship_status'].isnull()].T
hardships.loc[mask, 'hardship_status'] = hardships.loc[mask, 'hardship_status'].fillna('ACTIVE')
#!
# hardship_reason preenchendo com 'Unknown', não foi percebido padrão
#hardships.loc[mask][hardships.loc[mask, 'hardship_reason'].isnull()]
hardships.loc[mask, 'hardship_reason'] = hardships.loc[mask, 'hardship_reason'].fillna('UNKNOWN')
#!
# orig_projected_additional_accrued_interest preenchido com zero, em função de 'hardship_type'
#hardships.loc[mask][hardships.loc[mask, 'orig_projected_additional_accrued_interest'].isnull()]
hardships.loc[mask, 'orig_projected_additional_accrued_interest'] = hardships.loc[mask, 'orig_projected_additional_accrued_interest'].fillna(0)
#!
#!
# Todas as features com dtypes numéricos terão fillna(0)
# As variáveis numéricas (desconsiderando 'fez_hardship')
lista = hardships.dtypes[hardships.dtypes == 'float64'].index.to_list()
hardships[lista] = hardships[lista].fillna(0)
#!
#!
# Todas as features com dtypes categóricos terão fillna específico
lista = hardships.dtypes[hardships.dtypes == 'object'].index.to_list()
hardships['hardship_flag'] = hardships['hardship_flag'].fillna('N')  # não tem motivo por ter sido flaggado
hardships[lista] = hardships[lista].fillna('Not Applicable')
#!
#!
# A feature hardship_reason tem classes repetidas, com grafia diferente.
# De repente pode ser unido DISABILITY com MEDICAL também.
# Ajuste abaixo.
mask = hardships['hardship_reason'] == 'INCOMECURT'
hardships.loc[mask, 'hardship_reason'] = 'INCOME_CURTAILMENT'
#!
mask = hardships['hardship_reason'] == 'UNEMPLOYED'
hardships.loc[mask, 'hardship_reason'] = 'UNEMPLOYMENT'
#!
mask = hardships['hardship_reason'] == 'REDCDHOURS'
hardships.loc[mask, 'hardship_reason'] = 'REDUCED_HOURS'
#!
mask = hardships['hardship_reason'] == 'NATDISAST'
hardships.loc[mask, 'hardship_reason'] = 'NATURAL_DISASTER'
#!
mask = hardships['hardship_reason'] == 'EXCESSOBLI'
hardships.loc[mask, 'hardship_reason'] = 'EXCESSIVE_OBLIGATIONS'
#!
mask = hardships['hardship_reason'] == 'FINANCIAL'  # Agregando 'financial' em 'excessive_obligations' pois, via de regra, tem a mesma causa
hardships.loc[mask, 'hardship_reason'] = 'EXCESSIVE_OBLIGATIONS'
#!
mask = hardships['hardship_reason'] == 'DEATH'  # Agregando 'death' em 'family_death', deduz-se que o falecido não vai pedir renegociação do próprio contrato
hardships.loc[mask, 'hardship_reason'] = 'FAMILY_DEATH'
#!
# Trabalhar com uppercase é desconfortável, ajustando:
for i in range(0, len(lista)):
    hardships[lista[i]] = hardships[lista[i]].str.lower()


############
# DATA WRANGLING: TRATAMENTO DOS CONTRATOS COM APLICAÇÃO CONJUNTA
#=========================
joint_ap = pd.read_parquet('data/joint_ap.parquet')
sep_ap = joint_ap.columns.to_list()
#!
# Criação de indicador de contrato com composição de renda
mask = joint_ap['annual_inc_joint'].isnull()
joint_ap['contrato_conjunto'] = 0
joint_ap.loc[~mask, 'contrato_conjunto'] = 1
#!
# Verificação dos dtypes 'object' primeiro:
lista = joint_ap.dtypes[joint_ap.dtypes == 'object'].index.to_list()
#!
# As observações de 'verification_status_joint' (lista[0]) com NaN que foram
# identificadas como contrato conjunto terão fillna('Not Verified')
mask = joint_ap['contrato_conjunto'] == 1
#!
# Preenche primeiro os contratos marcados como conjuntos
joint_ap.loc[mask, lista[0]] = joint_ap.loc[mask, lista[0]].fillna('Not Verified')
#!
# Em seguida preenche os demais contratos com 'Not Applicable'
joint_ap[lista[0]] = joint_ap[lista[0]].fillna('Not Applicable')
#!
#!
# A outra feature categórica, 'sec_app_earliest_cr_line' (lista[1]),
# refere-se à data do primeiro empréstimo tomado pelo segundo proponente.
# Como é impossível essa estimação, toma-se a data do contrato em pleito
# como data de primeiro empréstimo.
valores = emprestimos.loc[mask][joint_ap.loc[mask, lista[1]].isnull()]['issue_d'].values
indices = emprestimos.loc[mask][joint_ap.loc[mask, lista[1]].isnull()]['issue_d'].index.to_list()
joint_ap.loc[indices, lista[1]] = valores
#!
# O restante das NaNs substitui-se por 'Not Applicable'
joint_ap[lista[1]] = joint_ap[lista[1]].fillna('Not Applicable')
#!
#!
# Em seguida, verificação dos dtypes float64
lista = joint_ap.dtypes[joint_ap.dtypes == 'float64'].index.to_list()
#!
# Primeiro os contratos que são individuais serão preenchidos com zero
mask = joint_ap['contrato_conjunto'] == 0
joint_ap[mask] = joint_ap[mask].fillna(0)
#!
# Trata-se individualmente as variáveis dos contratos conjuntos
mask = joint_ap['contrato_conjunto'] == 1
#!
# Explicação das features que necessitam ajuste
analise(lista, joint_ap)

# sec_app_fico_range_low(high) temos problemas... ver como faz para estimar depois. de repente tem um last_fico_range_low/high do 2o proponente
# todas as outras, zeradas pq faz sentido


# NaNs na feature dti_joint assumem o valor de dti
indices = joint_ap[joint_ap[lista[1]].isnull()][lista[1]].index.to_list()
valores = emprestimos.iloc[indices]['dti'].values
joint_ap.loc[indices, lista[1]] = valores
#!
# revol_bal_joint assume o valor de revol_bal
indices = joint_ap[joint_ap[lista[2]].isnull()][lista[2]].index.to_list()
valores = emprestimos.iloc[indices]['revol_bal'].values
joint_ap.loc[indices, lista[2]] = valores
#!
# sec_app_revol_util assume o valor de revol_util
indices = joint_ap[joint_ap[lista[8]].isnull()][lista[8]].index.to_list()
valores = emprestimos.iloc[indices]['revol_util'].values  # existem NaNs no dataset original, depois se lida com isso
joint_ap.loc[indices, lista[8]] = valores
#!
indices = joint_ap[joint_ap[lista[8]].isnull()][lista[8]].index.to_list() # refazendo índice, ainda ficam 53 NaNs
lista = joint_ap.isnull().sum()[joint_ap.isnull().sum() > 53].index.to_list()


# As fico ranges copia-se do mutuário principal, uma vez que é impossível estimar. Pode-se partir da
# premissa que muito embora o segundo mutuário não seja idêntico ao primeiro eles devem possuir
# hábitos e comportamentos semelhantes na maioria das vezes.

# 

joint_ap[lista]


joint_ap.loc[indices]

emprestimos.iloc[idx].to_csv('data/thiago.csv')

indices2 = []
for i in range(0, len(indices)):
    indices2.append(indices[i]+300)
emprestimos.loc[indices2, thiago].to_csv('data/thiago2.csv')

indices2














# Verificação das dimensões do dataset limpo
print(f'Número de observações (linhas): {emprestimos.shape[0]}')  # 2260701
print(f'Número de features (colunas): {emprestimos.shape[1]}')  # 105


float_cols = emprestimos.dtypes[emprestimos.dtypes == 'float64'].index.values  # 81
obj_cols = emprestimos.dtypes[emprestimos.dtypes == 'object'].index.values  # 24


# Grafico das correlações no df completo
temp = emprestimos.dtypes[emprestimos.dtypes == 'float64'].index.to_list()
correl = emprestimos[temp].corr()
sns.heatmap(correl,
            annot = False,
            cmap = 'Oranges')
plt.show()

