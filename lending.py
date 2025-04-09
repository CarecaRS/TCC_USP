import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#from datetime import datetime
pd.set_option('display.max_rows', 250)
pd.set_option('display.max_columns', None)
np.set_printoptions(suppress=True, precision=4)
%autoindent OFF 

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

# Função para manipulação das datas
"""
#LEGACY
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
"""
def separa_datas(df, col):
    temp = pd.DataFrame()
    print('\nResgatando informações do dataframe...')
    temp[col] = pd.to_datetime(df[col], format='%b-%Y')
    dias = []  # cria lista vazia para ser preenchida com a informação dos dias
    mes_ano = []  # cria lista vazia para ser preenchida com mês/ano
    if temp[col].isnull().sum() > 0:
        print(f'\nATENÇÃO! Identificadas NaNs na feature {col}, impossível continuar.')
        print(f'Favor resolver as {temp[col].isnull().sum()} NaNs antes de tentar novamente. Seu bosta.')
    else:
        #
        print('\nSeparando parâmetros de data e gerando nova série, só um instante por favor.')
        for i in range(0, len(temp)):  # preenche a lista com mês e ano
            mes_ano.append(temp.loc[i, col].strftime('%b-%Y').lower())
        #
        print('\nGerando nova série ajustada.')
        #
        return pd.Series(data=mes_ano, name=col)  # retorna uma série nova


# Função para buscar a explicação de cada feature no dicionário e analisar NaNs
def analise_geral(lista, dataframe: pd.DataFrame):
    """
    :param lista: lista de features a se verificar
    :dtype lista: list
    :param dataframe: o banco de dados que contem todas as features que serão verificadas
    :type dataframe: DataFrame
    """
    def significado(x: object):
        dic = pd.read_csv('./data/dictionary.csv', index_col=False)
        print(f'Nome: {str(x).upper()}')
        print(f'Definição: {dic[dic['Feature'] == str(x)]['Descrição'].values}')
    if len(lista) > 1:
        for i in range(0, len(lista)):
            print('\n')
            print(f'Feature #{i+1}/{len(lista)}')
            significado(lista[i])
            print(f'Tipo de dado: {str(dataframe[lista[i]].dtypes).upper()}')
            print(f'Quantidade total de NaNs: {dataframe[lista[i]].isnull().sum()}')
            print(f'Proporção desses NaNs: {round((dataframe[lista[i]].isnull().sum()/len(dataframe))*100, 2)}%')
    elif len(lista) == 1:
        print('\n')
        significado(lista[0])
        print(f'Tipo de dado: {str(dataframe[lista[i]].dtypes).upper()}')
        print(f'Quantidade total de NaNs: {dataframe[lista[0]].isnull().sum()}')
        print(f'Proporção desses NaNs: {round((dataframe[lista[0]].isnull().sum()/len(dataframe))*100, 2)}%')


# Função que analisa apenas as features sem NaNs
def analise_nans(lista, dataframe: pd.DataFrame):
    """
    :param lista: lista de features a se verificar
    :dtype lista: list
    :param dataframe: o banco de dados que contem todas as features que serão verificadas
    :type dataframe: DataFrame
    """
    def significado(x: object):
        dic = pd.read_csv('./data/dictionary.csv', index_col=False)
        print(f'Nome: {str(x).upper()}')
        print(f'Definição: {dic[dic['Feature'] == str(x)]['Descrição'].values}')
    if len(lista) > 1:
        print('\n')
        print('Calculando tamanho da solicitação... Só um instante por favor.')
        tam = len(dataframe[lista].isnull().sum()[dataframe[lista].isnull().sum() > 0])
        idx = 1
        for i in range(0, len(lista)):
            if dataframe[lista[i]].isnull().sum() == 0:
                pass
            else:
                print('\n')
                print(f'Feature #{idx}/{tam}')
                significado(lista[i])
                print(f'Tipo de dado: {str(dataframe[lista[i]].dtypes).upper()}')
                print(f'Quantidade total de NaNs: {dataframe[lista[i]].isnull().sum()}')
                print(f'Proporção desses NaNs: {round((dataframe[lista[i]].isnull().sum()/len(dataframe))*100, 2)}%')
                idx += 1
    elif len(lista) == 1:
        if dataframe[lista[0]].isnull().sum() == 0:
            print('\n')
            print('Sem NaNs nessa lista.')
        else:
            print('\n')
            significado(lista[0])
            print(f'Tipo de dado: {str(dataframe[lista[0]].dtypes).upper()}')
            print(f'Quantidade total de NaNs: {dataframe[lista[0]].isnull().sum()}')
            print(f'Proporção desses NaNs: {round((dataframe[lista[0]].isnull().sum()/len(dataframe))*100, 2)}%')


############
# DATA WRANGLING: TAXAS DE JUROS
#=========================

# Tratamento do arquivo com as taxas de juros
effr = pd.read_csv('./data/EFFR.csv')
effr['observation_date'] = pd.to_datetime(effr['observation_date'])  # transforma a informação de cronologia de object para datas
#!
mask_pos = effr['observation_date'] >= '2021-09-01'  # seleciona as datas até 2021Q3
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
effr['mes_ano'] = separa_datas(effr, 'mes_ano')  # faz o ajuste de mês e ano novamente
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
analise_geral(feature_drop, emprestimos)

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
#!
# Relembrando cada uma das features
analise_geral(hard_cols, hardships)
#!
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
#!
#!
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
analise_geral(lista, joint_ap)
#!
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
#!
# As fico ranges copia-se do mutuário principal, uma vez que é impossível estimar. Pode-se partir da
# premissa que muito embora o segundo mutuário não seja idêntico ao primeiro eles devem possuir
# hábitos e comportamentos semelhantes na grande maioria das vezes.
#!
idx_conj = joint_ap[joint_ap['contrato_conjunto'] == 1].index.to_list()  # armazena os índices de todos contratos conjuntos
vazios = joint_ap.loc[idx_conj][joint_ap.loc[idx_conj, lista[0]].isnull()].index.to_list()  # filtra apenas os vazios em sec_app_fico_range_low
joint_ap.loc[vazios, lista[0]] = emprestimos.loc[vazios, 'fico_range_low']  # registra o score do primeiro proponente também no segundo
#!
idx_conj = joint_ap[joint_ap['contrato_conjunto'] == 1].index.to_list()  # armazena os índices de todos contratos conjuntos
vazios = joint_ap.loc[idx_conj][joint_ap.loc[idx_conj, lista[1]].isnull()].index.to_list()  # filtra apenas os vazios em sec_app_fico_range_high
joint_ap.loc[vazios, lista[1]] = emprestimos.loc[vazios, 'fico_range_high']  # registra o score do primeiro proponente também no segundo
#!
# As demais features (com exceção de sec_app_revol_util) são variáveis discretas sobre consultas a cadastro,
# quantidade de recuperações de dívidas, hipotecas, etc., do segundo proponente. NaNs preenchidos com zero:
zeros = joint_ap.isnull().sum()[joint_ap.isnull().sum() > 53].index.to_list()
joint_ap[zeros] = joint_ap[zeros].fillna(0)


############
# DATA WRANGLING: TRATAMENTO DAS DEMAIS FEATURES
#=========================
emprestimos = pd.read_parquet('data/emprestimos_dropado.parquet')
dropar = ['Unnamed: 0', 'url', 'zip_code']  # features sem utilização no contexto deste estudo
emprestimos.drop(dropar, axis=1, inplace=True)
#acima = emprestimos.isnull().sum()[emprestimos.isnull().sum()/emprestimos.shape[0] > 0.3].index.to_list()
# Uma observação completamente NaN, dropando abaixo e já refazendo o index
nulo = emprestimos['home_ownership'].isnull()
nulo = emprestimos.loc[nulo].index.values
emprestimos.drop(nulo, inplace=True)
emprestimos.reset_index(drop=True, inplace=True)
#!
# 'tax_liens' se preenche com zeros, como não tem como estimar essa informação então
# parte-se do princípio que o tomador não tenha essa recuperação judicial
emprestimos['tax_liens'] = emprestimos['tax_liens'].fillna(0)
#!
# As observações em 'next_pymnt_d' são todas ou liquidadas ou transferidas para prejuízo
mask = emprestimos['next_pymnt_d'].isnull()
#emprestimos.loc[mask, 'loan_status'].value_counts()  # caso seja necessária a verificação
#!
# Criação dos índices dos contratos e substituição dos NaNs
pagos = emprestimos.loc[mask, 'loan_status'][emprestimos.loc[mask, 'loan_status'] == 'Fully Paid'].index.values
pagos = emprestimos.iloc[pagos]['next_pymnt_d'].fillna('fully_paid')
emprestimos['next_pymnt_d'] = emprestimos['next_pymnt_d'].fillna(pagos)
#!
prejuizo = emprestimos.loc[mask, 'loan_status'][emprestimos.loc[mask, 'loan_status'] == 'Charged Off'].index.values
prejuizo = emprestimos.iloc[prejuizo]['next_pymnt_d'].fillna('charged_off')
emprestimos['next_pymnt_d'] = emprestimos['next_pymnt_d'].fillna(prejuizo)
#!
#!
# 'mths_since_rcnt_il': se não existe informação sobre a última concessão de crédito
# ao tomador, infere-se que o contrato vigente foi o último na análise. Logo, fillna(-1)
emprestimos['mths_since_rcnt_il'] = emprestimos['mths_since_rcnt_il'].fillna(-1)
#!
#!
# 'emp_length' fillna('Unknown'), não tem como inferir tempo de emprego dos proponentes
emprestimos['emp_length'] = emprestimos['emp_length'].fillna('Unknown')
#!
#!
# 'earliest_cr_line' também não tem como inferir data do primeiro crédito concedido
# ao tomador, então  utiliza-se este como referência
mask = emprestimos['earliest_cr_line'].isnull()
earliest = emprestimos.loc[mask, 'issue_d']
emprestimos['earliest_cr_line'] = emprestimos['earliest_cr_line'].fillna(earliest)
#!
#!
# 'emp_title' fillna('Unknown'), impossível inferir cargo do proponente
emprestimos['emp_title'] = emprestimos['emp_title'].fillna('Unknown')
#!
#!
# 'title' - nome do empréstimo. Já tem uma feature com o objetivo do
# empréstimo proposto ('purpose'). Pode apenas fazer uma feature nova
# binária tipo 'empréstimo tem nome? sim/não'.
# Procedimento de binning aqui, as categorias mais significativas
# serão mantidas como estão, as demais serão enquandadas como 'outros'. Utilizada
# representatividade > 0.65% em função de ser o limiar que não repete categorias
# similares.
# Antes de qualquer processo de binning, os valores NaN são imputados como 'unknown'.
#!
sem_nome = emprestimos['title'].isnull()
emprestimos['title'] = emprestimos['title'].fillna('Unknown')
relacao = emprestimos.loc[~sem_nome, 'title'].value_counts(normalize=True)
relacao = relacao[relacao > 0.0065].index.to_list()
relacao.append('Unknown')
mask_fica = emprestimos['title'].isin(relacao)
#nomes = emprestimos.loc[~mask_fica, 'title']
nomes = emprestimos['title'].copy()

#!
# Primeiro faz uma leitura de possíveis enquadramentos
# de cartão de crédito (geral)
card = []
for obs in nomes.to_list():
    if "american express" in obs.lower():
        card.append(obs)
    if "card" in obs.lower():
        card.append(obs)
#!
# Em seguida faz a substituição dos valores de grafia diversa
# como relacao[1]
for obs in pd.Series(card).unique():
    mask = nomes == obs
    nomes.loc[mask] = relacao[1]
#!
# Segunda etapa, consolidações de débitos
consolid = []
for obs in nomes.to_list():
    if "eliminat" in obs.lower():
        consolid.append(obs)
    if "consol" in obs.lower():
        consolid.append(obs)
    if "debt" in obs.lower():
        consolid.append(obs)
    if "pay" in obs.lower():
        consolid.append(obs)
    if "bills" in obs.lower():
        consolid.append(obs)
    if "finance" in obs.lower():
        consolid.append(obs)
# Substituindo os valores como relacao[0]
for obs in pd.Series(consolid).unique():
    mask = nomes == obs
    nomes.loc[mask] = relacao[0]
#!
# Terceira etapa, despesas médicas e de saúde
medic = []
for obs in nomes.to_list():
    if "medic" in obs.lower():
        medic.append(obs)
    if "hospital" in obs.lower():
        medic.append(obs)
    if "health" in obs.lower():
        medic.append(obs)
    if "dentist" in obs.lower():
        medic.append(obs)
# Substituindo os valores como relacao[5]
for obs in pd.Series(medic).unique():
    mask = nomes == obs
    nomes.loc[mask] = relacao[5]
#!
# Quinta etapa, relativo a reforma de casa própria
casa_reforma = []
for obs in nomes.to_list():
    if "improv" in obs.lower():
        casa_reforma.append(obs)
    if "house" in obs.lower():
        casa_reforma.append(obs)
    if "kitchen" in obs.lower():
        casa_reforma.append(obs)
    if "room" in obs.lower():
        casa_reforma.append(obs)
    if "backyard" in obs.lower():
        casa_reforma.append(obs)
    if "back yard" in obs.lower():
        casa_reforma.append(obs)
    if "floor" in obs.lower():
        casa_reforma.append(obs)
    if "pool" in obs.lower():
        casa_reforma.append(obs)
    if "basement" in obs.lower():
        casa_reforma.append(obs)
    if "roof" in obs.lower():
        casa_reforma.append(obs)
    if "hot tub" in obs.lower():
        casa_reforma.append(obs)
    if "windows" in obs.lower():
        casa_reforma.append(obs)
# Substituindo os valores como relacao[2]
for obs in pd.Series(casa_reforma).unique():
    mask = nomes == obs
    nomes.loc[mask] = 'House improvement'
#!
# Sexta etapa, relativo a compra de imóveis
casa_compra = []
for obs in nomes.to_list():
    if "real state" in obs.lower():
        casa_compra.append(obs)
    if "property" in obs.lower():
        casa_compra.append(obs)
    if "construc" in obs.lower():
        casa_compra.append(obs)
    if "moving" in obs.lower():
        casa_compra.append(obs)
    if "apartment" in obs.lower():
        casa_compra.append(obs)
    if "home" in obs.lower():
        casa_compra.append(obs)
# Substituindo os valores como relacao[2]
for obs in pd.Series(casa_compra).unique():
    mask = nomes == obs
    nomes.loc[mask] = 'Real Estate'
#!
# Quarta etapa, relativo a empresas e empreendimentos
startup = []
for obs in nomes.to_list():
    if "business" in obs.lower():
        startup.append(obs)
    if "personel" in obs.lower():
        startup.append(obs)
    if "startup" in obs.lower():
        startup.append(obs)
    if "start-up" in obs.lower():
        startup.append(obs)
# Substituindo os valores como relacao[7]
for obs in pd.Series(startup).unique():
    mask = nomes == obs
    nomes.loc[mask] = relacao[7]
#!
# Setima etapa, veículos e similares
veiculos = []
for obs in nomes.to_list():
    if "car " in obs.lower():
        veiculos.append(obs)
    if "motor" in obs.lower():
        veiculos.append(obs)
    if "bike" in obs.lower():
        veiculos.append(obs)
    if "vehicle" in obs.lower():
        veiculos.append(obs)
    if "triumph" in obs.lower():
        veiculos.append(obs)
    if "wheeler" in obs.lower():
        veiculos.append(obs)
    if "honda" in obs.lower():
        veiculos.append(obs)
    if "boat" in obs.lower():
        veiculos.append(obs)
    if "truck" in obs.lower():
        veiculos.append(obs)
    if "engine" in obs.lower():
        veiculos.append(obs)
# Substituindo os valores como relacao[6]
for obs in pd.Series(veiculos).unique():
    mask = nomes == obs
    nomes.loc[mask] = relacao[6]
#!
# Oitava etapa, green loans e empréstimos pessoais
verdes = []
for obs in nomes.to_list():
    if "green" in obs.lower():
        verdes.append(obs)
# Substituindo os valores como relacao[6]
for obs in pd.Series(verdes).unique():
    mask = nomes == obs
    nomes.loc[mask] = 'Green loan'
#!
pessoal = []
for obs in nomes.to_list():
    if "personal" in obs.lower():
        pessoal.append(obs)
    if "my" in obs.lower():
        pessoal.append(obs)
    if "freedom" in obs.lower():
        pessoal.append(obs)
    if "fresh start" in obs.lower():
        pessoal.append(obs)
    if "wedding" in obs.lower():
        pessoal.append(obs)
# Substituindo os valores como relacao[6]
for obs in pd.Series(pessoal).unique():
    mask = nomes == obs
    nomes.loc[mask] = 'Personal Loan'
#!
# Última etapa, tudo que não for enquadramento constante em
# 'relacao' é classificado como 'Outros'. O dataset original já
# possui um enquandramentto 'Others', aqui opta-se por separar
# essas classificações para evitar ruídos
#!
relacao = nomes.value_counts(normalize=True).head(8).index.to_list()
mask = nomes.isin(relacao)
nomes.loc[~mask] = 'Outros'


emprestimos['title'].head(20)

len(nomes)

"""
"""

nomes.head(20)

emprestimos['title'] = emprestimos['title'].fillna(nomes)  # retorna ao dataset original depois de pronto

emprestimos.loc[sem_nome, 'title']

emprestimos['title'].isnull().sum()

# comecei 13:34 / 14:04 ainda no primeiro / segunda em 14:17, quinte em 15:47 / oitava começou 16:13


nomes.value_counts(normalize=True)

names = pd.read_parquet('data/titles.parquet')



# Feature last_credit_pull_d refere-se ao mês em que foi realizada
# a última consulta ao FICO (credito score) do cliente. Pressupõe-se
# que pelo menos no mês de liberação do crédito tenha sido consultado.
mask = emprestimos['last_credit_pull_d'].isnull()
liberados = emprestimos.loc[mask, 'issue_d']
emprestimos['last_credit_pull_d'] = emprestimos['last_credit_pull_d'].fillna(liberados)
#!
#!


# A feature 'last_pymnt_d' refere-se ao último pagamento recebido (com ref base 09/2020).
# Imputações baseam-se nas informações de 'loan_status':
# [0] Charged off: 150 dias de atraso
# [1] Issued: pelo menos 121 dias de atraso
# [2] Late (31-120 days)
# [3] In grace period: até 15 dias de atraso (não tem ônus)
# [4] Does not meet the credit policy. Status: Charged Off' (pois é, não sei)
# [5] Late (16-30 days)
# [6] Current: em dia, não está em atraso. 
mask = emprestimos['last_pymnt_d'].isnull()
statuses = emprestimos.loc[mask, 'loan_status'].value_counts().index.to_list()
#!
# Status Charged off
#!
# Status Issued
#!
# Status Late (31-120 days)
#!
# Status In grace period
#!
# Status Does not meet the credit policy. Status: Charged Off' (pois é, não sei)
#!
# Status Late (16-30 days)
#!
# Status Current
mask_stat = emprestimos.loc[mask, 'loan_status'][emprestimos.loc[mask, 'loan_status'] == statuses[6]].index.values
stat = emprestimos.iloc[mask_stat]['last_pymnt_d'].fillna(emprestimos['issue_d'].max())
emprestimos['last_pymnt_d'] = emprestimos['last_pymnt_d'].fillna(stat)
#!
# Transforma a feature de str para data:
# emprestimos['last_pymnt_d'] = separa_datas(emprestimos, 'last_pymnt_d')
#!
#!
#!
emprestimos.loc[mask, 'loan_status'].value_counts()
#!



numeros = emprestimos.dtypes[emprestimos.dtypes == 'float64'].index.values
objetos = emprestimos.dtypes[emprestimos.dtypes == 'object'].index.values
analise_nans(objetos, emprestimos)

analise_nans(numeros, emprestimos)

emprestimos['total_rev_hi_lim']

mask = emprestimos['last_pymnt_d'].isnull()
emprestimos.loc[mask, ['loan_status', 'issue_d']]#.value_counts()
emprestimos.loc[mask, 'issue_d'].value_counts()


emprestimos[['purpose', 'title']].head(100)

emprestimos['title'].value_counts()

"""
approach 1
mês atual (set/2020) pega os prazos de enquadramento em cada uma das situações e infere uma data plausível qualquer

approach 2 MELHOR
cálculo com mês de liberação do empréstimo e prazo total. se tiver saldo devedor total (total_bal_il ou tot_cur_bal ou tot_hi_cred_lim) dá para inferir quantas parcelas o cara pagou.

tem loan_amount, int_rate e installment

pub_rec_bankruptcies (com cuidado essa), inq_fi, mort_acc se tá NaN substitui por zero

total_cu_tl = open_act_il ou total_acc

"""

emprestimos.head(500).to_csv('data/thiago.csv')


# Verificação das dimensões do dataset limpo
print(f'Número de observações (linhas): {emprestimos.shape[0]}')  # 2260701
print(f'Número de features (colunas): {emprestimos.shape[1]}')  # 105




# Grafico das correlações no df completo
temp = emprestimos.dtypes[emprestimos.dtypes == 'float64'].index.to_list()
correl = emprestimos[temp].corr()
sns.heatmap(correl,
            annot = False,
            cmap = 'Oranges')
plt.show()

