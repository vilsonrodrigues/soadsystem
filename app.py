import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
#import plotly.express as px
#import joblib


#--------------------------0.0 funções -------------------------------------



#----------0.1 funcoes para carregar os dados para a aplicacao ---------
#@st.cache
def get_df_treino():
	return pd.read_csv('data/df_treino.csv')

def get_df_predict():
	return pd.read_csv('data/df_predict.csv')

def get_df_relatorio():
	return pd.read_csv('data/df_relatorio.csv')


#----0.2 funcao que retorna quais sao os modelos disponiveis conforme o departamento ----
#@st.cache
def get_lista_modelos_disponiveis(df_relatorio, departamento):
	return df_relatorio.loc[df_relatorio['codigo'].str.contains(departamento), 'codigo']


#---------------0.3 função para carregar os modelos conforme a disciplina --------------
#@st.cache
def get_model(disciplina_selecionada):
	return joblib.load( open('model/modelo_' + disciplina_selecionada + '.pkl','rb'))


#----------------- 0.4 função para carregar o pipeline ----------------
#@st.cache
def pipeline_predicao():
	return joblib.load( open('pipeline/pipeline_predicao.pkl','rb'))


#-----------nao apaga -------
def drop_duplicates_in_list(x):
	return list(dict.fromkeys(x))


def get_departamentos(df_treino):	
	#lista com todos os departamentos
	departamentos = list(df_treino['codigo'].str[0:3].unique())
	#inserindo um estado inicial
	departamentos.append('---')
	#ordenando
	departamentos = sorted(departamentos)
	return departamentos


def get_disciplinas(df_treino, departamento_selecionado):	
	#lista de todas as disciplinas no departamento selecionado
	disciplinas = df_treino.loc[df_treino['codigo'].str.contains(departamento_selecionado), 'codigo']
	disciplinas = list(disciplinas)
	disciplinas = drop_duplicates_in_list(disciplinas)
	disciplinas = sorted(disciplinas)
	return disciplinas


def get_matriculas(df_predict):
	#pegando todas as matriculas
	matriculas = list(df_predict['matricula'].unique())		
	return matriculas


def get_dados_estudante(df_predict, matricula):
	cols = ['Matrícula','Média das Notas','Menor Nota','Maior Nota','Quantidade de Disciplinas Cursadas','Quantidade de Aprovações',
	        'Quantidade de Reprovacões','Média das Aprovadas','Média das Reprovadas','Carga Horária','Taxa de Sucesso','Semestre']
	df = df_predict.iloc[:,:12]
	df.columns = cols
	df.sort_values(by = ['Semestre'], ascending = False, inplace = True)
	df['Semestre'] = df['Semestre'].astype(str)
	df['Matrícula'] = df['Matrícula'].astype(str)
	df = df.loc[df['Matrícula'].str.contains(matricula),:]
	return df


def dadosGraficoTaxaDeSucesso(df_treino, codigo):
	df_treino = df_treino.groupby(by=['codigo','semestrePassado']).mean()['taxaDeSucesso'].reset_index()
	return df_treino.loc[df_treino['codigo'].str.contains(codigo),:]


def concatenaDisciplinasTotalAprovacaoReprovacao(df_historico):
	df = pd.DataFrame(columns = ['Semestre','Quantidade','Campo'])
	df_total = df_historico[['Semestre','Quantidade de Disciplinas Cursadas']]
	df_total['Campo'] = 'Quantidade de Disciplinas'
	df_total = df_total.rename(columns = {'Quantidade de Disciplinas Cursadas':'Quantidade'})

	df_aprov = df_historico[['Semestre','Quantidade de Aprovações']]
	df_aprov['Campo'] = 'Quantidade de Aprovações'
	df_aprov = df_aprov.rename(columns = {'Quantidade de Aprovações':'Quantidade'})

	df_reprov = df_historico[['Semestre','Quantidade de Reprovacões']]
	df_reprov['Campo'] = 'Quantidade de Reprovacões'
	df_reprov = df_reprov.rename(columns = {'Quantidade de Reprovacões':'Quantidade'})

	
	df = df.append(df_total, ignore_index=True)
	#df = df.append(df_aprov, ignore_index=True)
	df = df.append(df_reprov, ignore_index=True)
	df.sort_values(by = ['Semestre'], ascending = True, inplace = True)
	return df

def concatenaMediaMaiorMenor(df_historico):
	df = pd.DataFrame(columns = ['Semestre','Quantidade','Campo'])
	df_media = df_historico[['Semestre','Média das Notas']]
	df_media['Campo'] = 'Média das Notas'
	df_media = df_media.rename(columns = {'Média das Notas':'Quantidade'})

	df_menor = df_historico[['Semestre','Menor Nota']]
	df_menor['Campo'] = 'Menor Nota'
	df_menor = df_menor.rename(columns = {'Menor Nota':'Quantidade'})

	df_maior = df_historico[['Semestre','Maior Nota']]
	df_maior['Campo'] = 'Maior Nota'
	df_maior = df_maior.rename(columns = {'Maior Nota':'Quantidade'})

	df = df.append(df_media, ignore_index=True)
	df = df.append(df_maior, ignore_index=True)
	df = df.append(df_menor, ignore_index=True)
	df.sort_values(by = ['Semestre'], ascending = True, inplace = True)

	return df

def concatenaMediaAproRepro(df_historico):
	df = pd.DataFrame(columns = ['Semestre','Quantidade','Campo'])
	df_apro = df_historico[['Semestre','Média das Aprovadas']]
	df_apro['Campo'] = 'Média das Aprovadas'
	df_apro = df_apro.rename(columns = {'Média das Aprovadas':'Quantidade'})

	df_repro = df_historico[['Semestre','Média das Reprovadas']]
	df_repro['Campo'] = 'Média das Reprovadas'
	df_repro = df_repro.rename(columns = {'Média das Reprovadas':'Quantidade'})
	
	df = df.append(df_apro, ignore_index=True)
	df = df.append(df_repro, ignore_index=True)
	df.sort_values(by = ['Semestre'], ascending = True, inplace = True)

	return df	

def taxaSucessoNaDisciplina(df_treino, codigo):
	df_cursadas = df_treino.groupby(by=['codigo','semestreAtual'])['situacao'].count().reset_index()
	df_cursadas.rename(columns = {'situacao':'total'},inplace=True)
	df_cursadas = df_cursadas.loc[df_cursadas['codigo'].str.contains(codigo)]		
	df_cursadas.sort_values(by='semestreAtual',inplace=True)

	df_aprovados = df_treino.loc[df_treino['codigo'].str.contains(codigo) & df_treino['situacao'].str.contains('APROVADO'),
		                    ['codigo','semestreAtual','situacao']]		
	df_aprovados = df_aprovados.groupby(by=['codigo','semestreAtual'])['situacao'].count().reset_index()
	df_aprovados.rename(columns = {'situacao':'aprovados'},inplace=True)		
	df_aprovados.sort_values(by='semestreAtual',inplace=True)

	df = pd.merge(df_cursadas,df_aprovados, on=['codigo','semestreAtual'])
	df.fillna(0)

	df['taxa'] = np.divide(df['aprovados'],df['total'])

	return df


def main():

	#-----------------------1.0 página inicial ----------------------------------
	st.markdown('## **SISTEMA DE ORIENTAÇÃO ACADÊMICA - SOAD**')
	st.sidebar.title('Painel de Atributos do SOAD')

	#-------------------------- 2.0 sidebar -------------------------------------


	#------------------------ 2.1 upload de dados -------------------------------

	df_treino = get_df_treino()
	df_predict = get_df_predict()
	df_relatorio = get_df_relatorio()


	#---------------------- 2.2 checkbox para verificar se é docente------------

	#se for professor, deixar digitar a senha
	if st.sidebar.checkbox('Você é docente da UFRN?'):
		senha = st.sidebar.text_input('Digite a sua senha', type = 'password')
	else:
		senha = ''


	#------------------------ 2.3 seleciona departamento -----------------------

	#selecionando departamentos da universidade
	departamentos = get_departamentos(df_treino)

	#primeiro escolhe o departamento 
	departamento_selecionado = st.sidebar.selectbox('Selecione um departamento:', departamentos)


	#-------------------- 2.3 Seleciona as disciplinas com base no departamento --------------

	if departamento_selecionado:
		disciplinas = get_disciplinas(df_treino, departamento_selecionado)
	else:
		discplinas = []

	disciplinas = sorted(disciplinas)

	disciplina_selecionada = st.sidebar.selectbox('Selecione uma disciplina', disciplinas)


	#------------------ 2.4 Seleciona o aluno ---------------------

	if disciplina_selecionada:
		matriculas = get_matriculas(df_predict)
	else:
		matriculas = []	

	#se a senha do professor estiver correta, deixa digitar a matricula
	if senha == 'teamlab':
		matricula = st.sidebar.text_input('Digite a matricula do estudante')
	#acessa os dados do aluno


	#---------------------- 2.5 submissão das consultas -----------------
	#botao para submissão da disciplina selecionada
	if st.sidebar.button('Submeter'):
		try:
			st.sidebar.success('Submissão realizada com sucesso')
			codigo = disciplina_selecionada
			st.markdown('## Dashboard')
			st.markdown('')
			
			#----------------------------------------------
			#dados do grafico taxa de sucesso na disciplina

			df_graph = taxaSucessoNaDisciplina(df_treino, codigo)	
			df_graph['semestreAtual'] = df_graph['semestreAtual'].astype(str)
			#grafico
			st.subheader('Taxa de Sucesso em '+codigo+' por Semestre') 
			a = alt.Chart(df_graph).mark_area(clip = True).encode(
				x=alt.X('semestreAtual', axis=alt.Axis(title='Semestre')),
				y=alt.Y('taxa', scale=alt.Scale(domain=(0,1)),axis=alt.Axis(format='%', 
					title='Taxa de Sucesso'))
	    	).configure_mark(
			    opacity=0.6,
			    color='purple'
			)
			st.altair_chart(a, use_container_width=True)

			if matricula:

				#----------------- dados do historico do aluno ------------------

				df_historico = get_dados_estudante(df_predict, matricula)


				#------------------taxa de sucesso do aluno por semestre
				st.subheader('Taxa de Sucesso do Estudante por Semestre')
				b = alt.Chart(df_historico).mark_bar().encode(
					x=alt.X('Semestre', axis=alt.Axis(title='Semestre')),
					y=alt.Y('Taxa de Sucesso',scale=alt.Scale(domain=(0,1)), 
						axis=alt.Axis(format='%')),
		    	).configure_mark(
				    opacity=0.6,
				    color='#ff4d4d'
				)
				st.altair_chart(b, use_container_width=True)


				#-----------Comparação das Notas---------------
				df_notas = concatenaMediaMaiorMenor(df_historico)
				st.subheader('Notas: Maior, Menor e Média, por Semestre')
				d = alt.Chart(df_notas).mark_area(opacity = 0.7).encode(
				    x='Semestre',
				    y=alt.Y('Quantidade',axis=alt.Axis(title=''),scale=alt.Scale(domain=(0,10))),
				    color=alt.Color('Campo', scale=alt.Scale(scheme='category20c'), 
				    	  legend=alt.Legend(title='Categorias',orient='bottom')),
				    row='Campo'
				).properties(
				    height=100
				)
				st.altair_chart(d, use_container_width=True)


				
				#----------Quantidade de Disciplinas por Semestre-------------
				
				df_quant = concatenaDisciplinasTotalAprovacaoReprovacao(df_historico)
				st.subheader('Disciplinas Cursadas vs. Reprovações por Semestre')
				c = alt.Chart(df_quant).mark_area(opacity = 0.7).encode(
							x=alt.X('Semestre', axis=alt.Axis(title='Semestre')),
							y=alt.Y('Quantidade', axis=alt.Axis(title='Quantidades'),stack=None),
							color=alt.Color('Campo', scale=alt.Scale(scheme='greens'), 
				    	 	      legend=alt.Legend(title='Categorias',orient='bottom'))).properties(
				width=200,height=400)
				st.altair_chart(c, use_container_width=True)


				#-----------Comparação das Notas---------------
				df_media_apro_repro = concatenaMediaAproRepro(df_historico)
				st.subheader('Média das Disciplinas Aprovadas e Reprovadas por Semestre')
				d = alt.Chart(df_media_apro_repro).mark_area(opacity = 0.7).encode(
				    x='Semestre',
				    y=alt.Y('Quantidade',axis=alt.Axis(title=''),scale=alt.Scale(domain=(0,10))),
					color=alt.Color('Campo', scale=alt.Scale(scheme='dark2'), 
									 legend=alt.Legend(title='Categorias',orient='bottom')),
				    row='Campo'
				).properties(
				width=200,height=150)
				
				st.altair_chart(d, use_container_width=True)



				#---------------- Carga Horaria ------------------
				st.subheader('Carga Horária do Estudante por Semestre')
				f = alt.Chart(df_historico).mark_bar().encode(
					x=alt.X('Semestre', axis=alt.Axis(title='Semestre')),
					y=alt.Y('Carga Horária', 
						axis=alt.Axis(title='Carga Horária(H)')),
		    	).configure_mark(
				    opacity=0.8,
				    color='#FCBE37'
				)
				st.altair_chart(f, use_container_width=True)

			
		except:
			pass

	else: 
		st.markdown('### Seja bem vindo ao SOAD!')
		st.markdown('')
		st.markdown('#### Para utilizar o SOAD siga os passos a seguir:')
		st.markdown('')
		st.markdown('* Se você for docente da UFRN, marque caixa de seleção')
		st.markdown('* Insira a senha de Acesso')
		st.markdown('* Selecione um departamento')
		st.markdown('* Selecione uma disciplina')
		st.markdown('* Caso docente, digite a matrícula do estudante')
		st.markdown('* Clique no botão de submissão')



if __name__ == '__main__':
	main()

