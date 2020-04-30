import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def normalizar_dados(dataset, inicio_index, fim_index, tamanho_historico, tamanho_previsao):
  data = []
  labels = []

  inicio_index = inicio_index + tamanho_historico
  if fim_index is None:
    fim_index = len(dataset) - tamanho_previsao

  for i in range(inicio_index, fim_index):
    indices = range(i-tamanho_historico, i)
    data.append(np.reshape(dataset[indices], (tamanho_historico, 1)))
    labels.append(dataset[i+tamanho_previsao])
  return np.array(data), np.array(labels)

def criar_lista_periodos(tamanho):
  return list(range(-tamanho, 0))

def mostrar_pontos(plot_data, delta, titulo):
  labels = ['Historico de temperaturas', 'Temperatura real', 'Temperatura prevista']
  marcador = ['.-', 'rx', 'go']
  time_steps = criar_lista_periodos(plot_data[0].shape[0])
  if delta:
    futuro = delta
  else:
    futuro = 0

  plt.title(titulo)
  for i, x in enumerate(plot_data):
    if i:
      plt.plot(futuro, plot_data[i], marcador[i], markersize=10,
               label=labels[i])
    else:
      plt.plot(time_steps, plot_data[i].flatten(), marcador[i], label=labels[i])
  plt.legend()
  plt.xlim([time_steps[0], (futuro+5)*2])
  plt.xlabel('Periodos de tempo')
  return plt

modelo_treinado = tf.keras.models.load_model('modelo-prever-tempo.h5')
arquivo_csv = pd.read_csv("dados-meterologicos-brasilia.csv")

dados_csv = arquivo_csv['Temperatura']
dados_csv.index = arquivo_csv['Horario']
dados_csv = dados_csv.values

historio_temperatura = 720
prever_qtd_periodos_a_frente = 0

periodo, temperatura = normalizar_dados(uni_data, 22050, None,
                                       historio_temperatura,
                                       prever_qtd_periodos_a_frente)


dados_prontos = tf.data.Dataset.from_tensor_slices((periodo, temperatura))
dados_prontos = dados_prontos.batch(256).repeat()

for x, y in val_univariate.take(10):
  plot = mostrar_pontos([x[0].numpy(), y[0].numpy(),
                    modelo_treinado.predict(x)[0]], 0, 'Modelo LSTM')
  plot.show()