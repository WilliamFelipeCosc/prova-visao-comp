# prova-visao-comp

O código desse reposotório se destina a aplicar certos filtros de imagens, sendo estes (redimensionamento para 128 x 128), GaussianBlur e Histograma em imagens de animais, e também, o treinamento de um modelo para reconhecer cães e gatos.

1. Como executar:
  Primeiro execute `pip install -r "requirements.txt"` e depois `python main.py`

2. Para os filtros, a biblioteca cv2 está sendo usada, principalmente as funções: imread, resize, GaussianBlur e EqualizeHist, essas funções são usadas no pipeline de filtro para, respectivamente: ler imagem, redimensionar, aplicar Blur Gaussiano e aplicar histograma. Após isso a biblioteca matplotlib es'ta sendo usada para mostrar as imagens e filtros.

3.
