# Tinkoff Lab. Тестовое задание NLP

В папке `tasks` находятся Jupyter-ноутбуки с сравнением различных моделей классификации. На GitHub не отображаются выходы ячеек, поэтому ноутбуки можно посмотреть здесь: [sst-2](https://nbviewer.org/github/pkseniya/nlp-testing/blob/main/tasks/sst2.ipynb), [cola](https://nbviewer.org/github/pkseniya/nlp-testing/blob/main/tasks/cola.ipynb), [rte](https://nbviewer.org/github/pkseniya/nlp-testing/blob/main/tasks/rte_.ipynb). Их лучше всего запускать в Google Colab с GPU. Для ноутбуков sst2.ipynb и cola.ipynb нужно будет загрузить модуль `meltools.py` (monitoring learning and evaluation) из папки `tasks`.

Тестирование проводилось на валидационной выбоке, так как для нее известны истинные классы.

### Результаты:
  
  1. Accuracy на `sst-2` - 0.85
  2. Mattews correlation на `cola` - 0.17 (Accuracy 0.5)
  3. Accuracy на `rte` - 0.57

Эти результаты можно сравнить с baseline решениями авторов GLUE из статьи [GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding](https://openreview.net/pdf?id=rJ4km2R5t7). Полученные мной результаты на `sst-2` и `cola` сравнимы с BiLSTM, а accuracy на `rte` превосходит приведенную в статье для Single-Task Training. Результаты лидирующих [SOTA моделей](https://paperswithcode.com/sota) довольно сильно превосходят мои, на то они и State-of-the-Art, но встречаются модели с сравнимым качеством (для `sst-2`: GloVe + Emo2Vec - 0.82, RNTN - 0.85; для `cola`: T5-Base - 0.51 accuracy, SqueezeBERT - 0.47 accuracy; для `rte`: 24hBERT - 0.58).
