## Скрытые Марковские Модели Переменного Порядка (СММПП)
#### Variable Length Markov Models (VLHMM)

Репозиторий содержит два модуля *vlhmm*, *chipseq*
* *vlhmm* включает в себя инструменты для решения следующих задач:
    * построение, обучение, отрисовка контекстных деревьев переходов
    * построение, обучение СММПП, графическое представление результатов
* *chipseq* позволяет решить следующие задачи:
    * применение СММПП к данным ChIp-seq
    * создание файлов для просмотра результатов в геномном браузере http://genome.ucsc.edu/


Язык: Python 3.x

Зависимости:
* Cython
* NumPy, SciPy
* Pylab
* datrie
* PyGraphviz

Инсталляция:

    git clone https://github.com/atanna/hmm.git

Компиляция Cython-файлов:

    python setup.py build_ext --inplace
___
## Примеры
Директория `vlhmm_/examples/` содержит примеры тестов, иллюстрирующих обучение СММПП и контекстных деревьев на симулированных данных.

`chipseq/real_test.py` иллюстрирует пример работы СММПП на данных ChIP-seq эксперимента
_ _ _
## Суть
##### Контекстные деревья переходов
* Контекстное дерево переходов задает случайный процесс.
* Контекст состояния процесса -- любой префикс из предшествующих состояний (процесс движется справа налево, т.е. состояния идут по убыванию времени).
* Вершина ~ контекст.
* Ребро ~ состояние.
* Исходящая степень внутренней вершины -- число состояний.
* Лист задает распределение текущего состояния.
* ###### Примеры
    1. "Неравновероятная монетка"<br>
![alt text](https://raw.githubusercontent.com/atanna/hmm/master/diploma/img/sample_mixture/real_trie_.png)

    2. Марковский процесс<br>
![alt text](https://raw.githubusercontent.com/atanna/hmm/master/diploma/img/sample_hmm1/real_trie_.png)

    3. Марковский процесс второрго порядка<br>
![alt text](https://raw.githubusercontent.com/atanna/hmm/master/diploma/img/Context_trie.png)
    4. Марковский процесс переменного порядка (эквивалентно 3.)
![alt text](https://raw.githubusercontent.com/atanna/hmm/master/diploma/img/Prune_c_trie.png)
* Контекстное дерево переходов задает Марковский процесс переменного порядка.

##### СММПП
* Аналогичны СММ.
* Скрытый слой задается Марковским процессом переменного порядка (контекстным деревом переходов)

