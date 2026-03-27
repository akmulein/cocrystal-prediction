# Cocrystal Prediction

В проекте решается задача предсказания со-кристаллизации молекул.  
В текущей версии репозитория опубликована стадия **подготовки и обработки данных**.  
Обучение моделей и финальный анализ в данный репозиторий пока не включены.

---

## Repository structure

```text
notebooks/
    eda.ipynb                

scripts/
    triple_pos_xtb.py
    split_holdout.py
    neg_candidates.py
    triple_dataset.py
    rebuilt_xtb.py
    holdout_dataset.py
    xtb/                    

src/
resources/
requirements/

```

## Окружения

В проекте использовались два отдельных окружения:

openbabel_env — для ноутбука notebooks/eda.ipynb
babel_xtb — для основного пайплайна обработки данных

Зависимости сохранены в папке requirements/:

```
requirements/
    openbabel_env.txt
    babel_xtb.txt
```

# Последовательность выполнения
## Первичный EDA и сохранение датасета
```
python -m venv venv
source venv/bin/activate
pip install -r requirements/openbabel_env.txt
conda activate openbabel_env
```
Выполнить:
notebooks/eda.ipynb

## Обработка датасета, подготовка к обучению
```
python -m venv venv
source venv/bin/activate
pip install -r requirements/babel_xtb.txt
conda activate babel_xtb

python -m scripts.triple_pos_xtb    #Генерация положительных троек с XTB-признаками
python -m scripts.split_holdout     # Выдление holdout теста
python -m scripts.neg_candidates    # Генерация негативных кандидатов
python -m scripts.triple_dataset    # Построение основного датасета 
python -m scripts.rebuilt_xtb       # Пересчёт XTB-признаков
python -m scripts.holdout_dataset   # Подготовка holdout-датасета
```

