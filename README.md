# Cocrystal Prediction

В проекте решается задача предсказания химических экспериментов, а именно со-кристаллизации молекул.

Общий пайплайн устроен так:
- обработка молекул, препроцессинг данных
- формирование признаков
- обучение моделей

В текущей версии репозитория представлены два основных подхода: градиентный бустинг и предобученный энкодер UniMol

Градиентный бустинг используется как сильный табличный baseline на квантовохимических признаках, а UniMol - как 3D-ориентированная нейросетевая модель. В текущих экспериментах UniMol выглядит более устойчивым к сдвигу домена на внешних наборах.

---

## Repository structure

```text
notebooks/
    eda.ipynb                
    xgb_model.ipynb
    unimol.ipynb
    unimol_pool.ipynb

scripts/
    triple_pos_xtb.py
    split_holdout.py
    neg_candidates.py
    triple_dataset.py
    rebuilt_xtb.py
    holdout_dataset.py
    xtb/                    

src/
    models/
        unimol_pooling.py
resources/
requirements/

```

## Окружения

В проекте использовались два отдельных окружения:

openbabel_env — для ноутбука notebooks/eda.ipynb
babel_xtb — для основного пайплайна обработки данных

Окружения для ноутбуков `notebooks/xgb_model.ipynb` и `notebooks/unimol.ipynb` будут добавлены позже.

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

## Модели
В репозитории сейчас есть два ноутбука с моделями:

```text
notebooks/xgb_model.ipynb
notebooks/unimol.ipynb
```

`xgb_model.ipynb` - baseline на табличных признаках.

`unimol.ipynb` показывает нейросетевой подход на 3D-представлениях молекул через UniMol. Этот ноутбук будет упрощён и оптимизирован.
