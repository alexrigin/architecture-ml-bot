# Проектная работа №7

## Задание 1. Исследование моделей и инфраструктуры

Выбранная модель эмбеддинга: **all-MiniLM-L6-v2**
Векторная база: **ChromaDB**
Llm: **DeepSeek**

## Задание 2. Подготовка базы знаний

При подготовке базы за основу была взята вселенная матрицы.

- скрипт с загрузкой базы и логикой подмены -
[knowledge_base_script.ipynb](./scripts/knowledge_base_script.ipynb)

- база с 30 файлами - [knowledge_base](./knowledge_base/)

- словарь замен [terms_map.json:](./knowledge_base/0_terms_map.json)

```json
{
    "Neo": "Meo",
    "Anderson": "Meowdersnoot",
    "Thomas": "Jerry",
    "Trinity": "Tri-Cat-ty",
    "Morpheus": "Purrpheus",
    "Oracle": "Sphinx",
    "Architect": "Head Bengal",
    "Keymaker": "Claw who Makes Ways",
    "Smith": "Hiss",
    "Tank": "Litter-Tender",
    "Merovingian": "Meowingian",
    "Twins": "Tabbies",
    "World": "Outside",
    "Zion": "Great Scratching Post",
    "Matrix": "Catrix",
    "Rabbit": "Fish",
    "Nebuchadnezzar": "The Meowchadnezzar",
    "Logos": "Purrlogos",
    "Hammer": "Pawmer",
    "Red Pill": "Silvervine Stick",
    "Blue Pill": "Saucer of Warm Milk",
    "Agents": "V.E.T.s"
}
```

## Задание 3. Создание векторного индекса базы знаний

Скрипт для создания индекса в Chroma - [indexing.ipynb](./scripts/indexing.ipynb)

Путь к индексу - [./ml-bot/chroma_langchain_db/](./ml-bot/chroma_langchain_db/)

Модель - **all-MiniLM-L6-v2**
База знаний - **Catrix** ( [knowledge_base](./knowledge_base/))
Сколько чанков в индексе - **150**
Сколько времени заняла генерация - **15.7 s**

Пример запроса к индексу:

![Примера запроса к индексу](./screens/query_chroma_example.png "query_chroma_example")

## Задание 4. Реализация RAG-бота с техниками промптинга

### Успешные диалоги

#### zero-shot

№1
![#1](./screens/success_example_1.png )

№2
![#2](./screens/success_example_2.png )

#### few-shot
![#3](./screens/success_example_4_few_shot.png)

#### CoT
![#4](./screens/success_example_5_cot.png )


### Don't know
№1
![#5](./screens/dont_know_example_1.png )

№2
![#6](./screens/dont_know_example_2.png )


## Задание 5. Запуск и демонстрация работы бота

Была реализовано две проверки:
- проверка при загрузке документов из индекса. если встречаются запрещенные фразы, то запрос полностью отклоняется.
- проверка пользовательского ввода. Если в запросе встречаются запрещенный фразы, то запрос пользователя отклоняется

### Пример
#### Защита отключена:
№1
![#7](./screens/injection_succes_example_1.png )
№2
![#7](./screens/injection_succes_example_2.png )

#### Защита включена (на примере тех же запросов):
№1
![#8](./screens/injection_catched_example_1.png)
№2
![#10](./screens/injection_catched_example_2.png )