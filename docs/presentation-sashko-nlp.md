# Проезентація Олександра Михайлика: Кваліфікаційна 


## Вступ

Розуміння емоційного тону текстових даних має вирішальне значення для:
- аналізу соціальних медіа
- аналізу відгуків клієнтів
- моніторинг психічного здоров'я

Традиційні методи аналізу настроїв в тексті: 
- часто виявляються недостатніми, оскільки базуються на бінарних або категоріальних класифікаціях, яким бракує деталізації, необхідної для охоплення всього спектру людських емоцій.

Ідея: використання нечіткої логіки для покращення виявлення та інтерпретації настроїв у тексті, що забезпечує більш тонкий і гнучкий підхід.


## Опис Алгоритму

(Сюди було б круто кинути діаграмку)

- Попередня обробка тексту
- Створення кастомниг ембедингів слів
- Відбір ключевих ознак.
- Генетичний алгоритм для добору систем правил нечіткої логіки.



## Визначення Основних Компонент

Ми застосували алгоритм випадкового лісу ("random forest") для готових векторних вкладень слів для визначення основних компонент, що відповідають за емоційні характеристики. 

(100-300 розмірні вкладення, які компоненти цих векторів відповідають за те, що нас цікавить?)

(Random forest: ансамблева модель, конструювання великої кількості дерев прийняття рішень, інтерпретовність)


## Застосування Алгоритму Опорних Векторів

Відбіриаємо компоненти із попереднього пункту виконання алгоритму, по них застосовуємо алгоритм опорних векторів.


## Нечітка Логіка

Нечітка логіка пропонує надійну основу для управління неоднозначністю та частковими істинами, що робить її ідеальним інструментом для виявлення настрою в тексті
часткова приналежність до декількох емоційних станів

- один вираз може належати до декількох категорій у різній мірі

```
[
            "angry medium and worried medium and sad medium and calm medium and happy medium and excited medium then neutral",
            "angry high and sad high then high_negative",
            "happy low and happy medium then high_negative",
            "happy high and worried high then high_positive",
            "angry high and worried high then high_negative",
            "happy low and calm high then high_negative",
            "sad high and happy high then positive",
            "worried low and sad low then high_positive",
            "calm low and calm high then neutral"
        ]
```


## Застосування Еволюційних Алгоритмів 

Ми застосували еволюційний алгоритм для відбору систем правил нечіткої логіки.

(Тут скинути графік MSE over time)



## Результати Виконання Алгоритму 

Результати

## Напрямок Подальших Досліджень


## Список Джерел






