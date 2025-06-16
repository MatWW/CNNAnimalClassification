# Raport Projektu: CNN Animal Classification - Animals-10 Dataset

## 1. Wyniki Testowe i Treningowe

### Wyniki Finalne:
- **Test Accuracy**: 85.53% (2240/2619 poprawnych predykcji)
- **Precision (weighted avg)**: 86.00%
- **Recall (weighted avg)**: 85.53%  
- **F1-Score (weighted avg)**: 85.62%

### Szczegółowe Wyniki per Klasa:
| Klasa | Precision | Recall | F1-Score | Support |
|-------|-----------|---------|----------|---------|
| Dog | 85.96% | 88.02% | 86.98% | 501 |
| Horse | 91.27% | 76.84% | 83.43% | 272 |
| Elephant | 90.14% | 83.66% | 86.78% | 153 |
| Butterfly | 90.65% | 86.61% | 88.58% | 224 |
| Chicken | 90.17% | 91.10% | 90.63% | 292 |
| Cat | 87.58% | 78.82% | 82.97% | 170 |
| Cow | 65.18% | 76.04% | 70.19% | 192 |
| Sheep | 78.45% | 81.61% | 80.00% | 174 |
| Spider | 91.45% | 92.64% | 92.04% | 462 |
| Squirrel | 76.00% | 84.92% | 80.21% | 179 |

### Analiza Treningu:
Model trenowany przez 100 epok z early stopping (patience=10).  
Najlepszy wynik walidacyjny osiągnięty w 54 epoce. Training accuracy osiągnęła ~95%, podczas gdy validation accuracy ustabilizowała się na ~85%.  
Po 64 epoce trening został przerwany.   
Dokładne logi z treningu można zobaczyć w katalogu other w pliku logs.txt lub logs.pdf.  

## 2. Uzasadnienie Wyboru Techniki/Modelu

## 3. Strategia Podziału Danych

### Podział: 70% / 20% / 10%
- **Training set**: 70% (18,354 obrazów) - główny zestaw do uczenia parametrów
- **Validation set**: 20% (5,244 obrazów) - monitorowanie postępu, zapisywanie najlepszego modelu i early stopping
- **Test set**: 10% (2,619 obrazów) - końcowa ewaluacja modelu

### Augmentacja Danych:
- **Random Horizontal Flip (p=0.5)**: Zwiększenie różnorodności
- **Random Rotation (±10°)**: Odporność na orientację 
- **Color Jitter**: Odporność na różne warunki oświetleniowe
- **Normalizacja**: Standaryzacja do zakresu [-1, 1]

## 4. Opis Danych Wejściowych

### Dataset: Animals-10
- **Źródło**: Kaggle Animals-10 Dataset
- **Opis**: Animal pictures of 10 different categories taken from google images
- **Wielkość**: ~26,000 obrazów w 10 klasach
- **Format**: Obrazy kolorowe w różnych rozdzielczościach
- **Klasy**: dog, horse, elephant, butterfly, chicken, cat, cow, sheep, spider, squirrel

### Preprocessing:
- **Resize**: Wszystkie obrazy przeskalowane do 128x128 pikseli
- **Normalizacja**: Pixel values znormalizowane do [-1, 1]
- **Typ**: RGB (3 kanały)
- **Batch size**: 128 obrazów na batch

### Charakterystyka Danych:
- **Nierównomierne rozkłady**: Niektóre klasy mają więcej próbek (dog: 501 vs elephant: 153)
- **Różnorodność**: Obrazy w różnych pozach, oświetleniu i tle
- **Jakość**: Wysokiej jakości obrazy z wyraźnymi obiektami

## 5. Analiza Wyników
