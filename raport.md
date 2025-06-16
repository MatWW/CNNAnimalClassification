# Mateusz Wołowiec - Raport Projektu: CNN Animal Classification - Animals-10 Dataset

## 1. Opis Danych Wejściowych

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

## 2. Strategia Podziału Danych

### Podział: 70% / 20% / 10%
- **Training set**: 70% (18,354 obrazów) - główny zestaw do uczenia parametrów
- **Validation set**: 20% (5,244 obrazów) - monitorowanie postępu, zapisywanie najlepszego modelu i early stopping
- **Test set**: 10% (2,619 obrazów) - końcowa ewaluacja modelu

### Augmentacja Danych:
- **Random Horizontal Flip (p=0.5)**: Zwiększenie różnorodności
- **Random Rotation (±10°)**: Odporność na orientację 
- **Color Jitter**: Odporność na różne warunki oświetleniowe
- **Normalizacja**: Standaryzacja do zakresu [-1, 1]

## 3. Model i techniki

## Architektura Modelu

Model AnimalCNN składa się z:
- **Feature Extractor**: 4 bloki konwolucyjne z BatchNorm i MaxPooling
- **Classifier**: 3-warstwowa sieć fully-connected z Dropout
- **Parametry**: ~21.5M parametrów trenowalnych

### Szczegóły Architektury:
```
Conv Block 1: 3→64 channels (128x128 → 64x64)
Conv Block 2: 64→128 channels (64x64 → 32x32)  
Conv Block 3: 128→256 channels (32x32 → 16x16)
Conv Block 4: 256→512 channels (16x16 → 8x8)
FC Layers: 512*8*8 → 512 → 256 → 10
```  
Sieci konwolucyjne są naturalnym wyborem dla klasyfikacji obrazów. Progresywne zwiększanie liczby filtrów (64→128→256→512) pozwala na hierarchiczne uczenie się cech.

## Optymalizacje i Techniki

- **Data Augmentation**: Flip, rotacja, color jitter - Zwiększa różnorodność datasetu poprzez transformacje obrazów, co poprawia generalizację modelu i redukuje overfitting.
- **Batch Norm** - Stabilizuje proces treningu poprzez normalizację wejść do każdej warstwy
- **Max Pooling** - Redukuje wymiarowość przestrzenną i zmniejsza liczbę parametrów.
- **Dropout** - Zapobiega overfittingowi poprzez losowe wyłączanie neuronów
- **Optimizer Adam z Learning Rate Scheduling**: ReduceLROnPlateau - Automatycznie zmniejsza learning rate gdy accuracy przestaje się poprawiać. Pozwala na fine-tuning w późniejszych fazach treningu
- **CELoss**: jako funkcja straty
- **Early Stopping**: Zatrzymuje trening gdy model zaczyna się przeuczać

## 4. Wyniki Testowe i Treningowe

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
Model osiągnął bardzo dobre wyniki (85.53% accuracy), ale istnieje potencjał do dalszej poprawy, szczególnie dla klas cow, sheep i cat. Można rozważyć
wykorzystanie transfer learningu.
