# Mateusz Wołowiec - CNN Animal Classification - Animals-10 Dataset

## Opis Projektu

Projekt implementuje konwolucyjną sieć neuronową (CNN) do klasyfikacji 10 różnych gatunków zwierząt. Korzysta ze zbioru danych Animals-10 dataset (https://www.kaggle.com/datasets/alessiocorrado99/animals10/data)

### Klasyfikowane Gatunki
- Pies (dog)
- Koń (horse) 
- Słoń (elephant)
- Motyl (butterfly)
- Kurczak (chicken)
- Kot (cat)
- Krowa (cow)
- Owca (sheep)
- Pająk (spider)
- Wiewiórka (squirrel)

## Struktura Projektu

```
CNNAnimalClassification/
├── main.py                 # Główny skrypt uruchamiający projekt
├── src/
│   ├── __init__.py
│   ├── data_pipeline.py    # Pipeline do przetwarzania danych
│   ├── model.py           # Architektura modelu CNN
│   ├── train_pipeline.py  # Pipeline treningu
│   └── eval_pipeline.py   # Pipeline ewaluacji
├── models/                # Zapisane modele
├── results/               # Wyniki i wizualizacje
├── animals10_data/        # Dane wejściowe
├── requirements.txt       # Zależności projektu
└── README.md             # Dokumentacja projektu
```

### Instalacja

1. Sklonuj repozytorium:
```bash
git clone git@github.com:MatWW/CNNAnimalClassification.git
cd CNNAnimalClassification
```

2. Zainstaluj zależności:
```bash
pip install -r requirements.txt
```

3. Pobierz dataset Animals-10 i foldery z obrazami przechowuj w folderze `animals10_data/raw-img/`

## Użycie

### Podstawowe Uruchomienie

```bash
python main.py
```

Domyślnie uruchomi się pełny proces treningu i ewaluacji z następującymi parametrami:
- 100 epok treningu
- Batch size: 128
- Learning rate: 0.001

### Parametry Konfiguracyjne

```bash
python main.py --mode [train|eval|both] --epochs 65 --batch_size 128 --lr 0.001 --model_path models/best_model.pth
```

**Argumenty:**
- `--mode`: Tryb działania (train/eval/both)
- `--epochs`: Liczba epok treningu
- `--batch_size`: Rozmiar batcha
- `--lr`: Szybkość uczenia
- `--model_path`: Ścieżka do zapisu/wczytania modelu

### Przykłady Użycia

**Tylko trening:**
```bash
python main.py --mode train --epochs 50 --lr 0.0005
```

**Tylko ewaluacja:**
```bash
python main.py --mode eval --model_path models/best_model.pth
```

**Dostosowane parametry:**
```bash
python main.py --epochs 100 --batch_size 64 --lr 0.002
```

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

## Pipeline'y

### 1. Data Pipeline (`data_pipeline.py`)
- Augmentacja danych (flip, rotacja, color jitter)
- Normalizacja obrazów
- Automatyczny podział na train/val/test (70/20/10%)
- Efektywne ładowanie danych z DataLoader

### 2. Train Pipeline (`train_pipeline.py`)
- Optymalizator Adam z learning rate scheduling
- Early stopping (patience=10)
- Wizualizacja krzywych uczenia

### 3. Evaluation Pipeline (`eval_pipeline.py`)
- Metryki (accuracy, precision, recall, F1)
- Macierz konfuzji
- Raport klasyfikacji
- Wizualizacja przykładowych predykcji

## Wyniki

### Metryki Performance:
- **Test Accuracy**: 85.53%
- **Precision (weighted avg)**: 86.00%
- **Recall (weighted avg)**: 85.53%
- **F1-Score (weighted avg)**: 85.62%

### Najlepsze Klasy:
1. Spider: 92.04% F1-score
2. Chicken: 90.63% F1-score
3. Butterfly: 88.58% F1-score

### Najtrudniejsze Klasy:
1. Cow: 70.19% F1-score
2. Cat: 82.97% F1-score
3. Squirrel: 80.21% F1-score

## Wizualizacje i Wyniki

Po uruchomieniu projektu zostają wygenerowane:
- `results/training_curves.png` - Krzywe uczenia
- `results/confusion_matrix.png` - Macierz konfuzji
- `results/sample_predictions.png` - Przykładowe predykcje
- `results/classification_report.txt` - Szczegółowy raport

## Optymalizacje i Techniki

- **Normalizacja danych**
- **Data Augmentation**: Flip, rotacja, color jitter
- **Batch Norm**
- **Dropout**
- **Learning Rate Scheduling**: ReduceLROnPlateau
- **Early Stopping**:
