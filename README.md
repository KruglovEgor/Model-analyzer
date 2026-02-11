# Model Analyzer

Простое десктоп приложение для проверки YOLO моделей по датасету и вывода метрик.

## Установка

### Вариант 1: Использовать готовый .exe (рекомендуется)

Скачайте `ModelAnalyzer.exe` из папки `dist/` — работает без установки Python.

### Вариант 2: Запуск из исходников

1. Создать и активировать виртуальное окружение.
2. Установить зависимости:

```bash
pip install -r requirements.txt
```

**Важно**: Если возникает ошибка `ModuleNotFoundError: No module named 'tkinter'`:
- **Windows**: переустановите Python с галочкой "tcl/tk and IDLE"
- **Linux**: выполните `sudo apt-get install python3-tk`
- **macOS**: обычно tkinter уже установлен

## Запуск

### Из exe:
```bash
.\dist\ModelAnalyzer.exe
```

### Из исходников:
```bash
python main.py
```

## Что нужно для работы

- Файл модели `.pt` или `.onnx`.
- YAML файл датасета с путями к `train/val`.

## Метрики

- Среднее время инференса на изображение.
- Precision, Recall.
- mAP@0.5 и mAP@0.5:0.95.
- Количество правильных и ложных срабатываний.

## Сборка exe

Используется PyInstaller
```bash
python -m PyInstaller ModelAnalyzer.spec
```

