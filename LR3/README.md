# Лабораторная работа №3: Спектрограмма сигнала

## Цель работы

Реализовать вычисление спектрограммы звукового сигнала с использованием кратковременного преобразования Фурье (STFT) на CPU и GPU, сравнить производительность и корректность результатов.

## Технические характеристики

| Компонент | Характеристики |
|-----------|----------------|
| **GPU** | NVIDIA Tesla T4 |
| **GPU память** | 14.6 GB |
| **Библиотеки** | numpy, scipy, torch, matplotlib, PIL |
## Обзор функций кода

### 1. Генерация тестового сигнала

```python
def generate_chirp_signal(duration=3.0, sample_rate=44100, f0=50, f1=10000):
    """Генерирует chirp сигнал с линейной развёрткой частоты"""
    t = np.linspace(0, duration, int(sample_rate * duration))
    signal = scipy.signal.chirp(t, f0, duration, f1, method='linear')
    return signal.astype(np.float32), sample_rate
```


Создаёт тестовый сигнал типа chirp (линейная развёртка частоты) — частота возрастает от 50 Гц до 10 кГц. Такой сигнал идеально подходит для проверки корректности спектрограммы: на ней должна быть видна диагональная линия.

### 2. Спектрограмма на CPU

```python
def compute_spectrogram_cpu(signal_data, sample_rate, window_size=4096, hop_size=1024):
    start = time.perf_counter()
    frequencies, times, spectrogram = scipy.signal.spectrogram(
        signal_data, fs=sample_rate, window='hann',
        nperseg=window_size, noverlap=window_size - hop_size, mode='psd'
    )
    end = time.perf_counter()
    return spectrogram, frequencies, times, end - start
```
Использует библиотеку SciPy для вычисления STFT. Параметры:
- window_size = 4096 — размер окна (количество сэмплов на один фрагмент)
- hop_size = 1024 — шаг между окнами (перекрытие 75%)
- оконная функция — Ханна (hann) для уменьшения спектральных утечек

### 3. Спектрограмма на GPU
```python
def compute_spectrogram_gpu_vectorized(signal_data, sample_rate, window_size=4096, hop_size=1024):
    device = torch.device('cuda')
    signal_gpu = torch.from_numpy(signal_data.astype(np.float32)).to(device)
    
    segments = signal_gpu.unfold(0, window_size, hop_size)
    
    window = torch.from_numpy(np.hanning(window_size).astype(np.float32)).to(device)
    segments_windowed = segments * window
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    fft_result = torch.fft.rfft(segments_windowed, dim=1)
    spectrogram = (fft_result.abs() ** 2).T.cpu().numpy()
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    return spectrogram, frequencies, times, end - start
```
Функция принимает звуковой сигнал и вычисляет его спектрограмму с использованием GPU (видеокарты). Весь процесс происходит на видеокарте: разбиение сигнала на сегменты, применение оконной функции и быстрое преобразование Фурье (БПФ).

### Таблица результатов

| Длительность (с) | Сегментов | CPU время (с) | GPU время (с) | Ускорение (x) |
|-------------------|-----------|---------------|---------------|---------------|
| 30 | 1288 | 0.046138 | 0.003269 | **14.11** |
| 60 | 2580 | 0.116237 | 0.005509 | **21.10** |
| 120 | 5164 | 0.278762 | 0.011836 | **23.55** |

### Анализ результатов

С увеличением длительности сигнала с 30 до 120 секунд ускорение выросло с 14.11 до 23.55. Это объясняется тем, что накладные расходы на передачу данных и запуск ядра распределяются на больший объём вычислений — чем больше сегментов, тем эффективнее используется GPU.

CPU время растёт пропорционально количеству сегментов: при увеличении длительности в 4 раза время выросло с 0.046 до 0.279 секунды. GPU время растёт значительно медленнее — с 0.0033 до 0.0118 секунды, что демонстрирует преимущество параллельной обработки.

Для 2-минутного аудиосигнала GPU обрабатывает данные за 12 миллисекунд, тогда как CPU требуется 279 миллисекунд. Такая разница критически важна для систем реального времени, где GPU позволяет обрабатывать аудиопоток без заметных задержек.