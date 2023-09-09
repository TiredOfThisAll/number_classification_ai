import torch

# Проверка наличия CUDA
if torch.cuda.is_available():
    # Получение количества доступных устройств CUDA
    device_count = torch.cuda.device_count()
    print(f'Доступно устройств CUDA: {device_count}')
else:
    print('CUDA не доступно на этом компьютере.')