# Пути к входным и выходным данным
input_path = "/path/to/input/las/files"
output_path = "/path/to/output/las/files"
log_path = "/path/to/logs"

# Параметры фильтрации
filter_params = {
    'global_filter': {
        'z_sigma_threshold': 3,  # Количество сигм для глобальной фильтрации
    },
    'local_filter': {
        'grid_size': 100,  # Размер ячеек сетки для локальной фильтрации (в метрах)
        'z_sigma_threshold': 3,  # Количество сигм для локальной фильтрации
    },
}