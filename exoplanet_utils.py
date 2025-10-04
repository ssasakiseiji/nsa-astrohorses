def load_data(file_path):
    import pandas as pd
    data = pd.read_csv(file_path)
    return data

def filter_exoplanets_by_mass(data, mass_threshold):
    return data[data['mass'] < mass_threshold]

def get_exoplanet_statistics(data):
    return {
        'total_exoplanets': len(data),
        'average_mass': data['mass'].mean(),
        'average_radius': data['radius'].mean(),
        'average_temperature': data['temperature'].mean()
    }