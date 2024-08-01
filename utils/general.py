import yaml


def load_yaml(yaml_path: str) -> dict:
    """
    Load  data YAML and return it's properties as dictionnary

    """
    try:
        with open(yaml_path, 'r') as file:
            data = yaml.safe_load(file)
            return data
    except:
            return (f"Error loading YAML file: {yaml_path}")


def get_properties(data: dict, key: str) -> str:
    """
    Get the value of a specific property from a dictionary.

    :param data: A dictionary containing properties
    :param key: The key for which to retrieve the value
    :return: The value associated with the key as a string, or a message if the key is not found
    """
    if isinstance(data, dict):
        if key in data:
            value = data[key]
            return str(value)
        else:
            return f"Key '{key}' not found in the data.yaml."

if __name__ == "__main__":
    data = load_yaml('/Users/chaos/Documents/Chaos_working/Chaos_projects/VGG16-from-scratch-Pytorch/dat1a.yaml')
    proerties = get_properties(data,'train')
    print(data)
