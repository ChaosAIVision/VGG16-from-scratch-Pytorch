import yaml

class ManagerDataYaml:
    def __init__(self, yaml_path: str):
        self.yaml_path = yaml_path
        self.data = None

    def load_yaml(self) -> dict:
        """
        Load data from YAML file and return its properties as a dictionary.
        """
        try:
            with open(self.yaml_path, 'r') as file:
                self.data = yaml.safe_load(file)
                return self.data
        except Exception as e:
            return f"Error loading YAML file: {self.yaml_path}. Exception: {e}"

    def get_properties(self, key: str) -> str:
        """
        Get the value of a specific property from the loaded YAML data.
        """
        if isinstance(self.data, dict):
            if key in self.data:
                value = self.data[key]
                return str(value)
            else:
                return f"Key '{key}' not found in the data."
        else:
            return "Data has not been loaded or is not a dictionary."

if __name__ == "__main__":
    # Create an instance of ManagerDataYaml
    manager = ManagerDataYaml('/Users/chaos/Documents/Chaos_working/Chaos_projects/VGG16-from-scratch-Pytorch/data.yaml')
    
    # Load YAML data
    data = manager.load_yaml()
    
    # Get specific property
    properties = manager.get_properties('train')
    
    # Print results
    print(data)
    print(properties)
