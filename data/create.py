import os

def create_directory(path):
    os.makedirs(path, exist_ok=True)

def create_file(path):
    with open(path, 'w') as f:
        pass  # Create an empty file

def create_project_structure():
    # Create main project directory
    project_name = "NLP_G7"
    create_directory(project_name)

    # Create subdirectories
    directories = [
        "app",
        "data/raw",
        "data/processed",
        "models/",
        "notebooks",
        "src/data",
        "src/models_definition",
        "notebooks",
        "reports/figures",       # Gráficos y visualizaciones
        "tests",
        "mlflows/mlruns"      
    ]

    for directory in directories:
        create_directory(os.path.join(project_name, directory))

    # Create Python files
    python_files = [
        "app/__init__.py",
        "src/__init__.py",
        "src/data/__init__.py",
        "src/models_definition/__init__.py",
        "notebooks/",
        "tests/__init__.py",
        "data/raw",
        "data/processed",
        "models/__init__.py",
        "mlflows/mlruns"

    ]

    for file in python_files:
        create_file(os.path.join(project_name, file))

    # # Create notebook files
    # notebook_files = [
    #     "notebooks/exploratory_climate_analysis.ipynb",
    #     "notebooks/xgboost_model_development.ipynb"
    # ]

    # for file in notebook_files:
    #     create_file(os.path.join(project_name, file))

    # # Create other files
    # other_files = [
    #     ".gitignore",
    #     "requirements.txt",
    #     "setup.py",
    #     "README.md",
    #     "config.yaml"  # For storing model parameters and data paths
    # ]

    # for file in other_files:
    #     create_file(os.path.join(project_name, file))

    # print(f"Project structure for '{project_name}' has been created.")

if __name__ == "__main__":
    create_project_structure()