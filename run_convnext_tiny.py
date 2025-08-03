from run_main_template import main
from src.model_convnext_tiny import MultiscaleFusionClassifier

if __name__ == '__main__':
    main(MultiscaleFusionClassifier, output_dir="./outputs/mhgf3")
