from run_main_template import main
from src.model_resnet50 import MultiscaleFusionClassifier

if __name__ == '__main__':
    main(MultiscaleFusionClassifier, output_dir="./outputs/mhgf2")
