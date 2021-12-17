from DatasetGenerator import DatasetGenerator
from model import create_model


def main():
    #model = create_model(4)
    dataset_generator = DatasetGenerator()
    dataset_generator.generate_dataset()


main()
