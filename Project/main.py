from DatasetGeneratorMock import DatasetGeneratorMock
from model import create_model


def main():
    #model = create_model(4)
    dataset_generator = DatasetGeneratorMock()
    dataset_generator.decimate_and_interpolate()


main()
