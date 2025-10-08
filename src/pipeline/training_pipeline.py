from src.component.data_ingestion import DataIngestion
from src.component.data_transformation import DataTransformation
from src.component.model_trainer import ModelTraining

def main():
    ingest_obj = DataIngestion()
    train_path, test_path = ingest_obj.initiate_data_ingestion()

    transform_obj = DataTransformation()
    train_arr, test_arr, _ = transform_obj.initiate_data_transformation(train_path, test_path)

    model_obj = ModelTraining()
    mod, res = model_obj.model_trainer(train_arr, test_arr)

    print(f"\nThe best model: {mod}")
    print(f"\nR-squared score: {round(res, 3)}")

if __name__ == "__main__":
    main()
