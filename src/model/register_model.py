import mlflow
from mlflow.tracking import MlflowClient
import json
import logging
from dotenv import load_dotenv
import dagshub

load_dotenv()
logging.basicConfig(level=logging.INFO)

dagshub.init(repo_owner="mailmohsin747", repo_name="Mlops_practice", mlflow=True)

def load_model_info(path="reports/experiment_info.json"):
    with open(path, 'r') as f:
        return json.load(f)

def register_model_to_dagshub(model_info, model_name):
    client = MlflowClient()
    run_id = model_info["run_id"]
    model_uri = f"runs:/{run_id}/model"

    logging.info(f"📌 Creating model '{model_name}' if not exists...")
    try:
        client.create_registered_model(model_name)
    except:
        logging.info(f"Model '{model_name}' already exists.")

    logging.info("📌 Creating a new version in registry...")
    mv = client.create_model_version(
        name=model_name,
        source=model_uri,
        run_id=run_id
    )

    logging.info(f"✅ Registered Version {mv.version} for model '{model_name}'")

    client.transition_model_version_stage(
        name=model_name,
        version=mv.version,
        stage="Staging"
    )
    logging.info("✅ Moved to stage: Staging")

def main():
    model_info = load_model_info()
    register_model_to_dagshub(model_info, "my_best_model_03")

if __name__ == "__main__":
    main()
