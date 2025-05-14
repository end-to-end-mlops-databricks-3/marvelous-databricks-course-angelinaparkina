# COMMAND ----
import pandas as pd
from hotel_reservations.config import ProjectConfig

config_path = f"../project_config.yml"

config = ProjectConfig.from_yaml(config_path=config_path,env="dev")

df = pd.read_csv(f"/Volumes/{config.catalog_name}/{config.schema_name}/files/Hotel Reservations.csv",header=True)