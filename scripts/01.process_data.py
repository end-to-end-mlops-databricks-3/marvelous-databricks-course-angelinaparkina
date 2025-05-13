# COMMAND ----
config_path = f"../project_config.yml"

config = ProjectConfig.from_yaml(config_path=config_path,env="dev")

df = pd.read_csv(f"/Volumes/{config.catalog_name}/{config.schema_name}/files/Hotel Reservations.csv",header=True)