import os


def load_env_file():
    """
    Parses and loads environment variables from a local .env file into os.environ.
    This zero-dependency helper eliminates duplicate env-loading scripts.
    """
    try:
        # Resolve the project root (two levels up from the shared directory)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env_path = os.path.join(project_root, ".env")

        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    # Skip empty lines and comment lines
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("'").strip('"')
                        if k:
                            os.environ[k] = v
    except Exception:
        # Fail silently to allow alternative execution environments (e.g. system env)
        pass
