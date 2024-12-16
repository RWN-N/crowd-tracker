import os
import subprocess

def install_requirements():
    # install requirements from base project
    subprocess.check_call(["pip", "install", "-r", "requirements.txt"])

    # install requirements based on submodules
    base_path = "libs"
    for root, dirs, files in os.walk(base_path):
        if "requirements.txt" in files:
            req_path = os.path.join(root, "requirements.txt")
            print(f"Installing dependencies from: {req_path}")
            subprocess.check_call(["pip", "install", "-r", req_path])

if __name__ == "__main__":
    install_requirements()
