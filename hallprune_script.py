import os
import subprocess
from dotenv import load_dotenv
from huggingface_hub import HfApi, login, whoami

# Load environment variables from .env
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in environment or .env file.")

# Authenticate
login(token=HF_TOKEN)

try:
    user_info = whoami()
    print(f"✅ Successfully logged in as: {user_info['name']}")
except Exception as e:
    print("❌ Login failed! Error:", e)
    exit(1)

# Configuration
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
PERCDAMP = "0.01"            # must match the SparseGPT baseline
HALL_ALPHA = "1.0"           # relative to mean(diag(H)); sweep over orders of magnitude
HALL_ALPHA_SCALE = "mean_diag"
HALL_DIRECTION = "probe"     # "random" for the random-direction control
RUN_NAME = f"hallprune-{HALL_DIRECTION}-a{HALL_ALPHA}-{HALL_ALPHA_SCALE}-llama-3.1-8b"
OUTPUT_DIR = f"./hallprune/src/out/{RUN_NAME}"
REPO_ID = f"{user_info['name']}/{RUN_NAME}"

# Step 1: Execute Pruning Script
print("\n Starting Pruning Process...")
prune_command = [
    "python", "./hallprune/src/main.py",
    "--model", MODEL_ID,
    "--prune_method", "hallprune",
    "--sparsity_ratio", "0.5",
    "--sparsity_type", "unstructured",
    "--percdamp", PERCDAMP,
    "--hall_alpha", HALL_ALPHA,
    "--hall_alpha_scale", HALL_ALPHA_SCALE,
    "--hall_direction", HALL_DIRECTION,
    "--save", OUTPUT_DIR,
    "--save_model", OUTPUT_DIR
]

subprocess.run(prune_command, check=True)

# Step 2: Upload to Hugging Face Hub
print(f"\n📦 Creating repository: {REPO_ID}...")
api = HfApi()
api.create_repo(repo_id=REPO_ID, private=True, exist_ok=True)

print("🚀 Uploading pruned model to Hugging Face...")
api.upload_folder(
    folder_path=OUTPUT_DIR,
    repo_id=REPO_ID,
    repo_type="model",
)

print("🎉 Model successfully pruned and uploaded!")