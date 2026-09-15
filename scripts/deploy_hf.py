"""Explicit free-tier deployment helper. Requires `hf auth login` (write token).

Uploads only an allowlist of source directories, never .env, caches or result data.
"""
import argparse
from pathlib import Path
from huggingface_hub import HfApi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("space_id", help="YOUR_USERNAME/rag-bench-api")
    args = parser.parse_args()
    api = HfApi()
    api.whoami()  # Fail before any mutations if account setup is incomplete.
    api.create_repo(args.space_id, repo_type="space", space_sdk="docker", space_hardware="cpu-basic", exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    api.upload_folder(repo_id=args.space_id, repo_type="space", folder_path=root,
        allow_patterns=["Dockerfile", "README.md", "backend/*.py", "backend/requirements*.txt", "sample_data/*"],
        commit_message="Deploy RAG Bench FastAPI backend")
    print(f"Uploaded https://huggingface.co/spaces/{args.space_id}; configure GROQ_API_KEY and API_TOKEN as Space secrets.")


if __name__ == "__main__":
    main()
