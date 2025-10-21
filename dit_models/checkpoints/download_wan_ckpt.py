from huggingface_hub import snapshot_download


repo_id = "WoW-world-model/WoW-1-Wan-14B-600k"
target_local_dir = "./WoW-1-Wan-14B-600k" 


snapshot_download(
    repo_id=repo_id,
    local_dir=target_local_dir,
    local_dir_use_symlinks=False 
)
print(f"Checkpoint from {repo_id} has been downloaded to {target_local_dir}")