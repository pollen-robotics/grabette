from lerobot.datasets.lerobot_dataset import LeRobotDataset

for local_id, remote_id in [
        ("sim_grasp_train_v1", "SteveNguyen/sim_grasp_train_v1"),
        ("sim_grasp_eval_v1",  "SteveNguyen/sim_grasp_eval_v1"),
]:
    ds = LeRobotDataset(local_id)
    ds.repo_id = remote_id            # retarget to the namespaced repo
    ds.meta.repo_id = remote_id       # so the card / metadata match
    ds.push_to_hub(private=True, push_videos=True)
    print(f"Pushed: {remote_id}")
