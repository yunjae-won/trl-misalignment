# Container Restore

This file records the persistent backup made before pausing the container on
2026-04-28.

Backup root:

```bash
/yj_data/trl_misalignment/container_backup_20260428_101941
```

Restore from a fresh container with:

```bash
bash /yj_data/trl_misalignment/container_backup_20260428_101941/root/trl-misalignment/scripts/restore_container_from_yj_data.sh /yj_data/trl_misalignment/container_backup_20260428_101941
```

For a stricter package reinstall from the portable pip snapshot:

```bash
RESTORE_EXACT_PIP=1 bash /yj_data/trl_misalignment/container_backup_20260428_101941/root/trl-misalignment/scripts/restore_container_from_yj_data.sh /yj_data/trl_misalignment/container_backup_20260428_101941
```

The backup contains:

- `/root/trl-misalignment`, including experiment outputs and local checkpoints.
- `/root/logprob-engine`.
- `/root/.cache`, including Hugging Face, pip, W&B, uv, vLLM, and small runtime caches.
- Selected `/root` configuration under `private_root_home/`, including SSH, `.netrc`,
  shell config, Codex state, and editor/runtime settings.
- Environment manifests and git bundles under `manifests/`.

Symlink handling:

- Backup copies were made with symlinks dereferenced.
- Verified backup symlink count: `0`.
- Verified backed-up checkpoint directories: `24`.
- Verified backed-up Hugging Face cache symlink count: `0`.

The Conda installation itself is intentionally not backed up. Recreate the base
runtime in the restarted container, then use the restore script to put the
project data and Python packages back in place. The recorded environment details
are in:

```bash
/yj_data/trl_misalignment/container_backup_20260428_101941/manifests
```

Important files:

- `python-env.json`
- `requirements-portable.txt`
- `pip-freeze-all.txt`
- `pip-list.json`
- `pip-list.txt`

Treat `private_root_home/` as sensitive because it may contain credentials or
SSH material.
