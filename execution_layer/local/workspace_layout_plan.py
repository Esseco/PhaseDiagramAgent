"""Plan explicit local moves; no guesses about scientific stage or task identity."""
from pathlib import Path
from analysis_layer.phase.phase_snapshot_paths import model_output_directory


def workspace_layout_plan(root, state):
    root = Path(root).resolve()
    moves = []

    def add(source, destination):
        old, new = root / source, root / destination
        if old.exists() and old != new:
            moves.append((old, new))

    for old, new in (
        ("config_session.json", "config/config_session.json"),
        ("search_config.project.json", "config/search_config.project.json"),
        ("config_snapshots", "config/snapshots"),
        ("current/state.json", "runtime/state.json"),
        ("current/phase_data.json", "runtime/ledgers/phase_data.json"),
        ("current/branch_energy_pools.json", "runtime/ledgers/branch_energy_pools.json"),
        ("current/phase_identification_cache.json", "runtime/cache/phase_identification_cache.json"),
        ("current/candidate_structures", "inputs/candidate_structures"),
        ("current/approvals", "runtime/approvals"),
        ("current/approved_batches", "runtime/approved_batches"),
        ("current/dft_template_review", "runtime/template_reviews/dft"),
        ("current/work", "runtime/work"), ("current/qbc", "runtime/cache/qbc"),
        ("upload_batches/RELAX_UPLOAD_PLAN.json", "runtime/submission_plans/RELAX_UPLOAD_PLAN.json"),
        ("agent_server.log", "logs/legacy_root/agent_server.log"),
        ("open_webui_server.log", "logs/legacy_root/open_webui_server.log"),
        ("OUTPUTS.md", "docs/legacy_OUTPUTS.md"),
        ("outputs/README.md", "docs/legacy_outputs_README.md"),
        ("outputs/legacy", "backups/legacy_outputs"),
        ("outputs/output_index.csv", "backups/legacy_indexes/output_index.csv"),
    ):
        add(old, new)
    planned = {source for source, _ in moves}
    current = root / "current"
    if current.is_dir():
        for path in current.iterdir():
            if path not in planned:
                # Old state backups/unknown records are retained, not merged into
                # live state or treated as new analysis.
                add(path.relative_to(root), Path("backups/legacy_current") / path.name)
    models = (state.get("upload_layout") or {}).get("model_rounds") or {}
    for version in models:
        old = model_output_directory(root / "outputs", version)
        new = model_output_directory(root / "outputs", version, state=state)
        if old.is_dir():
            for child in old.iterdir():
                if child.name == "phase_diagrams":
                    destination = new / "phase_diagrams/mlip"
                elif child.name == "combined":
                    phase = child / "phase_diagrams"
                    add(phase.relative_to(root), (new / "phase_diagrams/combined").relative_to(root))
                    continue
                else:
                    destination = new / child.name
                add(child.relative_to(root), destination.relative_to(root))
    active = state.get("active_model_version")
    dft = root / "outputs/dft/phase_diagrams"
    if dft.is_dir():
        if active not in models:
            raise ValueError("DFT 输出缺少当前 epoch 登记，不能猜测归属")
        destination = model_output_directory(root / "outputs", active, state=state) / "phase_diagrams/dft"
        add(dft.relative_to(root), destination.relative_to(root))
    for source, target in moves:
        if source.is_symlink() or not source.resolve().is_relative_to(root) or not target.resolve().is_relative_to(root):
            raise ValueError(f"迁移路径超出工作区或为链接：{source}")
        if source.is_dir() and any(path.is_symlink() for path in source.rglob("*")):
            raise ValueError(f"目录包含链接，不自动迁移：{source}")
        if target.exists():
            raise FileExistsError(f"迁移目标已存在，不合并或覆盖：{target}")
    destinations = [target for _, target in moves]
    if len(destinations) != len(set(destinations)):
        raise ValueError("迁移目标重复")
    return moves
