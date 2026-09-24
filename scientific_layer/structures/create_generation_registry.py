"""注册层氧体系现有五种 branch 生成策略。"""

from .generate_competing_phase_branches import generate_competing_phase_branches
from .generate_composition_branches import generate_composition_branches
from .generate_coverage_branches import generate_coverage_branches
from .generate_periodic_extension_branches import generate_periodic_extension_branches
from .generate_tm_ordering_branches import generate_tm_ordering_branches
from .generation_strategy_registry import GenerationStrategyRegistry


def create_generation_registry():
    registry = GenerationStrategyRegistry()
    registry.register("coverage", lambda context, quota, seed, options: generate_coverage_branches(context["manager"], context["frameworks"], context["phase_references"], quota=quota, seed=seed, oxidation_states=options.get("oxidation_states")))
    registry.register("composition", lambda context, quota, seed, options: generate_composition_branches(context["manager"], context["frameworks"], context["phase_references"], parent_branch_ids=context["parents"], quota=quota, seed=seed, oxidation_states=options.get("oxidation_states")) if context["parents"] else [])
    registry.register("competing_phase", lambda context, quota, seed, options: generate_competing_phase_branches(context["manager"], context["frameworks"], context["phase_references"], parent_branch_ids=context["parents"], quota=quota, seed=seed, site_mappings=options.get("site_mappings"), oxidation_states=options.get("oxidation_states")) if context["parents"] else [])
    registry.register("tm_ordering", lambda context, quota, seed, options: generate_tm_ordering_branches(context["manager"], context["phase_references"], parent_branch_ids=context["parents"], quota=quota, seed=seed, oxidation_states=options.get("oxidation_states")) if context["parents"] else [])
    registry.register("periodic_extension", lambda context, quota, seed, options: generate_periodic_extension_branches(context["manager"], context["frameworks"], context["phase_references"], parent_branch_ids=context["parents"], quota=quota, seed=seed, site_mappings=options.get("site_mappings"), oxidation_states=options.get("oxidation_states")) if context["parents"] else [])
    return registry
