from phase_agent.configuration.defaults.default_layered_search_config import default_layered_search_config
from phase_agent.configuration.defaults.default_mace_committee_config import default_mace_committee_config
from phase_agent.configuration.session.answer_default_parameter_prompt import answer_default_parameter_prompt
from phase_agent.configuration.session.create_config_draft import create_config_draft
from phase_agent.configuration.session.propose_config_revision import propose_config_revision
from phase_agent.configuration.schema.validate_search_config import validate_search_config


def test_reviewed_four_member_committee_defaults_and_main_model():
    config = default_mace_committee_config(foundation_model="mace-mh-1.model")
    assert config["main_model_index"] == 0
    assert [row["loss"] for row in config["committee"]] == [
        "huber", "stress", "universal", "l1l2energyforces"
    ]
    assert [row["energy_weight"] for row in config["committee"]] == [10, 1, 10, 5]
    assert [row["forces_weight"] for row in config["committee"]] == [10, 10, 3, 5]
    assert [row["stress_weight"] for row in config["committee"]] == [1, 1, 1, 0]
    assert [row["lr"] for row in config["committee"]] == [1e-3, 7.5e-4, 5e-4, 1.25e-3]
    assert [row["seed"] for row in config["committee"]] == [2026, 2027, 2028, 2029]


def test_configuration_dialogue_asks_once_and_accepts_selective_overrides():
    draft = create_config_draft(default_layered_search_config())
    prompt = propose_config_revision(draft)
    assert prompt["status"] == "awaiting_default_parameter_choice"
    assert prompt["default_parameters"]["dft"]["parameters"] == {}
    answered = answer_default_parameter_prompt(
        draft, use_defaults=True,
        overrides={"mlip.model_path": "/models/mace-mh-1.model",
                   "run.batch_size": 6},
    )
    assert answered["default_parameter_prompt"]["status"] == "answered"
    assert answered["default_parameter_collection"]["mlip"]["model_path"].endswith("mace-mh-1.model")
    output = answered["dialogue"][-1]
    assert output["type"] == "default_parameter_collection"
    assert output["parameters"] == answered["default_parameter_collection"]
    assert answered["config"]["run"]["batch_size"] == 6
    assert propose_config_revision(answered)["status"] in {"ready_for_confirmation", "needs_user_input"}


def test_atomate_default_dft_parameters_are_a_valid_explicit_choice():
    config = default_layered_search_config()
    audit = validate_search_config(config)
    assert config["dft"]["parameter_source"] == "atomate_defaults"
    assert "dft.parameters" not in audit["missing"]
