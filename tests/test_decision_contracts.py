import json

from phase_agent.decisions.agent.decision_contracts import PostDFTReview, review_contract_errors
from phase_agent.decisions.agent.post_dft_review import post_dft_review_errors
from tests.test_post_dft_review import review


def test_review_contract_schema_serializable_and_valid():
    json.dumps(PostDFTReview.model_json_schema())
    assert not review_contract_errors(review())


def test_strict_types_and_unknown_fields():
    data = review()
    data.update(error_assessment=123, execute=True)
    errors = review_contract_errors(data)
    assert any("error_assessment" in item for item in errors)
    assert any("execute" in item for item in errors)


def test_empty_analysis_and_invalid_enum():
    data = review()
    data.update(limitations="  ", choice="automatic_training")
    errors = review_contract_errors(data)
    assert any("limitations" in item for item in errors)
    assert any("choice" in item for item in errors)


def test_schema_does_not_replace_scientific_action_consistency():
    data = review()
    assert not review_contract_errors(data)
    assert post_dft_review_errors({"tool": "update_mlip", "post_dft_review": data})


def test_request_and_repair_share_contract_and_do_not_execute_tools():
    from phase_agent.decisions.agent.post_dft_review import request_validated_action
    requests = []
    def client(payload):
        requests.append(payload)
        data = review()
        if len(requests) == 1:
            data["limitations"] = " "
        return {"tool": "update_mlip", "post_dft_review": {**data, "choice": "finetune", "finetune_recommendation": "now"}}
    result = request_validated_action(client, {"instruction": "test", "allowed_tools": ["update_mlip"],
                                              "decision_context": {"post_dft_assessment": {"status": "evaluated"}}})
    assert len(requests) == 2
    assert requests[0]["output_contracts"] == requests[1]["output_contracts"]
    assert any("limitations" in e for e in requests[1]["validation_errors"])
    assert result["tool"] == "update_mlip"
