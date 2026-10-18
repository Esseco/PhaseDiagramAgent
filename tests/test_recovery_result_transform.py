from phase_agent.tools.remote import legacy_recovery_runner as runner
from phase_agent.tools.remote.legacy_recovery import prepare_recovery_handoff,recovery_plan
from tests.test_legacy_delivery_recovery import setup_original,remote_copy,fake_result,CPU,frozen
import zipfile
import pytest

@pytest.fixture
def tmp_path(tmp_path_factory):
    return tmp_path_factory.mktemp('x')

def test_collection_transform_is_optional_and_original_is_read_only(tmp_path,monkeypatch):
    state,h=setup_original(tmp_path)
    derived=prepare_recovery_handoff(state,h,CPU)
    source,remote=remote_copy(tmp_path,h,derived)
    task=recovery_plan(state,h)['tasks'][0]
    with zipfile.ZipFile(source/'inputs.zip') as archive:fake_result(source,task,archive)
    before=frozen(source)
    observed=[]
    def factory(work,plan):
        def transform(destination,source,task,data):
            observed.append(task['expected']['task_id'])
            assert destination.is_dir()
            return {'status':'checked_copy'}
        return transform
    report=runner.collect(remote,transform_factory=factory)
    assert observed==[task['expected']['task_id']],report
    assert frozen(source)==before
    assert report['task_reassessments'][observed[0]]['status']=='checked_copy'
