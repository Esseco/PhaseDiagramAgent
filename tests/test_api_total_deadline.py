import json
import threading
import time
from unittest.mock import patch

from phase_agent.decisions.agent.create_deepseek_client import create_deepseek_client, DeepSeekResponseError
from phase_agent.runtime.turn_process import record_turn, render_process_panel


class Response:
    def __init__(self, release=None): self.release = release
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def read(self):
        if self.release: self.release.wait(3)
        return json.dumps({'choices':[{'message':{'content':'{"answer":"late"}'}}]}).encode()


def test_stalled_body_has_wall_deadline_and_no_retry():
    release = threading.Event()
    client = create_deepseek_client(api_key='test', total_timeout=.08)
    started = time.monotonic()
    try:
        with patch('urllib.request.urlopen', return_value=Response(release)) as request:
            try: client({'state':{'tasks':[{'status':'completed'}]}})
            except DeepSeekResponseError as error:
                assert error.code == 'total_timeout'
                assert '1' in str(error) and '本地' in str(error)
            else: assert False
            assert request.call_count == 1
            assert time.monotonic()-started < .5
    finally: release.set()


def test_transport_timeout_stops_without_retry():
    release = threading.Event()
    client = create_deepseek_client(api_key='test', total_timeout=.12)
    def open_response(*args, **kwargs):
        if not getattr(open_response, 'called', False):
            open_response.called = True
            time.sleep(.07)
            raise TimeoutError()
        return Response(release)
    started = time.monotonic()
    try:
        with patch('urllib.request.urlopen', side_effect=open_response) as request:
            try: client({})
            except DeepSeekResponseError as error: assert error.code == 'timeout'
            else: assert False
            assert request.call_count == 1
            assert time.monotonic()-started < .3
    finally: release.set()


def test_live_progress_and_local_fallback_preserve_state(tmp_path):
    root = tmp_path
    (root/'workflow_state').mkdir()
    state = root/'workflow_state/state.json'
    original = json.dumps({'tasks':[{'status':'completed'},{'status':'pending'}], 'pending_execution_policies':{'a':{'decision':'pending'}}})
    state.write_text(original)
    config = root/'agent_runtime.json'; config.write_text('{}')
    release = threading.Event()
    client = create_deepseek_client(api_key='test', total_timeout=.15)
    snapshot = []
    def inspect(request, **kwargs):
        snapshot.append(render_process_panel(config))
        return Response(release)
    try:
        with record_turn(config,'check','thread'), patch('urllib.request.urlopen',side_effect=inspect):
            try: client({})
            except DeepSeekResponseError as error:
                assert '已完成 1 个' in str(error)
                assert '失败 0 个' in str(error)
            else: assert False
        assert '等待模型响应' in snapshot[0]
        assert '0.15' in snapshot[0]
        assert '已完成 1 个' in snapshot[0]
        assert state.read_text() == original
    finally: release.set()


def test_invalid_deadline_rejected():
    for value in [0,-1,float('nan'),float('inf')]:
        try: create_deepseek_client(api_key='test',total_timeout=value)
        except ValueError: pass
        else: assert False
