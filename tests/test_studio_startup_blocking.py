"""Exercise application startup under Studio's blocking-call detector, without services."""
import asyncio
import threading
from unittest.mock import Mock, AsyncMock
from blockbuster import blockbuster_ctx
from phase_agent.runtime import studio_runtime


def test_startup_with_blocking_detector(tmp_path, monkeypatch):
    config = tmp_path / 'agent_runtime.json'
    config.write_text('{}', encoding='utf-8')
    monkeypatch.setenv('PHASE_AGENT_RUNTIME_CONFIG', str(config))
    control = Mock()
    monkeypatch.setattr('phase_agent.runtime.agent_api.create_agent_runtime', lambda p: Mock())
    monkeypatch.setattr('phase_agent.runtime.agent_web_service.create_control_server', lambda *a, **k: control)
    monkeypatch.setattr('phase_agent.runtime.studio_checkpoint_persistence.install_checkpoint_persistence', lambda: None)
    monkeypatch.setattr('phase_agent.runtime.studio_checkpoint_persistence.flush_checkpoint_persistence', lambda: None)
    monkeypatch.setattr('phase_agent.runtime.studio_run_lifecycle.recover_studio_runs', AsyncMock(return_value=0))
    payload_threads = []
    main_thread = threading.get_ident()
    monkeypatch.setattr(
        'phase_agent.runtime.studio_payloads.install_studio_payload_views',
        lambda: payload_threads.append(threading.get_ident()),
    )

    async def run():
        with blockbuster_ctx(scanned_modules='phase_agent'):
            async with studio_runtime.lifespan(None):
                assert studio_runtime._handler is not None
        assert studio_runtime._handler is None

    asyncio.run(run())
    control.serve_forever.assert_called_once()
    control.shutdown.assert_called_once()
    control.server_close.assert_called_once()
    assert len(payload_threads) == 1 and payload_threads[0] != main_thread

def test_async_sqlite_checkpoint_operations_with_blocking_detector(tmp_path, monkeypatch):
    from langgraph_runtime_inmem import checkpoint as runtime
    from phase_agent.persistence.studio_checkpoint_store import checkpoint_container
    from phase_agent.persistence.studio_storage import install_memory_adapter
    from langgraph.checkpoint.memory import InMemorySaver
    from types import SimpleNamespace
    monkeypatch.chdir(tmp_path)
    fake = SimpleNamespace(InMemorySaver=runtime.InMemorySaver, MEMORY=None)
    monkeypatch.setattr(runtime, 'PersistentDict', checkpoint_container)
    install_memory_adapter(fake)
    saver = fake.InMemorySaver(serde=InMemorySaver().serde)
    monkeypatch.setattr(saver, '_decrypt_json', AsyncMock(side_effect=lambda data, *args: data))
    config = {'configurable': {'thread_id':'synthetic', 'checkpoint_ns':''}}
    cp = {'v':4, 'id':'0001', 'ts':'2026-10-15T00:00:00+00:00',
          'channel_values':{'messages':['saved']}, 'channel_versions':{'messages':1},
          'versions_seen':{}, 'pending_sends':[]}

    async def run():
        with blockbuster_ctx(scanned_modules='phase_agent'):
            saved = await saver.aput(config, cp, {}, {'messages':1})
            await saver.aput_writes(saved, [('fact','yes')], 'task')
            assert (await saver.aget_tuple(config)).checkpoint['channel_values']['messages']==['saved']
            rows = [row async for row in saver.alist(config)]
            assert len(rows)==1 and rows[0].pending_writes[0][2]=='yes'
            await saver.adelete_thread('synthetic')
            assert await saver.aget_tuple(config) is None
    asyncio.run(run())

