"""Compressed Studio recovery, shared views and change-triggered file flushing.

Pending namespaces and exact parent references are protected from age limits.
"""

import asyncio
from contextlib import contextmanager
from functools import wraps
from threading import RLock

from phase_agent.persistence.compact_checkpoints import CompressedSerde


class StudioStorage:
    def __init__(self, memory):
        self.memory = memory
        self.lock = RLock()
        self.revision = 0
        self.pruned = 0
        self.readers = 0
        self.unavailable_parent_refs = set()
        self.saved = {}
        for name in ("storage", "writes", "blobs"):
            store = getattr(memory, name)
            self.saved[name] = 0
            if hasattr(store, "sync"):
                original = store.sync

                def sync(name=name, original=original, store=store):
                    with self.lock:
                        if self.readers:
                            return
                        if self.saved[name] != self.revision:
                            if self.pruned != self.revision and not getattr(
                                store, "incremental", False
                            ):
                                self.prune()
                                self.pruned = self.revision
                            original()
                            self.saved[name] = self.revision

                store.sync = sync

    def changed(self):
        self.revision += 1

    @contextmanager
    def reading(self):
        with self.lock:
            self.readers += 1
        try:
            yield
        finally:
            with self.lock:
                self.readers -= 1

    def prune(self, keep=8, namespaces_to_keep=32):
        """Retain recent checkpoints plus explicitly referenced recovery IDs."""
        memory = self.memory
        retained = set()
        for thread, namespaces in memory.storage.items():
            recent = sorted(
                (ns for ns in namespaces if ns and namespaces[ns]),
                key=lambda ns: max(namespaces[ns]),
                reverse=True,
            )[:namespaces_to_keep]
            for namespace, entries in namespaces.items():
                if not entries:
                    continue
                latest = max(entries)
                data = memory.serde.loads_typed(entries[latest][0])
                pending = bool(data.get("pending_sends")) or any(
                    row[1] in {"__interrupt__", "__pregel_tasks"}
                    for row in memory.writes.get((thread, namespace, latest), {}).values()
                )
                for channel, version in data["channel_versions"].items():
                    if channel.startswith("branch:to:"):
                        typed = memory.blobs.get((thread, namespace, channel, version))
                        if typed is not None and typed[0] != "empty":
                            pending = True
                if namespace and namespace not in recent and not pending:
                    continue
                retained.update((thread, namespace, cid) for cid in sorted(entries)[-keep:])
        # Metadata parents bind nested subgraphs to exact checkpoint IDs.
        queue = list(retained)
        while queue:
            thread, namespace, cid = queue.pop()
            entry = memory.storage[thread][namespace].get(cid)
            if entry is None:
                raise ValueError("Studio recovery references a missing checkpoint")
            metadata = memory.serde.loads_typed(entry[1])
            for ns, parent in (metadata.get("parents") or {}).items():
                key = (thread, ns, parent)
                if parent not in memory.storage.get(thread, {}).get(ns, {}):
                    # Non-persisted dialogue parents still appear in tracing
                    # metadata. Preserve that metadata, never invent a recovery
                    # point or create an empty namespace during collection.
                    self.unavailable_parent_refs.add(key)
                    continue
                if key not in retained:
                    retained.add(key)
                    queue.append(key)
        used_blobs = set()
        removed = 0
        for thread, namespaces in memory.storage.items():
            for namespace, entries in namespaces.items():
                for cid in list(entries):
                    if (thread, namespace, cid) not in retained:
                        del entries[cid]
                        removed += 1
                        continue
                    checkpoint, metadata, parent = entries[cid]
                    if parent is not None and (thread, namespace, parent) not in retained:
                        entries[cid] = (checkpoint, metadata, None)
                    data = memory.serde.loads_typed(checkpoint)
                    used_blobs.update(
                        (thread, namespace, channel, version)
                        for channel, version in data["channel_versions"].items()
                    )
        for key in list(memory.writes):
            if key not in retained:
                del memory.writes[key]
        for key in list(memory.blobs):
            if key not in used_blobs:
                del memory.blobs[key]
        for namespaces in memory.storage.values():
            for ns in list(namespaces):
                if not namespaces[ns]:
                    del namespaces[ns]
        return removed


def install_memory_adapter(checkpoint):
    """Patch this process's dev saver; never edit third-party installations."""
    original = checkpoint.InMemorySaver
    if getattr(original, "_phase_compact_storage", False):
        return
    from langgraph.checkpoint.memory import InMemorySaver as BaseSaver

    class SharedSaver(original):
        _phase_compact_storage = True

        def __init__(self, *, serde=None, __persistence_hook__=None, **kwargs):
            memory = checkpoint.MEMORY
            if memory is None:
                super().__init__(serde=serde, __persistence_hook__=__persistence_hook__, **kwargs)
                self._phase_storage = StudioStorage(self)
            else:
                # The upstream factory builds unpack-hook views by loading all
                # three files and then discarding them. Share them immediately.
                BaseSaver.__init__(self, serde=serde)
                self.filename = memory.filename
                self.latest_iter = None
                self.storage, self.writes, self.blobs = (
                    memory.storage,
                    memory.writes,
                    memory.blobs,
                )
                self._phase_storage = memory._phase_storage
            self.serde = CompressedSerde(self.serde)

        def put(self, *args, **kwargs):
            with self._phase_storage.lock:
                result = super().put(*args, **kwargs)
                self._phase_storage.changed()
                return result

        def get_tuple(self, *args, **kwargs):
            with self._phase_storage.lock:
                return super().get_tuple(*args, **kwargs)

        def list(self, *args, **kwargs):
            with self._phase_storage.reading():
                yield from super().list(*args, **kwargs)

        def get_delta_channel_history(self, *args, **kwargs):
            with self._phase_storage.lock:
                return super().get_delta_channel_history(*args, **kwargs)

        def put_writes(self, *args, **kwargs):
            with self._phase_storage.lock:
                result = super().put_writes(*args, **kwargs)
                self._phase_storage.changed()
                return result

        async def aget_tuple(self, config):
            result = await asyncio.to_thread(self.get_tuple, config)
            if result is None:
                return None
            # Retain the native API's async metadata decryption.
            decrypt = getattr(self, "_decrypt_json", None)
            if decrypt is not None:
                metadata = await decrypt(result.metadata, "run", "metadata")
                result = result._replace(metadata=metadata)
            return result

        async def alist(self, *args, **kwargs):
            iterator = self.list(*args, **kwargs)
            sentinel = object()
            try:
                while True:
                    item = await asyncio.to_thread(next, iterator, sentinel)
                    if item is sentinel:
                        break
                    decrypt = getattr(self, "_decrypt_json", None)
                    if decrypt is not None:
                        metadata = await decrypt(item.metadata, "run", "metadata")
                        item = item._replace(metadata=metadata)
                    yield item
            finally:
                await asyncio.to_thread(iterator.close)

        async def aput(self, *args, **kwargs):
            return await asyncio.to_thread(self.put, *args, **kwargs)

        async def aput_writes(self, *args, **kwargs):
            return await asyncio.to_thread(self.put_writes, *args, **kwargs)

        async def adelete_thread(self, *args, **kwargs):
            return await asyncio.to_thread(self.delete_thread, *args, **kwargs)

        async def aget_delta_channel_history(self, *args, **kwargs):
            return await asyncio.to_thread(self.get_delta_channel_history, *args, **kwargs)

        def delete_thread(self, *args, **kwargs):
            with self._phase_storage.lock:
                result = super().delete_thread(*args, **kwargs)
                self._phase_storage.changed()
                return result

    checkpoint.InMemorySaver = SharedSaver
    if checkpoint.MEMORY is not None:
        memory = checkpoint.MEMORY
        memory.__class__ = SharedSaver
        memory.serde = CompressedSerde(memory.serde)
        memory._phase_storage = StudioStorage(memory)


def install_mutation_adapters(checkpoint, ops):
    """The dev API copies/deletes threads by editing dictionaries directly."""

    def guarded(original):
        if getattr(original, "_phase_tracked_mutation", False):
            return original

        @wraps(original)
        async def mutate(*args, **kwargs):
            memory = checkpoint.MEMORY
            if memory is None:
                return await original(*args, **kwargs)
            manager = memory._phase_storage
            stores = {name: getattr(memory, name) for name in ("storage", "writes", "blobs")}
            with manager.reading():
                try:
                    return await original(*args, **kwargs)
                finally:
                    with manager.lock:
                        # Upstream deletion replaces writes with a defaultdict.
                        # Keep the live PersistentDict (and all existing views).
                        for name, store in stores.items():
                            replacement = getattr(memory, name)
                            if replacement is not store:
                                if hasattr(store, "replace_from"):
                                    store.replace_from(replacement)
                                else:
                                    store.clear()
                                    store.update(replacement)
                                setattr(memory, name, store)
                        manager.changed()

        mutate._phase_tracked_mutation = True
        return mutate

    from phase_agent.persistence.studio_checkpoint_store import StudioCheckpointDict

    replace = getattr(ops, "_replace_thread_id", None)
    if replace is not None and not getattr(replace, "_phase_lazy_mapping", False):

        @wraps(replace)
        def replace_mapping(data, *args, **kwargs):
            if isinstance(data, StudioCheckpointDict):
                data = data.materialize()
            return replace(data, *args, **kwargs)

        replace_mapping._phase_lazy_mapping = True
        ops._replace_thread_id = replace_mapping

    ops.Threads.copy = staticmethod(guarded(ops.Threads.copy))
    original_delete_checkpoints = ops._delete_checkpoints_for_thread
    if not getattr(original_delete_checkpoints, "_phase_async_checkpoint_delete", False):

        @wraps(original_delete_checkpoints)
        async def delete_checkpoints(thread_id, conn, run_id=None):
            from langgraph_api import config as api_config

            if api_config.USE_CUSTOM_CHECKPOINTER:
                return await original_delete_checkpoints(thread_id, conn, run_id)

            def erase():
                memory = checkpoint.MEMORY
                if memory is None:
                    memory = ops.Checkpointer()
                manager = memory._phase_storage
                with manager.lock, manager.reading():
                    tid = str(thread_id)
                    if not run_id:
                        # The saver clears this thread's storage, writes and blobs.
                        memory.delete_thread(tid)
                    elif tid in memory.storage:
                        # Preserve the native run rollback's metadata selection.
                        for namespace, entries in list(memory.storage[tid].items()):
                            for identity, (_, metadata, _) in list(entries.items()):
                                if memory.serde.loads_typed(metadata).get("run_id") == str(run_id):
                                    del entries[identity]
                            if not entries:
                                del memory.storage[tid][namespace]
                    manager.changed()

            return await asyncio.to_thread(erase)

        delete_checkpoints._phase_async_checkpoint_delete = True
        ops._delete_checkpoints_for_thread = delete_checkpoints

    original_delete = getattr(ops.Threads, "delete", None)
    if original_delete is not None and not getattr(original_delete, "_phase_serial_delete", False):
        delete_lock = asyncio.Lock()

        @wraps(original_delete)
        async def delete_thread(*args, **kwargs):
            # Native delete retains a list index across checkpoint deletion's await.
            async with delete_lock:
                return await original_delete(*args, **kwargs)

        delete_thread._phase_serial_delete = True
        ops.Threads.delete = staticmethod(delete_thread)
