"""Identity-based resources whose lifetime follows a weak-referenceable workspace."""

from weakref import ref


class WorkspaceCache:
    def __init__(self):
        self._entries = {}

    def get_or_create(self, workspace, factory):
        key = id(workspace)
        entry = self._entries.get(key)
        if entry is not None and entry[0]() is workspace:
            return entry[1]
        value = factory()

        def released(owner_ref):
            current = self._entries.get(key)
            if current is not None and current[0] is owner_ref:
                del self._entries[key]

        self._entries[key] = (ref(workspace, released), value)
        return value

    def __getitem__(self, key):
        owner, value = self._entries[key]
        if owner() is None:
            raise KeyError(key)
        return value

    def values(self):
        return tuple(
            value for owner, value in self._entries.values() if owner() is not None
        )

    def __len__(self):
        return len(self._entries)
