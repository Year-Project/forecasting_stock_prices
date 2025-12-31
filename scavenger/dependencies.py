from db.registry import DatabaseRegistry

db_registry = DatabaseRegistry()


def get_session_maker(name: str):
    return db_registry.get(name).sessionmaker
