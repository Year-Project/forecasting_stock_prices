from db.registry import DatabaseRegistry

db_registry = DatabaseRegistry()


def get_session_maker(name: str):
    return db_registry.get(name).sessionmaker


def get_forecast_requests_session_maker():
    return get_session_maker("forecast_requests")
