from dependencies.dependencies import get_session_maker


def get_auth_session_maker():
    return get_session_maker("auth")
