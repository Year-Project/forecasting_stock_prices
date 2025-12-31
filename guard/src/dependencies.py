from dependencies.dependencies import get_session_maker
from guard.src.kafka.admin_secret_producer import AdminSecretProducer

admin_secret_producer: AdminSecretProducer | None = None


def get_auth_session_maker():
    return get_session_maker("auth")


def get_admin_secret_producer() -> AdminSecretProducer | None:
    return admin_secret_producer


def set_admin_secret_producer(producer: AdminSecretProducer):
    global admin_secret_producer
    admin_secret_producer = producer
