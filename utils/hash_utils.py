import hashlib


def hash_entity(entity: str) -> str:
    return hashlib.sha256(entity.encode()).hexdigest()
