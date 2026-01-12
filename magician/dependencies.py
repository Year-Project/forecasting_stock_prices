from magician.services.scavenger_client import ScavengerClient

scavenger_client: ScavengerClient | None = None


def get_scavenger_client() -> ScavengerClient:
    if scavenger_client is None:
        raise RuntimeError("Scavenger client is not configured")
    return scavenger_client


def set_scavenger_client(client: ScavengerClient) -> None:
    global scavenger_client
    scavenger_client = client
