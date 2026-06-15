import logging

from aiokafka.admin import AIOKafkaAdminClient, NewTopic
from aiokafka.errors import TopicAlreadyExistsError

logger = logging.getLogger(__name__)


async def ensure_kafka_topics(
    bootstrap_servers: str,
    security_protocol: str,
    topics: list[str],
    partitions: int = 3,
    replication_factor: int = 3,
) -> None:
    topic_names = sorted({topic for topic in topics if topic})
    if not topic_names:
        return

    admin = AIOKafkaAdminClient(
        bootstrap_servers=bootstrap_servers,
        security_protocol=security_protocol,
    )
    await admin.start()
    try:
        for topic in topic_names:
            try:
                await admin.create_topics(
                    [NewTopic(topic, num_partitions=partitions, replication_factor=replication_factor)]
                )
                logger.info("Created Kafka topic %s", topic)
            except TopicAlreadyExistsError:
                logger.info("Kafka topic %s already exists", topic)
    finally:
        await admin.close()
