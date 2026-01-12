#!/bin/bash
set -e

cat > /usr/local/etc/redis/users.acl <<EOF
user ${REDIS_USER_GUARD} on >${REDIS_PASSWORD_GUARD} ~guard:* +@all
user ${REDIS_USER_POSTMAN} on >${REDIS_PASSWORD_POSTMAN} ~postman:* +@all
user ${REDIS_USER_ENVOY} on >${REDIS_PASSWORD_ENVOY} ~envoy:* +@all
EOF

exec redis-server /usr/local/etc/redis/redis.conf