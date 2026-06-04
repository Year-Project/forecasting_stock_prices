#!/bin/sh
set -e

validate_acl_env() {
  name="$1"
  eval "value=\${$name:-}"

  if [ -z "$value" ]; then
    echo "Redis ACL env var $name is empty. Fill it in root .env." >&2
    exit 1
  fi

  case "$value" in
    *" "*|*"	"*|*"<"*|*">"*)
      echo "Redis ACL env var $name contains whitespace or placeholder brackets. Use a single token without spaces or < >." >&2
      exit 1
      ;;
  esac
}

validate_acl_env REDIS_USER_GUARD
validate_acl_env REDIS_PASSWORD_GUARD
validate_acl_env REDIS_USER_POSTMAN
validate_acl_env REDIS_PASSWORD_POSTMAN
validate_acl_env REDIS_USER_ENVOY
validate_acl_env REDIS_PASSWORD_ENVOY

cat > /usr/local/etc/redis/users.acl <<EOF
user default off
user ${REDIS_USER_GUARD} on >${REDIS_PASSWORD_GUARD} ~guard:* +@all
user ${REDIS_USER_POSTMAN} on >${REDIS_PASSWORD_POSTMAN} ~postman:* +@all
user ${REDIS_USER_ENVOY} on >${REDIS_PASSWORD_ENVOY} ~envoy:* +@all
EOF

exec redis-server /usr/local/etc/redis/redis.conf
