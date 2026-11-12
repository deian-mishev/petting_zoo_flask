FROM quay.io/keycloak/keycloak:26.8.2

ENV KC_DB=dev-file
ENV KC_HEALTH_ENABLED=true

RUN /opt/keycloak/bin/kc.sh build

COPY --chown=keycloak:keycloak petting-zoo-realm.json \
    /opt/keycloak/data/import/petting-zoo-realm.json

CMD ["start", "--optimized", "--import-realm", "--http-enabled=true", "--cache=local"]