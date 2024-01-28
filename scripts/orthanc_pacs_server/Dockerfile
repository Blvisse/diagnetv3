version: '3.5'

services:
  # Exposed server that's handling incoming web requests for all to see us
  # Underlying image: openresty/openresty:alpine-fat
  ohif_viewer:
    image: blaisse/diagnosoftdicom
    container_name: webappdiagnet

    # volumes:
    #   # Nginx config
    #   - ./config/nginx.conf:/usr/local/openresty/nginx/conf/nginx.conf:ro
    #   # - /usr/local/openresty/nginx/conf/nginx.conf:ro
    # Logs
    # - ./logs/nginx:/var/logs/nginx
    #   # Let's Encrypt
    #   # - letsencrypt_certificates:/etc/letsencrypt
    #   # - letsencrypt_challenges:/var/www/letsencrypt
    ports:
      - '443:443' # SSL
      - '80:80' # Web
    depends_on:
      - orthanc
    restart: on-failure

  # LINK: https://hub.docker.com/r/jodogne/orthanc-plugins/
  # TODO: Update to use Postgres
  # https://github.com/mrts/docker-postgresql-multiple-databases
  orthanc:
    image: jodogne/orthanc-plugins:1.9.7
    hostname: orthanc
    container_name: orthanc

    restart: unless-stopped

    environment:
    #   - ORTHANC__AWS_S3_STORAGE__BUCKET_NAME=brain-scans
      - ORTHANC__AWS_S3_STORAGE__BUCKET_NAME=diagnosoft-pacs-scans
      - ORTHANC__AWS_S3_STORAGE__REGION=us-east-2
    #   ORTHANC__AWS_S3_STORAGE__ACCESS_KEY=AKIARPEY5DNLOXYZDLFZ
      - ORTHANC__AWS_S3_STORAGE__ACCESS_KEY=AKIAQXQV2C3PO6BMIKWE
    #   ORTHANC__AWS_S3_STORAGE__SECRET_KEY=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
      - ORTHANC__AWS_S3_STORAGE__SECRET_KEY=Fid7HmJnNiF/AUgqKYvfCA+b8bxy9FX6xJJUIYXy

      - ORTHANC__AUTHENTICATION_ENABLED=false
      - ORTHANC__DICOM_SERVER_ENABLED=false
