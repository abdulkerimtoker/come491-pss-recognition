python -m celery -A pss_recognition worker -l INFO --concurrency=1 --without-gossip --without-mingle --without-heartbeat -Ofair --pool=solo