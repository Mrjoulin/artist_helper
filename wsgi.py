import logging

logging.basicConfig(
    format='[%(asctime)s: %(filename)s:%(lineno)s - %(funcName)10s()]%(levelname)s:%(name)s:%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logging.info('Logging level has been set to %s' % logging.getLogger().getEffectiveLevel())
logging.info('Starting app')

from server.routes import app as application
