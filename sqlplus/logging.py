import logging
import coloredlogs


coloredlogs.DEFAULT_FIELD_STYLES['levelname']['color'] = 'cyan'
coloredlogs.install(
    fmt='%(asctime)s %(levelname)s %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger(__name__)
