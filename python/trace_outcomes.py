"""Shared ray-tracing outcome constants and labels."""

EXIT_CODE = 0
BOUNCED_BACK_CODE = 1
ON_SENSOR_CODE = 2
BOUNCE_LIMIT_CODE = 3
UNKNOWN_CODE = 4
MIRROR_CODE = 10

EXIT_LABEL = 'exit'
BOUNCED_BACK_LABEL = 'bounced back'
ON_SENSOR_LABEL = 'on sensor'
BOUNCE_LIMIT_LABEL = 'bounce limit'
UNKNOWN_LABEL = 'unknown'
MIRROR_LABEL = 'mirror'

EXIT_COLOR = {
    EXIT_LABEL: 'b',
    BOUNCED_BACK_LABEL: 'r',
    BOUNCE_LIMIT_LABEL: 'm',
    ON_SENSOR_LABEL: 'g',
    UNKNOWN_LABEL: 'k',
}
