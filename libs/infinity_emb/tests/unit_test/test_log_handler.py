from infinity_emb.log_handler import UVICORN_LOG_LEVELS, logger


def test_levels_to_int():
    prev_level = logger.level
    try:
        for level in UVICORN_LOG_LEVELS:
            level_new = level.to_int()
            assert level_new
            logger.setLevel(level_new)
    finally:
        logger.setLevel(prev_level)
