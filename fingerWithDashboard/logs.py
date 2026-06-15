import os
import warnings


def hide_logs():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    os.environ["GLOG_minloglevel"] = "3"

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", message="Protobuf gencode version.*")

    try:
        from absl import logging as absl_logging
        absl_logging.set_verbosity(absl_logging.ERROR)
    except Exception:
        pass