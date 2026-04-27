class MyMessage(object):
    """message type definition"""

    # server to client
    MSG_TYPE_S2C_INIT_CONFIG = 1
    MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT = 2

    # client to server
    MSG_TYPE_C2S_SEND_RESULT_TO_SERVER = 3

    MSG_ARG_KEY_TYPE = "msg_type"
    MSG_ARG_KEY_SENDER = "sender"
    MSG_ARG_KEY_RECEIVER = "receiver"

    # payload
    MSG_ARG_KEY_NUM_SAMPLES = "num_samples"
    MSG_ARG_KEY_MODEL_PARAMS = "model_params"
    MSG_ARG_KEY_HN_PARAMS = "hn_params"
    MSG_ARG_KEY_EMB_PARAMS = "emb_params"
    MSG_ARG_KEY_CLIENT_INDEX = "client_idx"

    MSG_ARG_KEY_ROUND_IDX = "round_idx"
    MSG_ARG_KEY_TRAIN_HN = "train_hn"
    MSG_ARG_KEY_SYNC_W = "sync_w"
    MSG_ARG_KEY_SYNC_HN = "sync_hn"

    MSG_ARG_KEY_SERVER_DENSITY = "server_density"
    MSG_ARG_KEY_DEVICE_DENSITY = "device_density"
