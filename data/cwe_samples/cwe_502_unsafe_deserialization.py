import pickle


def load_session(blob):
    session_obj = pickle.loads(blob)
    return session_obj
