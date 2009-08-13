def check_rpy_installation():
    try:
        import rpy
        return False
    except:
        return True

def skip_rpy():
    return check_rpy_installation()
