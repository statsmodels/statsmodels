def skip_rpy():
    try:
        import rpy
        return False
    except:
        return True
