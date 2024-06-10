
def check(condition, exception, string):
    if not condition:
        raise exception(string)

def argmax_max(iterable):
    return max(enumerate(iterable), key=lambda x: x[1])

def argmin_min(iterable):
    return min(enumerate(iterable), key=lambda x: x[1])
