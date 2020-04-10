import sollya

enable_trace = False

def trace_print(*args):
    global enable_trace
    if enable_trace: print(*args)

def tracer_a0(func):
    def my_func():
        result = func()
        trace_print("[TRACE] {}() = \n\t\t{}".format(func.__code__.co_name, result))
        return result
    return my_func
def tracer_a1(func):
    def my_func(x0):
        result = func(x0)
        trace_print("[TRACE] {}({}) =\n\t\t{}".format(func.__code__.co_name, x0, result))
        return result
    return my_func
def tracer_a2(func):
    def my_func(x0, x1):
        args = x0, x1
        result = func(*args)
        trace_print("[TRACE] {}({}) = \n\t\t{}".format(func.__code__.co_name, ", \n\t\t\t".join(str(s) for s in args), result))
        return result
    return my_func
def tracer_a3(func):
    def my_func(x0, x1, x2):
        args = x0, x1, x2
        result = func(*args)
        trace_print("[TRACE] {}({}) = \n\t\t{}".format(func.__code__.co_name, ",\n\t\t\t".join(str(s) for s in args), result))
        return result
    return my_func
def tracer_a4(func):
    def my_func(x0, x1, x2, x3):
        args = x0, x1, x2, x3
        result = func(*args)
        trace_print("[TRACE] {}({}) = \n\t\t{}".format(func.__code__.co_name, ", \n\t\t\t".join(str(s) for s in args), result))
        return result
    return my_func


def function_trace(arity):
    return {
        0: tracer_a0,
        1: tracer_a1,
        2: tracer_a2,
        3: tracer_a3,
        4: tracer_a4,
    }[arity]

def function_trace_hex_2(func):
    def my_func(x0, x1):
        save = sollya.settings.display
        sollya.settings.display = sollya.hexadecimal
        args = x0, x1
        result = func(*args)
        prolog = "[TRACE] {}({}) = ".format(func.__code__.co_name, ", ".join(str(s) for s in args))
        sollya.settings.display = save
        epilog = str(result)
        trace_print(prolog + epilog)
        return result
    return my_func
