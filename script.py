def test_var_args(f_arg, *args):
    print("first normal arg:", f_arg)
    if args:
        print(args[0])
    else:
        print(0)

test_var_args('yasoob',10)