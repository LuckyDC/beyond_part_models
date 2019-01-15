from torch.nn import init


class Initializer:
    def __init__(self, init_func, args=None):
        self.init_func = init_func
        self.args = args if args is not None else dict()

    def __call__(self, param):
        classname = param.__class__.__name__

        # convolution
        if classname.find('Conv') != -1:
            self.init_func(param.weight, **self.args)
            if getattr(param, "bias", None) is not None:
                init.constant_(param.bias, 0.0)

        # linear
        elif classname.find('Linear') != -1:
            self.init_func(param.weight, **self.args)
            if getattr(param, "bias", None) is not None:
                init.constant_(param.bias, 0.0)

        # batch normalization
        elif classname.find('BatchNorm') != -1:
            if param.affine:
                init.constant_(param.weight, 1.0)
                init.constant_(param.bias, 0.0)
