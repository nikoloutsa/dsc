def get_datasets(name, **data_args):
    if name == 'cifar10':
        from .cifar10 import get_datasets
        return get_datasets(**data_args)
    else:
        raise ValueError('Dataset %s unknown' % name)
