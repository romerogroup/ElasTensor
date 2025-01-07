def chain(*iterators, sort=False):

    for iterator in iterators:
        for element in iterator:
            if sort:
                yield type(element)(sorted(element))
            else:
                yield indices
