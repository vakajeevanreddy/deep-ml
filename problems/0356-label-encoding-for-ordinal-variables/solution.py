def label_encode_ordinal(values, order):
    rank_map = {category: idx for idx, category in enumerate(order)}
    return [rank_map.get(value, -1) for value in values]