import random
import numpy

def create_episode(data, n_way, n_support, n_query):
    class_indices = random.sample(range(len(data.classes)), n_way)
    support = []
    query = []
    for class_idx in class_indices:
        class_items = [i for i, t in enumerate(data.targets) if t == class_idx]
        class_support = random.sample(class_items, n_support)
        class_query = random.sample([i for i in class_items if i not in class_support], n_query)
        support.extend(class_support)
        query.extend(class_query)
    return support, query