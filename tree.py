from entropy import get_entropy, information_gain
from schema import branch, feature, feature_IG, features_branches


def piles(features: list[feature], labels: list[str], indices: list[int]):
    parsed_features: list[features_branches] = []
    for feature in features:
        branches: list[branch] = []
        total = len(indices)
        values = []

        for i in indices:
            if feature["array"][i] not in values:
                values.append(feature["array"][i])

        for value in values:
            member_indices = [i for i in indices if feature["array"][i] == value]
            yes_no_arr = [labels[i] for i in member_indices]

            branches.append(
                {
                    "value": value,
                    "numerator": len(member_indices),
                    "yes_no_arr": yes_no_arr,
                    "indices": member_indices,
                }
            )

        parsed_features.append(
            {"name": feature["name"], "total": total, "branches": branches}
        )

    return parsed_features


def majority(labels: list[str]):
    return max(set(labels), key=labels.count)


def build_tree(indices: list[int], features: list[feature], labels: list[str]):
    subset = [labels[i] for i in indices]
    entropy = get_entropy(subset)

    if entropy == 0:
        return subset[0]
    if not features:
        return majority(subset)

    parsed_features = piles(features, labels, indices)

    features_IG: list[feature_IG] = []
    for f in parsed_features:
        IG = information_gain(entropy, f["branches"], f["total"])
        features_IG.append({"name": f["name"], "IG": IG})

    next_node = max(features_IG, key=lambda ig: ig["IG"])
    next_node_feature = next(
        f for f in parsed_features if f["name"] == next_node["name"]
    )

    remaining = [f for f in features if f["name"] != next_node["name"]]

    tree = {"feature": next_node["name"], "branches": {}}
    for branch in next_node_feature["branches"]:
        tree["branches"][branch["value"]] = build_tree(
            branch["indices"], remaining, labels
        )

    return tree


def model(features: list[feature], labels: list[str]):
    indices = list(range(len(labels)))
    tree = build_tree(indices, features, labels)
    return tree


def predict(tree, test):
    node = tree
    while not isinstance(node, str):
        feature = node["feature"]
        value = test[feature]
        node = node["branches"][value]
    return node
