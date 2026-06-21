import math

from schema import branch


def get_entropy(yes_no_arr: list[str]):
    p_yes = 0
    p_no = 0

    for i in range(0, len(yes_no_arr), 1):
        if yes_no_arr[i] == 'Yes':
            p_yes += 1
        elif yes_no_arr[i] == 'No':
            p_no += 1
        else:
            raise ValueError("Array has one or many words that aren't 'Yes' or 'No'")

    total = p_yes + p_no
    if total == 0:
        return 0.0

    prob_yes = p_yes / total
    prob_no = p_no / total

    entropy = 0.0
    if prob_yes > 0:
        entropy -= prob_yes * math.log2(prob_yes)
    if prob_no > 0:
        entropy -= prob_no * math.log2(prob_no)

    return entropy


def information_gain(before: float, yes_no_arrs: list[branch], total: int):
    after = 0

    for i in yes_no_arrs:
        entropy = get_entropy(i["yes_no_arr"])
        after += (i["numerator"] / total) * entropy

    return before - after
