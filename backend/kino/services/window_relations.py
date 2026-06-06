from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed


def split_draws(draws):
    half = len(draws) // 2
    return draws[:half], draws[half:]


def select_anchor_numbers(window_numbers, window_size):
    """
    Select:
    - 5 hot numbers
    - 5 cold numbers
    - 10 middle numbers closest to expected count
    """

    expected = window_size * 0.25

    sorted_by_hot = sorted(
        window_numbers,
        key=lambda item: item.count,
        reverse=True,
    )

    sorted_by_cold = sorted(
        window_numbers,
        key=lambda item: item.count,
    )

    hot = sorted_by_hot[:5]
    cold = sorted_by_cold[:5]

    selected_numbers = {item.number for item in hot + cold}

    middle_candidates = [
        item for item in window_numbers
        if item.number not in selected_numbers
    ]

    middle = sorted(
        middle_candidates,
        key=lambda item: abs(item.count - expected),
    )[:10]

    anchors = []

    for item in hot:
        anchors.append({
            "number": item.number,
            "type": "hot",
            "heat": item.count,
        })

    for item in cold:
        anchors.append({
            "number": item.number,
            "type": "cold",
            "heat": item.count,
        })

    for item in middle:
        anchors.append({
            "number": item.number,
            "type": "middle",
            "heat": item.count,
        })

    return anchors


def count_relations_for_anchor(anchor_number, draws):
    """
    Counts how often every other number appeared in the same draw
    as the anchor number.
    """

    counter = Counter()
    anchor_appearances = 0

    for draw in draws:
        numbers = draw.numbers

        if anchor_number not in numbers:
            continue

        anchor_appearances += 1

        for number in numbers:
            if number != anchor_number:
                counter[number] += 1

    return anchor_appearances, counter


def build_anchor_relation(anchor, all_draws, first_half_draws, second_half_draws, top_limit=10):
    anchor_number = anchor["number"]

    total_anchor_hits, total_counter = count_relations_for_anchor(
        anchor_number,
        all_draws,
    )

    first_anchor_hits, first_counter = count_relations_for_anchor(
        anchor_number,
        first_half_draws,
    )

    second_anchor_hits, second_counter = count_relations_for_anchor(
        anchor_number,
        second_half_draws,
    )

    related_numbers = []

    for number, total_count in total_counter.most_common(top_limit):
        first_count = first_counter.get(number, 0)
        second_count = second_counter.get(number, 0)

        related_numbers.append({
            "number": number,
            "total_count": total_count,
            "first_half_count": first_count,
            "second_half_count": second_count,
            "change": second_count - first_count,
        })

    return {
        "anchor_number": anchor_number,
        "anchor_type": anchor["type"],
        "anchor_heat": anchor["heat"],
        "anchor_appearances": total_anchor_hits,
        "first_half_anchor_appearances": first_anchor_hits,
        "second_half_anchor_appearances": second_anchor_hits,
        "related_numbers": related_numbers,
    }


def build_window_relations(analysis, draws, top_limit=10, max_workers=6):
    """
    Build relation analysis for selected anchor numbers.

    We fetch the draws once, then parallelize only the counting work.
    """

    window_numbers = list(analysis.numbers.all())
    anchors = select_anchor_numbers(window_numbers, analysis.window_size)

    first_half_draws, second_half_draws = split_draws(draws)

    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                build_anchor_relation,
                anchor,
                draws,
                first_half_draws,
                second_half_draws,
                top_limit,
            )
            for anchor in anchors
        ]

        for future in as_completed(futures):
            results.append(future.result())

    type_order = {
        "hot": 0,
        "cold": 1,
        "middle": 2,
    }

    results.sort(
        key=lambda item: (
            type_order[item["anchor_type"]],
            -item["anchor_heat"],
            item["anchor_number"],
        )
    )

    return results