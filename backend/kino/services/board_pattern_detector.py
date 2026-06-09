ROWS = {
    row_number: list(range((row_number - 1) * 10 + 1, row_number * 10 + 1))
    for row_number in range(1, 9)
}

COLUMNS = {
    column_number: [
        column_number + (row_index * 10)
        for row_index in range(0, 8)
    ]
    for column_number in range(1, 11)
}


def detect_row_patterns(draw_numbers, threshold=6):
    draw_set = set(draw_numbers)
    events = []

    for row_number, row_numbers in ROWS.items():
        hit_numbers = sorted(draw_set.intersection(row_numbers))

        if len(hit_numbers) >= threshold:
            events.append({
                "pattern_type": "row",
                "group_number": row_number,
                "group_numbers": row_numbers,
                "hit_numbers": hit_numbers,
                "hit_count": len(hit_numbers),
                "threshold": threshold,
            })

    return events


def detect_column_patterns(draw_numbers, threshold=5):
    draw_set = set(draw_numbers)
    events = []

    for column_number, column_numbers in COLUMNS.items():
        hit_numbers = sorted(draw_set.intersection(column_numbers))

        if len(hit_numbers) >= threshold:
            events.append({
                "pattern_type": "column",
                "group_number": column_number,
                "group_numbers": column_numbers,
                "hit_numbers": hit_numbers,
                "hit_count": len(hit_numbers),
                "threshold": threshold,
            })

    return events


def detect_board_patterns(
    draw_numbers,
    row_threshold=6,
    column_threshold=5,
):
    return (
        detect_row_patterns(draw_numbers, threshold=row_threshold)
        + detect_column_patterns(draw_numbers, threshold=column_threshold)
    )