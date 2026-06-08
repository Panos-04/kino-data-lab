def number_to_position(number):
    """
    Converts KINO number to board position.
    1  -> row 1, col 1
    10 -> row 1, col 10
    11 -> row 2, col 1
    80 -> row 8, col 10
    """
    row = (number - 1) // 10 + 1
    col = (number - 1) % 10 + 1
    return row, col


def position_to_number(row, col):
    if row < 1 or row > 8:
        return None

    if col < 1 or col > 10:
        return None

    return (row - 1) * 10 + col


def build_shape_numbers(center_row, center_col, offsets):
    numbers = []

    for row_offset, col_offset in offsets:
        number = position_to_number(
            center_row + row_offset,
            center_col + col_offset,
        )

        if number is not None:
            numbers.append(number)

    return numbers


SHAPE_TEMPLATES = {
    "cross": [
        (0, 0),
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
    ],

    "box_2x2": [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ],

    "l_shape": [
        (0, 0),
        (1, 0),
        (2, 0),
        (2, 1),
    ],

    "vertical_4": [
        (0, 0),
        (1, 0),
        (2, 0),
        (3, 0),
    ],

    "horizontal_4": [
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 3),
    ],

    "diagonal_down_4": [
        (0, 0),
        (1, 1),
        (2, 2),
        (3, 3),
    ],

    "diagonal_up_4": [
        (0, 0),
        (-1, 1),
        (-2, 2),
        (-3, 3),
    ],
}


def detect_shape(draw_numbers, shape_name, min_hits=None):
    if shape_name not in SHAPE_TEMPLATES:
        raise ValueError(f"Unknown shape: {shape_name}")

    offsets = SHAPE_TEMPLATES[shape_name]
    shape_size = len(offsets)

    if min_hits is None:
        min_hits = shape_size

    draw_set = set(draw_numbers)
    detected = []

    for center_number in range(1, 81):
        center_row, center_col = number_to_position(center_number)
        shape_numbers = build_shape_numbers(center_row, center_col, offsets)

        # Skip incomplete shapes near board edges
        if len(shape_numbers) < shape_size:
            continue

        hit_numbers = sorted(draw_set.intersection(shape_numbers))

        if len(hit_numbers) >= min_hits:
            detected.append({
                "shape": shape_name,
                "center_number": center_number,
                "center_row": center_row,
                "center_col": center_col,
                "shape_numbers": shape_numbers,
                "hit_numbers": hit_numbers,
                "hit_count": len(hit_numbers),
                "shape_size": shape_size,
            })

    return detected


def detect_all_shapes(draw_numbers, min_hits_by_shape=None):
    if min_hits_by_shape is None:
        min_hits_by_shape = {
            "cross": 4,
            "box_2x2": 4,
            "l_shape": 4,
            "vertical_4": 4,
            "horizontal_4": 4,
            "diagonal_down_4": 4,
            "diagonal_up_4": 4,
        }

    all_detected = []

    for shape_name in SHAPE_TEMPLATES.keys():
        min_hits = min_hits_by_shape.get(shape_name)

        all_detected.extend(
            detect_shape(
                draw_numbers=draw_numbers,
                shape_name=shape_name,
                min_hits=min_hits,
            )
        )

    return all_detected