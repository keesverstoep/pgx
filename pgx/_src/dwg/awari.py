from pgx.awari import State as AwariState

# simple initial version based on version for 2048 game

def _make_awari_dwg(dwg, state: AwariState, config):
    GRID_SIZE = config["GRID_SIZE"]
    BOARD_WIDTH = config["BOARD_WIDTH"]
    BOARD_HEIGHT = config["BOARD_HEIGHT"]
    color_set = config["COLOR_SET"]

    # background
    dwg.add(
        dwg.rect(
            (0, 0),
            (BOARD_WIDTH * GRID_SIZE, BOARD_HEIGHT * GRID_SIZE),
            fill=color_set.background_color,
        )
    )

    # board
    # grid
    board_g = dwg.g()
    pit_map = {
        # own stones
        0: (6, 0),
        1: (5, 0),
        2: (4, 0),
        3: (3, 0),
        4: (2, 0),
        5: (1, 0),
        # other stones
        6: (1, 2),
        7: (2, 2),
        8: (3, 2),
        9: (4, 2),
        10: (5, 2),
        11: (6, 2),
        # own home
        17: (0, 1),
        # other home
        23: (7, 1),
    }
    for i, _num in enumerate(state._x.board):
        num = int(_num)
        if i in pit_map.keys():
            x, y = pit_map[i]
            x = x * GRID_SIZE
            y = y * GRID_SIZE
        else:
            continue
        _color = f"{128:02x}"
        board_g.add(
            dwg.rect(
                (x + 2, y + 2),
                (
                    GRID_SIZE - 4,
                    GRID_SIZE - 4,
                ),
                fill=f"#{_color}{_color}{_color}",
                stroke=color_set.text_color,
                stroke_width="0.5px",
                rx="3px",
                ry="3px",
            )
        )

        font_size = 18
        large_num_color = (
            f"#{145:02x}{145:02x}{145:02x}"
        )
        board_g.add(
            dwg.text(
                text=str(num),
                insert=(
                    x + GRID_SIZE / 2 - font_size * len(str(num)) * 0.3,
                    y + GRID_SIZE / 2 + 5,
                ),
                fill=color_set.text_color,
                font_size=f"{font_size}px",
                font_family="Courier",
                font_weight="bold",
            )
        )
    return board_g
