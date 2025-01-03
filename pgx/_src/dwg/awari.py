from pgx.awari import State as AwariState

# simple initial version based on version for 2048 game

def _make_awari_dwg(dwg, state: AwariState, config):
    GRID_SIZE = config["GRID_SIZE"]
    BOARD_SIZE = config["BOARD_WIDTH"]
    color_set = config["COLOR_SET"]

    # background
    dwg.add(
        dwg.rect(
            (0, 0),
            (BOARD_SIZE * GRID_SIZE, BOARD_SIZE * GRID_SIZE),
            fill=color_set.background_color,
        )
    )

    # board
    # grid
    board_g = dwg.g()
    for i, _num in enumerate(state._x.board):
        num = int(_num)
        x = (i % 6) * GRID_SIZE
        y = (i // 6) * GRID_SIZE
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
