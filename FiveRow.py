import logging
import random
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Константы
BOARD_SIZE = 15
EMPTY = 0
BLACK = 1  # Игрок
WHITE = 2  # ИИ
MAX_DEPTH = 3  # Глубина поиска для ИИ

# Веса для различных паттернов
PATTERN_WEIGHTS = {
    'five': 100000,  # 5 в ряд
    'open_four': 10000,  # открытая четверка
    'four': 1000,  # четверка
    'open_three': 1000,  # открытая тройка
    'three': 100,  # тройка
    'open_two': 10,  # открытая двойка
    'two': 1  # двойка
}

# Хранилище игр
games = {}


def init_board():
    return [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]


def check_win(board, row, col, player):
    # Проверка победы для указанного игрока
    directions = [
        [(0, 1), (0, -1)],  # горизонталь
        [(1, 0), (-1, 0)],  # вертикаль
        [(1, 1), (-1, -1)],  # диагонал \
        [(1, -1), (-1, 1)]  # диагонал /
    ]

    for dir_pair in directions:
        count = 1
        for dr, dc in dir_pair:
            r, c = row + dr, col + dc
            while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r][c] == player:
                count += 1
                r += dr
                c += dc
        if count >= 5:
            return True
    return False


def is_valid_move(board, row, col):
    # Проверка допустимости хода
    return (0 <= row < BOARD_SIZE and
            0 <= col < BOARD_SIZE and
            board[row][col] == EMPTY)


def get_empty_cells_around(board, center_row, center_col, radius=2):
    # Получить пустые клетки вокруг указанной позиции
    cells = []
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            r, c = center_row + dr, center_col + dc
            if (0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and
                    board[r][c] == EMPTY and (dr != 0 or dc != 0)):
                cells.append((r, c))
    return cells


def evaluate_position(board, player):
    # Оценка позиции для указанного игрока
    score = 0
    opponent = WHITE if player == BLACK else BLACK

    # Проверяем все возможные линии длиной 5
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            # Горизонталь
            if col <= BOARD_SIZE - 5:
                line = [board[row][col + i] for i in range(5)]
                score += evaluate_line(line, player, opponent)

            # Вертикаль
            if row <= BOARD_SIZE - 5:
                line = [board[row + i][col] for i in range(5)]
                score += evaluate_line(line, player, opponent)

            # Диагонал \
            if row <= BOARD_SIZE - 5 and col <= BOARD_SIZE - 5:
                line = [board[row + i][col + i] for i in range(5)]
                score += evaluate_line(line, player, opponent)

            # Диагонал /
            if row <= BOARD_SIZE - 5 and col >= 4:
                line = [board[row + i][col - i] for i in range(5)]
                score += evaluate_line(line, player, opponent)

    return score


def evaluate_line(line, player, opponent):
    # Оценка линии из 5 клеток
    player_count = line.count(player)
    opponent_count = line.count(opponent)

    if opponent_count > 0 and player_count > 0:
        return 0  # Смешанная линия

    if player_count == 5:
        return PATTERN_WEIGHTS['five']
    elif player_count == 4 and line.count(EMPTY) == 1:
        return PATTERN_WEIGHTS['four']
    elif player_count == 3 and line.count(EMPTY) == 2:
        return PATTERN_WEIGHTS['three']
    elif player_count == 2 and line.count(EMPTY) == 3:
        return PATTERN_WEIGHTS['two']

    if opponent_count == 5:
        return -PATTERN_WEIGHTS['five']
    elif opponent_count == 4 and line.count(EMPTY) == 1:
        return -PATTERN_WEIGHTS['four']
    elif opponent_count == 3 and line.count(EMPTY) == 2:
        return -PATTERN_WEIGHTS['three']
    elif opponent_count == 2 and line.count(EMPTY) == 3:
        return -PATTERN_WEIGHTS['two']

    return 0


def minimax(board, depth, alpha, beta, maximizing_player, last_move=None):
    # Алгоритм минимакс с альфа-бета отсечением
    # Проверка терминальных состояний
    if last_move and check_win(board, last_move[0], last_move[1], WHITE if maximizing_player else BLACK):
        return (None, 100000) if maximizing_player else (None, -100000)

    if depth == 0:
        score = evaluate_position(board, WHITE) - evaluate_position(board, BLACK)
        return (None, score)

    # Получаем возможные ходы вокруг последнего хода
    if last_move:
        possible_moves = get_empty_cells_around(board, last_move[0], last_move[1])
    else:
        # Если нет последнего хода, рассматриваем все пустые клетки рядом с камнями
        possible_moves = []
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if board[row][col] != EMPTY:
                    possible_moves.extend(get_empty_cells_around(board, row, col))

        if not possible_moves:  # Если доска пустая
            possible_moves = [(BOARD_SIZE // 2, BOARD_SIZE // 2)]

    if not possible_moves:
        return (None, 0)

    if maximizing_player:
        max_eval = float('-inf')
        best_move = None

        for move in possible_moves:
            row, col = move
            board[row][col] = WHITE
            evaluation = minimax(board, depth - 1, alpha, beta, False, (row, col))[1]
            board[row][col] = EMPTY

            if evaluation > max_eval:
                max_eval = evaluation
                best_move = move

            alpha = max(alpha, evaluation)
            if beta <= alpha:
                break

        return best_move, max_eval
    else:
        min_eval = float('inf')
        best_move = None

        for move in possible_moves:
            row, col = move
            board[row][col] = BLACK
            evaluation = minimax(board, depth - 1, alpha, beta, True, (row, col))[1]
            board[row][col] = EMPTY

            if evaluation < min_eval:
                min_eval = evaluation
                best_move = move

            beta = min(beta, evaluation)
            if beta <= alpha:
                break

        return best_move, min_eval


def ai_move(board, last_move):
    # Ход ИИ
    # Сначала проверяем выигрышные ходы
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if board[row][col] == EMPTY:
                # Проверяем, может ли ИИ выиграть
                board[row][col] = WHITE
                if check_win(board, row, col, WHITE):
                    board[row][col] = EMPTY
                    return (row, col)
                board[row][col] = EMPTY

                # Проверяем, нужно ли блокировать игрока
                board[row][col] = BLACK
                if check_win(board, row, col, BLACK):
                    board[row][col] = EMPTY
                    return (row, col)
                board[row][col] = EMPTY

    # Используем минимакс для поиска лучшего хода
    move, score = minimax(board, MAX_DEPTH, float('-inf'), float('inf'), True, last_move)

    if move is None:
        # Если минимакс не нашел ход, выбираем случайный из возможных
        if last_move:
            possible_moves = get_empty_cells_around(board, last_move[0], last_move[1])
        else:
            possible_moves = [(BOARD_SIZE // 2, BOARD_SIZE // 2)]

        if possible_moves:
            move = random.choice(possible_moves)
        else:
            # Если нет возможных ходов, ищем любую пустую клетку
            for row in range(BOARD_SIZE):
                for col in range(BOARD_SIZE):
                    if board[row][col] == EMPTY:
                        return (row, col)

    return move


def board_to_str(board):
    # Преобразование доски в строку
    symbols = {EMPTY: '⬜', BLACK: '⚫', WHITE: '⚪'}
    header = "    " + "   ".join([f"{i:2}" for i in range(BOARD_SIZE)])
    rows = []
    for i, row in enumerate(board):
        row_str = f"{i:2} " + "".join([symbols[cell] for cell in row])
        rows.append(row_str)
    return header + "\n" + "\n".join(rows)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Обработчик команды /start
    await update.message.reply_text(
        "🎯 Добро пожаловать в Рэндзю!\n\n"
        "Правила:\n"
        "• Вы играете черными (⚫), ИИ - белыми (⚪)\n"
        "• Игроки ходят по очереди, ставя камни на доску\n"
        "• Побеждает тот, кто первым соберет 5 камней в ряд\n"
        "• Ряд может быть горизонтальным, вертикальным или диагональным\n\n"
        "Команды:\n"
        "/new - начать новую игру\n"
        "/rules - показать правила\n"
        "Ход указывается в формате: 'x y' (например, '7 7')"
    )


async def show_rules(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Обработчик команды /rules
    await update.message.reply_text(
        "📋 Правила Рэндзю:\n\n"
        "1. Игра ведется на доске 15×15\n"
        "2. Игрок (⚫) ходит первым\n"
        "3. ИИ (⚪) ходит вторым\n"
        "4. Камни ставятся на пересечения линий\n"
        "5. Побеждает тот, кто первым построит непрерывный ряд из 5 своих камней\n"
        "6. Ряд может быть горизонтальным, вертикальным или диагональным\n"
        "7. Игра заканчивается при построении ряда из 5 камней или при заполнении доски"
    )


async def new_game(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Начало новой игры
    chat_id = update.effective_chat.id
    games[chat_id] = {
        'board': init_board(),
        'last_move': None,
        'game_over': False
    }
    await update.message.reply_text(
        "🆕 Новая игра началась! Ваш ход (черные ⚫).\n"
        "Отправьте координаты в формате 'x y' (от 0 до 14).\n"
        "Рекомендуемый первый ход: '7 7' (центр доски)\n\n" +
        board_to_str(games[chat_id]['board'])
    )


async def make_move(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Обработка хода игрока
    chat_id = update.effective_chat.id
    if chat_id not in games:
        await update.message.reply_text("Начните новую игру командой /new")
        return

    game = games[chat_id]
    if game['game_over']:
        await update.message.reply_text("Игра завершена. Начните новую командой /new")
        return

    try:
        x, y = map(int, update.message.text.split())
        if not (0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE):
            raise ValueError("Координаты вне диапазона")
    except Exception as e:
        await update.message.reply_text("❌ Ошибка! Используйте формат: 'x y' (например, '7 7')")
        return

    if not is_valid_move(game['board'], y, x):
        await update.message.reply_text("❌ Эта клетка уже занята или недоступна!")
        return

    # Ход игрока
    game['board'][y][x] = BLACK
    game['last_move'] = (y, x)

    # Проверка победы игрока
    if check_win(game['board'], y, x, BLACK):
        game['game_over'] = True
        await update.message.reply_text(
            "🎉 Поздравляю! Вы победили!\n\n" +
            board_to_str(game['board']) +
            "\n\nНачните новую игру: /new"
        )
        return

    # Ход ИИ
    await update.message.reply_text("🤔 ИИ думает...")
    ai_row, ai_col = ai_move(game['board'], game['last_move'])
    game['board'][ai_row][ai_col] = WHITE
    game['last_move'] = (ai_row, ai_col)

    # Проверка победы ИИ
    if check_win(game['board'], ai_row, ai_col, WHITE):
        game['game_over'] = True
        await update.message.reply_text(
            "🤖 ИИ победил! Попробуйте еще раз!\n\n" +
            board_to_str(game['board']) +
            "\n\nНачните новую игру: /new"
        )
        return

    # Продолжение игры
    await update.message.reply_text(
        f"🤖 ИИ походил в позицию ({ai_col}, {ai_row})\n"
        f"Ваш ход (черные ⚫):\n\n" +
        board_to_str(game['board'])
    )


def main():
    # Основная функция
    # Замените 'YOUR_BOT_TOKEN' на токен вашего бота
    application = Application.builder().token("YOUR_BOT_TOKEN").build()

    # Регистрация обработчиков
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("new", new_game))
    application.add_handler(CommandHandler("rules", show_rules))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, make_move))

    # Запуск бота
    application.run_polling()


if __name__ == '__main__':
    main()
