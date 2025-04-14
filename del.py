import sqlite3

# Подключаемся к базе данных
conn = sqlite3.connect("users_data.db")
cursor = conn.cursor()

# Удаляем все строки из таблицы
cursor.execute("DELETE FROM user_progress")

# Фиксируем изменения
conn.commit()

# Закрываем соединение
conn.close()

print("Все данные из таблицы user_progress удалены.")
