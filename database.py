import sqlite3
from pathlib import Path
from datetime import datetime


# ============================================================
# DATABASE CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATABASE_NAME = BASE_DIR / "chat_history.db"


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_connection():

    return sqlite3.connect(
        str(DATABASE_NAME)
    )


# ============================================================
# CREATE / MIGRATE DATABASE
# ============================================================

def create_database():

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute("""
        PRAGMA foreign_keys = ON
    """)

    cursor.execute("""
        SELECT name
        FROM sqlite_master
        WHERE type='table'
        AND name='conversations'
    """)

    conversations_exists = cursor.fetchone()

    if conversations_exists:

        cursor.execute(
            "PRAGMA table_info(conversations)"
        )

        columns = cursor.fetchall()

        column_names = [
            column[1]
            for column in columns
        ]

        if (
            "user_message" in column_names
            and "ai_response" in column_names
        ):

            cursor.execute("""
                ALTER TABLE conversations
                RENAME TO old_conversations
            """)

    # --------------------------------------------------------
    # Conversations table
    # --------------------------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    # --------------------------------------------------------
    # Messages table
    # --------------------------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL,

            FOREIGN KEY (
                conversation_id
            )
            REFERENCES conversations(id)
            ON DELETE CASCADE
        )
    """)

    connection.commit()

    # --------------------------------------------------------
    # Migrate old database
    # --------------------------------------------------------

    cursor.execute("""
        SELECT name
        FROM sqlite_master
        WHERE type='table'
        AND name='old_conversations'
    """)

    old_table_exists = cursor.fetchone()

    if old_table_exists:

        cursor.execute("""
            SELECT
                user_message,
                ai_response
            FROM old_conversations
            ORDER BY id
        """)

        old_conversations = cursor.fetchall()

        for user_message, ai_response in old_conversations:

            title = user_message.strip()

            if len(title) > 50:

                title = title[:50] + "..."

            created_at = datetime.now().isoformat()

            cursor.execute("""
                INSERT INTO conversations
                (title, created_at)
                VALUES (?, ?)
            """, (
                title,
                created_at
            ))

            conversation_id = cursor.lastrowid

            cursor.execute("""
                INSERT INTO messages
                (
                    conversation_id,
                    role,
                    content,
                    created_at
                )
                VALUES (?, ?, ?, ?)
            """, (
                conversation_id,
                "user",
                user_message,
                created_at
            ))

            cursor.execute("""
                INSERT INTO messages
                (
                    conversation_id,
                    role,
                    content,
                    created_at
                )
                VALUES (?, ?, ?, ?)
            """, (
                conversation_id,
                "assistant",
                ai_response,
                created_at
            ))

        cursor.execute("""
            DROP TABLE old_conversations
        """)

    connection.commit()

    connection.close()


# ============================================================
# CREATE NEW CONVERSATION
# ============================================================

def create_conversation(title):

    connection = get_connection()

    cursor = connection.cursor()

    created_at = datetime.now().isoformat()

    cursor.execute("""
        INSERT INTO conversations
        (title, created_at)
        VALUES (?, ?)
    """, (
        title,
        created_at
    ))

    conversation_id = cursor.lastrowid

    connection.commit()

    connection.close()

    return conversation_id


# ============================================================
# SAVE MESSAGE
# ============================================================

def save_message(
    conversation_id,
    role,
    content
):

    connection = get_connection()

    cursor = connection.cursor()

    created_at = datetime.now().isoformat()

    cursor.execute("""
        INSERT INTO messages
        (
            conversation_id,
            role,
            content,
            created_at
        )
        VALUES (?, ?, ?, ?)
    """, (
        conversation_id,
        role,
        content,
        created_at
    ))

    connection.commit()

    connection.close()


# ============================================================
# GET ALL CONVERSATIONS
# ============================================================

def get_conversations():

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute("""
        SELECT
            id,
            title,
            created_at
        FROM conversations
        ORDER BY id DESC
    """)

    conversations = cursor.fetchall()

    connection.close()

    return conversations


# ============================================================
# SEARCH CONVERSATIONS
# ============================================================

def search_conversations(search_text):

    connection = get_connection()

    cursor = connection.cursor()

    search_pattern = f"%{search_text}%"

    cursor.execute("""
        SELECT
            id,
            title,
            created_at
        FROM conversations
        WHERE title LIKE ?
        ORDER BY id DESC
    """, (
        search_pattern,
    ))

    conversations = cursor.fetchall()

    connection.close()

    return conversations


# ============================================================
# GET ONE CONVERSATION
# ============================================================

def get_conversation(
    conversation_id
):

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute("""
        SELECT
            id,
            title,
            created_at
        FROM conversations
        WHERE id = ?
    """, (
        conversation_id,
    ))

    conversation = cursor.fetchone()

    connection.close()

    return conversation


# ============================================================
# GET MESSAGES
# ============================================================

def get_messages(
    conversation_id
):

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute("""
        SELECT
            role,
            content
        FROM messages
        WHERE conversation_id = ?
        ORDER BY id ASC
    """, (
        conversation_id,
    ))

    messages = cursor.fetchall()

    connection.close()

    return messages


# ============================================================
# DELETE CONVERSATION
# ============================================================

def delete_conversation(
    conversation_id
):

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute("""
        DELETE FROM messages
        WHERE conversation_id = ?
    """, (
        conversation_id,
    ))

    cursor.execute("""
        DELETE FROM conversations
        WHERE id = ?
    """, (
        conversation_id,
    ))

    connection.commit()

    connection.close()


# ============================================================
# CLEAR ALL CONVERSATIONS
# ============================================================

def clear_conversations():

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute("""
        DELETE FROM messages
    """)

    cursor.execute("""
        DELETE FROM conversations
    """)

    connection.commit()

    connection.close()


# ============================================================
# COUNT CONVERSATIONS
# ============================================================

def count_conversations():

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute("""
        SELECT COUNT(*)
        FROM conversations
    """)

    count = cursor.fetchone()[0]

    connection.close()

    return count