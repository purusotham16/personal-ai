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

    connection = sqlite3.connect(
        str(DATABASE_NAME)
    )

    # Enable foreign keys
    connection.execute(
        "PRAGMA foreign_keys = ON"
    )

    return connection


# ============================================================
# CREATE MEMORY TABLE
# ============================================================

def create_memory_table():

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS memories (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            memory TEXT NOT NULL,

            category TEXT NOT NULL,

            created_at TEXT NOT NULL,

            UNIQUE(memory)

        )
    """)

    connection.commit()

    connection.close()


# ============================================================
# SAVE MEMORY
# ============================================================

def save_memory(
    memory,
    category="general"
):

    if not memory:

        return None

    memory = memory.strip()

    category = category.strip()

    if not memory:

        return None

    if not category:

        category = "general"


    connection = get_connection()

    cursor = connection.cursor()

    # --------------------------------------------------------
    # Check duplicate
    # --------------------------------------------------------

    cursor.execute("""
        SELECT id
        FROM memories
        WHERE LOWER(memory) = LOWER(?)
    """, (
        memory,
    ))

    existing_memory = cursor.fetchone()


    if existing_memory:

        connection.close()

        return existing_memory[0]


    # --------------------------------------------------------
    # Save new memory
    # --------------------------------------------------------

    created_at = datetime.now().isoformat()

    cursor.execute("""
        INSERT INTO memories
        (
            memory,
            category,
            created_at
        )
        VALUES (?, ?, ?)
    """, (
        memory,
        category,
        created_at
    ))

    memory_id = cursor.lastrowid

    connection.commit()

    connection.close()

    return memory_id


# ============================================================
# GET ALL MEMORIES
# ============================================================

def get_memories():

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute("""
        SELECT
            id,
            memory,
            category,
            created_at
        FROM memories
        ORDER BY id DESC
    """)

    memories = cursor.fetchall()

    connection.close()

    return memories


# ============================================================
# SEARCH MEMORIES
# ============================================================

def search_memories(
    query,
    limit=5
):

    if not query:

        return []

    query_words = [
        word.lower()
        for word in query.split()
        if len(word) >= 2
    ]

    if not query_words:

        return []


    connection = get_connection()

    cursor = connection.cursor()


    # --------------------------------------------------------
    # Get all memories
    # --------------------------------------------------------

    cursor.execute("""
        SELECT
            id,
            memory,
            category,
            created_at
        FROM memories
        ORDER BY id DESC
    """)

    memories = cursor.fetchall()

    connection.close()


    # --------------------------------------------------------
    # Simple relevance scoring
    # --------------------------------------------------------

    scored_memories = []


    for memory in memories:

        memory_id = memory[0]

        memory_text = memory[1]

        category = memory[2]

        created_at = memory[3]


        memory_lower = memory_text.lower()

        score = 0


        for word in query_words:

            if word in memory_lower:

                score += 1


        if score > 0:

            scored_memories.append(
                (
                    score,
                    memory_id,
                    memory_text,
                    category,
                    created_at
                )
            )


    # --------------------------------------------------------
    # Sort by relevance
    # --------------------------------------------------------

    scored_memories.sort(
        key=lambda item: item[0],
        reverse=True
    )


    # --------------------------------------------------------
    # Return results
    # --------------------------------------------------------

    return [
        (
            memory_id,
            memory_text,
            category,
            created_at
        )

        for (
            score,
            memory_id,
            memory_text,
            category,
            created_at
        )

        in scored_memories[:limit]
    ]


# ============================================================
# GET MEMORY CONTEXT
# ============================================================

def get_memory_context(
    query=None,
    limit=5
):

    # --------------------------------------------------------
    # If query is provided, search relevant memories
    # --------------------------------------------------------

    if query:

        memories = search_memories(
            query,
            limit=limit
        )

    else:

        memories = get_memories()[:limit]


    if not memories:

        return ""


    context_parts = []


    for (
        memory_id,
        memory_text,
        category,
        created_at
    ) in memories:

        context_parts.append(
            f"- {memory_text}"
        )


    return "\n".join(
        context_parts
    )


# ============================================================
# GET ONE MEMORY
# ============================================================

def get_memory(
    memory_id
):

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute("""
        SELECT
            id,
            memory,
            category,
            created_at
        FROM memories
        WHERE id = ?
    """, (
        memory_id,
    ))

    memory = cursor.fetchone()

    connection.close()

    return memory


# ============================================================
# DELETE ONE MEMORY
# ============================================================

def delete_memory(
    memory_id
):

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute("""
        DELETE FROM memories
        WHERE id = ?
    """, (
        memory_id,
    ))

    connection.commit()

    deleted = cursor.rowcount

    connection.close()

    return deleted > 0


# ============================================================
# CLEAR ALL MEMORIES
# ============================================================

def clear_memories():

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute("""
        DELETE FROM memories
    """)

    connection.commit()

    connection.close()


# ============================================================
# COUNT MEMORIES
# ============================================================

def count_memories():

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute("""
        SELECT COUNT(*)
        FROM memories
    """)

    count = cursor.fetchone()[0]

    connection.close()

    return count


# ============================================================
# INITIALIZE MEMORY DATABASE
# ============================================================

create_memory_table()